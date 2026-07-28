#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <cassert>
#include "matrix.hpp"
#include <random>
#include <array>
#include <iomanip>
#include <filesystem>
#include "progress_bar.hpp"
#include <argparse/argparse.hpp>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Matrix<int> N_mask; 
    N_mask.load_from_text("N_mask.txt");

    int cols = N_mask.rows();
    int rows;
    int mc_steps;

    unsigned int shared_seed = 0;
    std::mt19937 gen_shared(shared_seed);

    unsigned int seed = world_rank;
    std::mt19937 gen(seed);

    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> uniform_col(0, cols-1);


    argparse::ArgumentParser program("mc_table");

    program.add_argument("-r", "--rows")
        .help("number of rows")
        .scan<'i', int>()
        .default_value(world_size)
        .store_into(rows);

    program.add_argument("-s", "--steps")
        .help("number of MC steps")
        .scan<'i', int>()
        .default_value(100)
        .store_into(mc_steps);

    double kT;
    program.add_argument("-T")
        .help("temperature in [kT]")
        .scan<'g', double>()
        .default_value(3.0)
        .store_into(kT);

    double kappa;
    program.add_argument("-k", "--kappa")
        .help("kappa")
        .scan<'g', double>()
        .default_value(0.0)
        .store_into(kappa);

    program.add_argument("-m", "--mu")
        .help("chemical potentials")
        .nargs(argparse::nargs_pattern::at_least_one)
        .scan<'g', double>()
        .default_value(std::vector<double>{0.0, 0.0});
        
    program.add_argument("-c", "--conc")
        .help("target concentrations")
        .nargs(argparse::nargs_pattern::at_least_one)
        .scan<'g', double>()
        .default_value(std::vector<double>{0.5});

    try {
    program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        if (world_rank==0){
            std::cerr << err.what() << std::endl;
            std::cerr << program;
        }
        return 1;
    }

    bool is_vcsgc = program.is_used("--kappa");

    std::vector<double> concentrations_target = program.get<std::vector<double>>("--conc");
    std::string c_str = "";
    for (double val : concentrations_target) {
        std::string s = std::to_string(val);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos); // Удаляем нули на конце
        if (s.back() == '.') s.pop_back();                       // Удаляем точку, если число целое
        c_str += s + " ";
    }

    std::vector<double> mu = program.get<std::vector<double>>("--mu");
    const int n_types = mu.size();
    std::string mu_str = "";
    for (double val : mu) {
        std::string s = std::to_string(val);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos); // Удаляем нули на конце
        if (s.back() == '.') s.pop_back();                       // Удаляем точку, если число целое
        mu_str += s + " ";
    }

    int dump_each = 10000;
    int print_each = 100;

    const std::vector<int> types = {0, 1, 2};    

    std::vector<double> mean_energy = {0.0, -20.0, -10.0};
    std::vector<double> std_energy = {0.0, 10.0, 10.0};

    std::vector<double> mean_interaction = {-7.0, 0.0, -1.0, 0.0}; // BB BC CC CB
    std::vector<double> std_interaction = {2.5, 1.0, 1.0, 1.0}; // BB BC CC CB

    if (world_rank==0){
        std::cout << "===================================" << std::endl;
        std::cout << "===== Input parameters ============" << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "       rows     : " << rows << std::endl;
        std::cout << "       cols     : " << cols << std::endl;
        std::cout << "       MC steps : " << mc_steps << std::endl;
        std::cout << "       types    : " << n_types << std::endl;
        std::cout << "       mu       : " << mu_str << std::endl;
        if (is_vcsgc){
        std::cout << "===== VCSGC ensemble is used ======" << std::endl;
        std::cout << "       kappa    : " << kappa << std::endl;
        std::cout << "       target c : " << c_str << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }

    int rows_in_domain = rows/world_size;
    std::uniform_int_distribution<int> uniform_row(0, rows_in_domain-1);
    rows = rows_in_domain*world_size;
    if (world_rank == 0) {
        std::cout << "Number of rows is set to " 
                    << rows << std::endl;
    }

    int natoms = rows*cols;
    std::vector<int> number_of_solutes_target(n_types, 0);
    double c0 = 1.0;
    for (int k = 1; k<n_types; k++){
        number_of_solutes_target[k] = static_cast<int>(concentrations_target[k-1]*natoms);
        c0 = c0 - concentrations_target[k-1];
    }
    if (c0<0.0){
        std::cerr << "[ERROR]: Total solute concentration exeeds 100%!!!!" << std::endl;
        return 1;
    }
    number_of_solutes_target[0] = static_cast<int>(c0*natoms);

    // occupation matrix
    Matrix<double> x_glob;
    if (world_rank==0){
        x_glob = Matrix(n_types, cols, 0.0); 
    }
    Matrix<int> m = Matrix(rows_in_domain, cols, types[0]); 
    // energy matrix
    Matrix<double> es = Matrix(cols, n_types, 0.0);
    for (int i = 0; i<cols; ++i){
        for (int j = 1; j<n_types; ++j){
            es(i, j) = dist(gen_shared)*std_energy[j]+mean_energy[j];
        }
    }
    // interaction matrix
    std::vector<std::unique_ptr<Matrix<double>>> interactions;
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            interactions.push_back(std::make_unique<Matrix<double>>(cols, cols, 0.0));
            int index = I*(n_types-1)+J;
            if (world_rank==0){
                std::cout << "interaction: " << index << " mean: " << mean_interaction[index] << " std: " << std_interaction[index] << std::endl;  
            }
            for (int i = 0; i<cols; ++i){
                for (int j = i+1; j<cols; ++j){
                    if (N_mask(i, j)!=0)
                    {
                        (*interactions[index])(i, j) = dist(gen_shared)*std_interaction[index]+mean_interaction[index];
                        (*interactions[index])(j, i) = (*interactions[index])(i, j);
                    }
                    
                }
            }
            
        }
    }
        
    if (world_rank==0){
        es.save_to_text("es.txt");
        for (int I = 0; I<n_types-1; ++I){
            for (int J = 0; J<n_types-1; ++J){
                interactions[I*(n_types-1)+J]->save_to_text("w"+std::to_string(10*(I+1)+J+1)+".txt");
            }
        }
    }

    int accepted;
    double energy;
    if (world_rank==0){
        accepted = 0;
        energy = 0.0;
    }
    int accepted_loc = 0;
    double energy_loc = 0.0;

    std::vector<int> number_of_solutes(n_types, 0);
    number_of_solutes[0] = rows*cols;
    
    std::vector<int> number_of_solutes_loc(n_types, 0);
    number_of_solutes_loc[0] = rows_in_domain*cols;

    std::ofstream out;
    std::string dump_dir = "dump";
    if (world_rank==0){

        try {
            std::uintmax_t deleted_count = std::filesystem::remove_all(dump_dir);
        } 
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "[ERROR]: error during removing of dump folder: " << e.what() << std::endl;
        }

        std::filesystem::create_directory(dump_dir); 
        out.open("mc_output.txt");
        if (!out.is_open()) {
            std::cerr << "[ERROR]: error during creating of mc_output.txt file:" << std::endl;
            return 1; // или другой способ обработки ошибки
        }
        out << "mc_output" << std::endl;
        out << std::left; 
        out << std::setw(14) << "step" 
            << std::setw(12) << "acc";
        for (int k = 0; k < n_types; k++) {
            out << std::setw(12) << ("X_" + std::to_string(k));
        }
        out << std::setw(15) << "energy" << std::endl;
    }
    
    ProgressBar bar(mc_steps, (world_rank == 0));

    for (int step = 1; step <= mc_steps; step++){
        int i = uniform_row(gen);
        int j = uniform_col(gen);
        int type_old = m(i, j);
        int type_new = type_old;
        while (type_new == type_old){
            type_new = types[uniform_col(gen)%n_types];//TODO make specific generator
        }

        double dE = es(j, type_new) - es(j, type_old);
        int index_old, index_new;
        for (int k = 0; k<cols; k++){ 
            if (N_mask(j, k)!=0) // neighbors
            {
                if (type_old > 0 && m(i, k) > 0){
                    index_old = (type_old-1)*(n_types-1)+m(i, k)-1;   
                    dE -= (*interactions[index_old])(j, k);
                }
                if (type_new > 0 && m(i, k) > 0){
                    index_new = (type_new-1)*(n_types-1)+m(i, k)-1;
                    dE += (*interactions[index_new])(j, k);
                }
            }
        }

        double dF = dE + mu[type_new] - mu[type_old];

        double prob = std::exp(-dF/kT);
        double p = uniform(gen);

        std::vector<int> dN_tot(n_types, 0);
        std::vector<int> dN(n_types, 0);
        if (is_vcsgc){
            if (p<prob){
                dN[type_new] = +1;
                dN[type_old] = -1;
            }
            MPI_Allreduce(dN.data(), dN_tot.data(), n_types, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        bool acceptance_flag = false;
        
        if (is_vcsgc){
            double dF_glob = 0;
            for (int k = 1; k<n_types; k++){
                dF_glob += kappa*dN_tot[k]*(dN_tot[k] + 2*(number_of_solutes[k]-number_of_solutes_target[k]));
            }
            double prob_glob = std::exp(-dF_glob/kT);
            double p_glob = uniform(gen_shared);
            acceptance_flag = (p_glob<prob_glob);
            /*   if (world_rank==0){
                out << dF_glob << " " <<  prob_glob << " " << acceptance_flag << std::endl;
            } */
        }
        else {
            acceptance_flag = true;
        }

        if (p<prob && acceptance_flag){
            /* if (world_rank==0){
                out << "acc " << dN_tot[1] << std::endl;
            } */
            m(i, j) = type_new;
            accepted_loc ++;
            number_of_solutes_loc[type_new]++;
            number_of_solutes_loc[type_old]--;
            energy_loc += dE;
        }
        

        if (acceptance_flag){
            for (int k = 0; k<n_types; k++){
                number_of_solutes[k] += dN_tot[k];
            }
        }

        if (step%print_each==0 || step == mc_steps)
        {
                    MPI_Reduce(
                        &accepted_loc,      
                        &accepted,        
                        1,
                        MPI_INT,
                        MPI_SUM,
                        0,
                        MPI_COMM_WORLD
                    );

                    MPI_Reduce(
                        &energy_loc,      
                        &energy,        
                        1,
                        MPI_DOUBLE,
                        MPI_SUM,
                        0,
                        MPI_COMM_WORLD
                    );

            if (world_rank == 0) 
            {
                bar.update(step); 
                //thermo
                double acc = static_cast<double>(accepted)/dump_each/world_size;
                out << std::left;
                out << std::setw(14) << step;
                out << std::setw(12) << std::fixed << std::setprecision(6) << acc;
                for (int k = 0; k<n_types; k++){
                    out << std::setw(12) << std::fixed << std::setprecision(4) << static_cast<double>(number_of_solutes[k])/natoms;
                }
                out << std::setw(15) << std::fixed << std::setprecision(4) << energy << std::endl;
            }
            accepted_loc = 0;
        }

        if (step%dump_each==0)
        {
            Matrix<double> mr(n_types, cols, 0.0);
            for (int ii = 0; ii<rows_in_domain; ii++){
                for (int jj = 0; jj<cols; jj++){
                    mr(m(ii, jj), jj)++;
                }
            }

            MPI_Reduce(
                mr.data(),      
                x_glob.data(),        
                cols*n_types,
                MPI_DOUBLE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
            );

            if (world_rank == 0) 
            {
                for (int k = 0; k<n_types; k++){
                    for (int p = 0; p<cols; p++){
                        x_glob(k, p) = x_glob(k, p)/rows;
                    }
                }
                // dump
                x_glob.save_to_text(dump_dir+"/x_"+std::to_string(step)+".txt");
            }
        }
    }

    if (world_rank==0){
        out.close();
    }

    MPI_Finalize();

    return 0;
}
