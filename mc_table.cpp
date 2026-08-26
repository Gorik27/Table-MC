#include <mpi.h>
#include <iostream>
#include <fstream>
#include <format>
#include <cmath>
#include <string>
#include <cassert>
#include <random>
#include <array>
#include <iomanip>
#include <filesystem>
#include "progress_bar.hpp"
#include <argparse/argparse.hpp>
#include "matrix.hpp"
#include "MpiDataLoader.h"
#include "PartitionLoader.h"
#include <algorithm>
#include "Communicator.h"
#include <unordered_set>


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    PartitionLoader partition;
    partition.load(std::format("partitions/{}.txt", world_rank+1));

    int z_max = 30;
    MpiDataLoader loader(z_max*2+1);
    try {
        loader.loadAndDistribute("new_neighbors.txt", "new_eint.txt", partition.partition, partition.nbrs);

    } catch (const std::exception& e) {
        std::cerr << "Rank " << world_rank << " поймал исключение: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int cols = partition.nsites;
    int cols_ghost = loader.local_ind.size();
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
        .default_value(10)
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
    std::uniform_int_distribution<int> uniform_type(0, n_types-1);

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
        std::cout << "       rows               : " << rows << std::endl;
        std::cout << "       total site types   : " << loader.total_site_types << std::endl;
        std::cout << "       MC steps           : " << mc_steps << std::endl;
        std::cout << "       types              : " << n_types << std::endl;
        std::cout << "       mu                 : " << mu_str << std::endl;
        if (is_vcsgc){
        std::cout << "===== VCSGC ensemble is used ======" << std::endl;
        std::cout << "       kappa              : " << kappa << std::endl;
        std::cout << "       target c           : " << c_str << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }

    
    std::uniform_int_distribution<int> uniform_row(0, rows-1);

    int natoms = rows*loader.total_site_types;

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
    Matrix<int> m = Matrix(rows, cols_ghost, types[0]); 
    // energy matrix
    Matrix<double> es = Matrix(cols, n_types, 0.0);
    for (int i = 0; i<cols; ++i){
        for (int j = 1; j<n_types; ++j){
            es(i, j) = dist(gen_shared)*std_energy[j]+mean_energy[j]; // TODO: load from file
        }
    }
    // interaction matrix
    std::vector<std::unique_ptr<Matrix<double>>> interactions;
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            interactions.push_back(std::make_unique<Matrix<double>>(cols, z_max, 0.0));
            int index = I*(n_types-1)+J;
            for (int i = 0; i<cols; ++i){
                for (int j = i+1; j<z_max; ++j){
                    (*interactions[index])(i, j) = loader.local_eint[i, j];      // TODO: сделать поддержку нескольких сортов атомов
                }
            }
            
        }
    } 
        
    if (world_rank==0){
        es.save_to_text("es.txt"); // TODO: убрать
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
    number_of_solutes[0] = natoms;
    
    std::vector<int> number_of_solutes_loc(n_types, 0);
    number_of_solutes_loc[0] = rows*cols;

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
            return 1; 
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
            type_new = types[uniform_type(gen)];
        }

        double dE = es(j, type_new) - es(j, type_old);
        int index_old, index_new;
        for (int k = 0; k<loader.local_z[j]; k++){ // over neighbors of j
            int jk = loader.getNbrLocalIndex(j, k); 
            if (type_old > 0 && m(i, jk) > 0){
                index_old = (type_old-1)*(n_types-1)+m(i, jk)-1;   
                dE -= (*interactions[index_old])(j, k);
            }
            if (type_new > 0 && m(i, jk) > 0){
                index_new = (type_new-1)*(n_types-1)+m(i, jk)-1;
                dE += (*interactions[index_new])(j, k);
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
                dF_glob += kappa*dN_tot[k]*(dN_tot[k] + 2*(number_of_solutes[k]-number_of_solutes_target[k]))/natoms;
            }
            double prob_glob = std::exp(-dF_glob/kT);
            double p_glob = uniform(gen_shared);
            acceptance_flag = (p_glob<prob_glob);
        }
        else {
            acceptance_flag = true;
        }

        if (p<prob && acceptance_flag){
            m(i, j) = type_new;
            accepted_loc ++;
            number_of_solutes_loc[type_new]++;
            number_of_solutes_loc[type_old]--;
            energy_loc += dE;
        }
        
        std::vector<int> target_ranks;
        std::unordered_map<int, int> block_ind;
        std::unordered_set<int> seen;
        int map_cnt = 0; 
        for (int j = 0; j<cols; j++){
                for (int k = 0; k<partition.z[j]; k++){ // over interblock bonds of j
                    // найдем нужный rank=block and nbr_id
                    int block  = partition.getNbrBlock(j, k);
                    if (!seen.count(block)) {
                        seen.insert(block);
                        target_ranks.push_back(block-1);
                        block_ind[block] = map_cnt;
                        map_cnt ++;
                    }
                }
            }
        int num_partners = target_ranks.size();
        std::vector<std::vector<int>> my_requests(num_partners);
        for (int j = 0; j<cols; j++){
                for (int k = 0; k<partition.z[j]; k++){ 
                    int block  = partition.getNbrBlock(j, k);
                    int nbr_id = partition.getNbrID(j, k);
                    my_requests[block_ind[block]].push_back(nbr_id);
                }
            }
        MatrixExchanger<int> exchanger(world_rank, world_size);
        exchanger.initialize_connections(m, target_ranks, my_requests, loader.local_ind);

        if (acceptance_flag){
            for (int k = 0; k<n_types; k++){
                number_of_solutes[k] += dN_tot[k];
            }
        }

        exchanger.exchange_step(m);

        // 3. Читаем полученные столбцы целиком
        for (int p = 0; p < num_partners; ++p) {
            // Сколько столбцов мы просили у p-го соседа?
            int num_requested_cols = my_requests[p].size();

            for (int k = 0; k < num_requested_cols; ++k) {
                int original_key_id = my_requests[p][k]; 
                // Получаем прямой указатель на k-й запрошенный столбец от p-го соседа
                const int* column_data = exchanger.get_received_column(p, k);
                
                // Теперь column_data — это обычный непрерывный массив
                for (int r = 0; r < rows; ++r) {
                    int val = column_data[r]; // Это элемент строки 'r' чужого столбца
                    // Использовать val...
                    m(r, loader.local_ind[original_key_id]) = val;
                }
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
            // TODO переписать со сбора по строкам (REDUCE) на сбор столбцов (GATHER)

            /* Matrix<double> mr(n_types, cols, 0.0); 
            for (int ii = 0; ii<rows; ii++){
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
            }*/
        } 
    }

    if (world_rank==0){
        out.close();
    }

    MPI_Finalize();

    return 0;
}
