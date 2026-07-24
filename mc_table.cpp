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

int main(int argc, char* argv[]) {
    std::vector<double> mu = {0.0, 10.0, 1.0};
    double kT = 3.0;

    std::vector<double> mean_energy = {0.0, -20.0};
    std::vector<double> std_energy = {0.0, 10.0};

    std::vector<double> mean_interaction = {-7.0, 0.0, -1.0, 0.0}; // BB BC CC CB
    std::vector<double> std_interaction = {2.5, 1.0, 1.0, 1.0}; // BB BC CC CB
    
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Matrix<int> N_mask; 
    N_mask.load_from_text("N_mask.txt");

    int cols = N_mask.rows();
    int rows = world_size;

    unsigned int shared_seed = 0;
    std::mt19937 gen_shared(shared_seed);

    unsigned int seed = world_rank;
    std::mt19937 gen(seed);

    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> uniform_col(0, cols-1);

    int mc_steps = 0;

    if (argc > 1) 
    {
        try 
        {
            rows = std::stoi(argv[1]);
        } 
        catch (const std::exception& e) 
        {
            if (world_rank == 0) 
            {
                std::cerr << "[ERROR] wrong argument for number of rows: " << argv[1] << std::endl;
                std::cout << "[INFO] Input number of rows is set to Nprocs: "  << rows << std::endl;
            }
        }

        if (argc > 2)
        {
            try 
            {
                mc_steps = std::stoi(argv[2]);
            } 
            catch (const std::exception& e) 
            {
                if (world_rank == 0) 
                {
                    std::cerr << "[ERROR] wrong argument for number of mc_steps: " << argv[2] << std::endl;
                    std::cout << "[INFO] Number of mc_steps is set to default: "  << mc_steps << std::endl;
                }
            }
        }
    } 
    else 
    {
        if (world_rank == 0) 
        {
            std::cout << "[INFO] Input number of rows is set to Nprocs: " << rows << std::endl;
            std::cout << "[INFO] Number of mc_steps is set to default: "  << mc_steps << std::endl;
        }
    }

    int rows_in_domain = rows/world_size;
    std::uniform_int_distribution<int> uniform_row(0, rows_in_domain-1);
    rows = rows_in_domain*world_size;
    if (world_rank == 0) {
        std::cout << "Number of rows is set to " 
                    << rows << std::endl;
    }
    

    int I = world_rank*rows_in_domain; // row index of domain
    
    // occupation matrix
    Matrix<double> x_glob;
    if (world_rank==0){
        x_glob = Matrix(1, cols, 0.0); 
    }
    Matrix<int> m = Matrix(rows_in_domain, cols, 0); 
    // energy matrix
    const int n_types = mean_energy.size();
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
    

    const std::vector<int> types = {0, 1};

    int accepted;
    double energy;
    std::vector<int> number_of_solutes;
    if (world_rank==0){
        accepted = 0;
        energy = 0.0;
        number_of_solutes = {0};
        number_of_solutes[0] = rows*cols;
    }
    int accepted_loc = 0;
    double energy_loc = 0.0;
    
    std::vector<int> number_of_solutes_loc = {0};
    number_of_solutes_loc[0] = rows_in_domain*cols;

    int dump_each = 10000;
    int print_each = 100;

    std::ofstream out;
    std::string dump_dir = "dump";
    if (world_rank==0){
        std::filesystem::create_directory(dump_dir); 
        out.open("mc_output.txt");
        if (!out.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл для записи!" << std::endl;
            return 1; // или другой способ обработки ошибки
        }
        out << "mc_output" << std::endl;
        out << std::left; 
        out << std::setw(10) << "step" 
            << std::setw(12) << "acc";
        for (int k = 0; k < types.size(); k++) {
            out << std::setw(10) << ("N" + std::to_string(k));
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
            type_new = types[uniform_col(gen)%n_types];
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
        if (p<prob){
            m(i, j) = type_new;
            accepted_loc ++;
            number_of_solutes_loc[type_new]++;
            number_of_solutes_loc[type_old]--;
            energy_loc = energy_loc + dE;
        }

        if (step%print_each==0 || step == mc_steps)
        {
                    MPI_Reduce(
                        number_of_solutes_loc.data(),      
                        number_of_solutes.data(),        
                        n_types,
                        MPI_INT,
                        MPI_SUM,
                        0,
                        MPI_COMM_WORLD
                    );

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
                out << std::setw(10) << step;
                out << std::setw(12) << std::fixed << std::setprecision(6) << acc;
                for (int k = 0; k<types.size(); k++){
                    out << std::setw(10) << number_of_solutes[k];
                }
                out << std::setw(15) << std::fixed << std::setprecision(4) << energy << std::endl;
            }
            accepted_loc = 0;
        }

        if (step%dump_each==0)
        {
            Matrix<int> mr_int = m.sum_axis_0();
            Matrix<double> mr_double(1, cols);
            for (int k = 0; k < cols; k++) {
                mr_double(0, k) = static_cast<double>(mr_int(0, k));
            }

            MPI_Reduce(
                mr_double.data(),      
                x_glob.data(),        
                cols,
                MPI_DOUBLE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
            );

            if (world_rank == 0) 
            {
                for (int k = 0; k<cols; k++){
                    x_glob(0, k) = x_glob(0, k)/rows;
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
