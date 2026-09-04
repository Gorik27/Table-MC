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
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <numeric> 
#include <limits.h>
#include <filesystem>

std::string dump_dir = "dump";
std::string restart_dir = "restart";


void signal_handler(int signal) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "!!! Процесс " << rank << " упал с сигналом " << signal << " (PID: " << getpid() << ")" << std::endl;
    exit(signal);
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char buffer[PATH_MAX];
    // Читаем символическую ссылку на текущий исполняемый файл
    ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
    std::filesystem::path exePath;
    std::filesystem::path exeDir;
    if (count != -1) {
        buffer[count] = '\0'; // Добавляем нулевой символ в конец строки
        exePath = buffer;
        exeDir = exePath.parent_path();
    }

    int python_status = 0;
    if (world_rank==0){
        const char* env_p = std::getenv("MC_PYTHON_PATH");
        std::string python_path;
        if (env_p == nullptr) {
            python_path = "python";
        }
        else {
            python_path = env_p;
        }
        std::filesystem::path script_path = exeDir / "clustering_nbrs.py";
        std::string command = "\"" + python_path + "\" \"" + script_path.string() + "\" " + std::to_string(world_size);

        std::cout << "Running: " << command << std::endl;
        int result = std::system(command.c_str());
        if (result != 0) {
            std::cerr << "ERROR in python clustering_nbrs.py (error code " << result << ")" << std::endl;
            python_status = 1;
        }
    }
    MPI_Bcast(&python_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (python_status != 0) {
        MPI_Finalize();
        return 2; 
    }
    MPI_Barrier(MPI_COMM_WORLD);

    argparse::ArgumentParser program("mc_table");

    program.add_argument("-r", "--restart")
           .help("load restart files")
           .implicit_value(true) 
           .default_value(false);
    
    int rows;
    program.add_argument("-n", "--rows")
        .help("number of rows")
        .scan<'i', int>()
        .default_value(10)
        .store_into(rows);
    
    size_t mc_steps;
    program.add_argument("-s", "--steps")
        .help("number of MC steps")
        .scan<'u', size_t>()
        .default_value(size_t{100})
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

    double interaction_coef;
    program.add_argument("-w")
        .help("coefficient to multiply interactions (for debbuging)")
        .scan<'g', double>()
        .default_value(1.0)
        .store_into(interaction_coef);


    size_t dump_each;
    program.add_argument("--dump-each")
        .help("number of steps between saving dump files")
        .scan<'u', size_t>()
        .default_value(size_t{10000000})
        .store_into(dump_each);

    size_t restart_each;
    program.add_argument("--restart-each")
        .help("number of steps between saving restart files")
        .scan<'u', size_t>()
        .default_value(size_t{10000000})
        .store_into(restart_each);

    size_t print_each;
    program.add_argument("--print-each")
        .help("number of steps between printing termo")
        .scan<'u', size_t>()
        .default_value(size_t{10000})
        .store_into(print_each);

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

    bool restart = program.get<bool>("--restart");
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

    std::vector<int> types(n_types);
    std::iota(types.begin(), types.end(), 0); // 0 1 2 3 ... n_types - 1

    PartitionLoader partition;
    partition.load(std::format("partitions/{}.txt", world_rank+1));

    std::vector<std::string> eint_filenames((n_types-1)*(n_types-1));
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            int index = I*(n_types-1)+J;
            eint_filenames[index] = "new_eint_"+std::to_string(I)+"_"+std::to_string(J)+".txt";
        }
    }

    int z_max = 30;
    MpiDataLoader loader(z_max*2+1, (n_types-1)*(n_types-1));
    try {
        loader.loadAndDistribute("new_neighbors.txt", eint_filenames, partition.partition, partition.nbrs);

    } catch (const std::exception& e) {
        std::cerr << "world_rank " << world_rank << " поймал исключение: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int cols = partition.nsites;
    int cols_ghost = loader.local_ind.size();

    unsigned int shared_seed = 0;
    std::mt19937 gen_shared(shared_seed);

    unsigned int seed = world_rank;
    std::mt19937 gen(seed);

    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> uniform_col(0, cols-1);


    Matrix<int> m_load;
    if (restart){
        try {
            m_load.load_from_text(restart_dir+"/m_"+std::to_string(world_rank+1)+".txt");
        }
        catch (...) {
            std::cerr << "[ERROR]: cannot open restart file" << std::endl;
            MPI_Finalize();
            return 1;
        }
        if (rows != m_load.rows()){
            std::cerr << "[WARNING]: Number of rows in restart file differs!!! Number from restart will be used" << std::endl;
        }
        rows = m_load.rows();
    }

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
        std::cout << "===================================\n\n" << std::endl;
        if (restart){
        std::cout << "===== RESTART LOADED ==============" << std::endl;
        std::cout << "===================================\n\n" << std::endl;
        }
    }

    
    std::uniform_int_distribution<int> uniform_row(0, rows-1);

    int natoms = rows*loader.total_site_types;
    
    int natoms_check;
    MPI_Allreduce(&natoms, &natoms_check, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (world_rank == 0){
        if (natoms != natoms_check){
            std::cerr << "[BUG] natoms differs across ranks!" << std::endl;
        }
        else {
            std::cout << "natoms: " << natoms << std::endl;
        }
    }


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
        x_glob = Matrix(loader.total_site_types, n_types, 0.0); 
    }
    Matrix<int> m = Matrix(rows, cols_ghost, types[0]); 
    if (restart){
        for (int i = 0; i<rows; i++){
            for (int j = 0; j<cols; j++){
                m(i, j) = m_load(i, j);
            }
        }
    }
    // energy matrix
    Matrix<double> es = Matrix(cols, n_types, 0.0);
    for (int k = 0; k<n_types-1; k++){
        Matrix<double> es_load;
        es_load.load_from_text("new_es_"+std::to_string(k)+".txt"); // TODO: заменить число на химический тип
        for (int i = 0; i<cols; ++i){
            es(i, k) = es_load(partition.partition[i]-1, 1);
        }
    }
    // interaction matrix
    std::vector<std::unique_ptr<Matrix<double>>> interactions;
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            interactions.push_back(std::make_unique<Matrix<double>>(cols, z_max, 0.0));
            int index = I*(n_types-1)+J;
            for (int i = 0; i<cols; ++i){
                for (int j = 0; j<loader.local_z[i]; ++j){
                    (*interactions[index])(i, j) = loader.getNbrEint(i, j, index)*interaction_coef;
                }
            }
            
        }
    } 


    
    double energy_loc = 0.0;
    std::vector<int> number_of_solutes_loc(n_types, 0);
    if (restart){
        for (int i = 0; i<rows; i++){
            for (int j = 0; j<cols; j++){
                energy_loc += es(j, m(i, j));
                if (m(i, j)>0){
                    for (int k = 0; k<loader.local_z[j]; k++){ // over neighbors of j
                        int jk = loader.getNbrLocalIndex(j, k); 
                        if (m(i, jk) > 0){
                            int index = (m(i, j)-1)*(n_types-1)+m(i, jk)-1;   
                            energy_loc += (*interactions[index])(j, k)/2;
                        }
                    }
                }
                number_of_solutes_loc[m(i, j)] ++;
            }
        }
    }
    else {
        number_of_solutes_loc[types[0]] = rows*cols;
    }
    double energy;
    MPI_Reduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::vector<int> number_of_solutes(n_types);
    MPI_Reduce(number_of_solutes_loc.data(), number_of_solutes.data(), n_types, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int accepted_loc = 0;
    int accepted;
    if (world_rank==0){
        accepted = 0;
    }

    int cols_tot = 0;
    MPI_Allreduce(&cols, &cols_tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (loader.total_site_types!=cols_tot){
        std::cerr << "[BUG]: Sum of number of cols: " << cols_tot << " is not equal to total number of cols: " << loader.total_site_types << std::endl;
        throw std::runtime_error("loader.total_site_types!=cols_tot");
    }

    std::ofstream out;
    if (world_rank==0){
        try {
            std::uintmax_t deleted_count = std::filesystem::remove_all(dump_dir);
        } 
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "[ERROR]: error during removing of dump folder: " << e.what() << std::endl;
        }
        std::filesystem::create_directory(dump_dir); 
        try {
            std::uintmax_t deleted_count = std::filesystem::remove_all(restart_dir);
        } 
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "[ERROR]: error during removing of restart folder: " << e.what() << std::endl;
        }
        std::filesystem::create_directory(restart_dir); 
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
        out << std::setw(15) << "per_site_energy" << std::endl;
    }

    /// Setting up MPI communication for occupation matrix (m)
    MatrixExchanger<int> exchanger(world_rank, world_size);
    int num_partners;
    std::vector<std::vector<int>> requests;
    if (world_size>1){
        /// find all blocks to communicate
        std::vector<int> target_ranks;
        std::unordered_map<int, int> block_ind;
        int block_ind_cnt = 0; 
        for (int j = 0; j<cols; j++){
                for (int k = 0; k<partition.z[j]; k++){ // over interblock bonds of j
                    // найдем нужный world_rank=block and nbr_id
                    int block  = partition.getNbrBlock(j, k);
                    if (block_ind.find(block) == block_ind.end()) { // there is no key "block" in dict (first appearance)
                        target_ranks.push_back(block-1);
                        block_ind[block] = block_ind_cnt;
                        block_ind_cnt ++;
                    }
                }
            }   

        /// setting up requests
        num_partners = target_ranks.size();
        requests.resize(num_partners); // IDs of target sites in neighboring block
        for (int j = 0; j<cols; j++){
                for (int k = 0; k<partition.z[j]; k++){ 
                    int block  = partition.getNbrBlock(j, k);
                    int nbr_id = partition.getNbrID(j, k);
                    requests[block_ind[block]].push_back(nbr_id);
                }
            }
        exchanger.initialize_connections(m, target_ranks, requests, loader.local_ind);
    }/// End of communication initialization
    
    ProgressBar bar(mc_steps, (world_rank == 0));

    for (size_t step = 1; step <= mc_steps; step++){
        if (world_size>1){  /// syncronization of occupation matrix's ghost columns (m)
            exchanger.exchange_step(m); // send to buffers

            for (int p = 0; p < num_partners; ++p) {
                int num_requested_cols = requests[p].size();
                for (int k = 0; k < num_requested_cols; ++k) {
                    int requested_id = requests[p][k]; 
                    const int* column_data = exchanger.get_received_column(p, k);
                    for (int r = 0; r < rows; ++r) {
                        int target_col = loader.local_ind[requested_id];
                        assert(target_col >= cols && "ghost write hits a REAL (owned) column!");
                        m(r, target_col) = column_data[r];
                    }
                }
            }
        }
        /// trial step
        int i = uniform_row(gen);
        int j = uniform_col(gen);
        int type_old = m(i, j);
        int type_new = type_old;
        while (type_new == type_old){
            type_new = types[uniform_type(gen)];
        }

        double dE = es(j, type_new) - es(j, type_old);
        double dE_int = 0.0;
        int index_old, index_new;
        for (int k = 0; k<loader.local_z[j]; k++){ // over neighbors of j
            int jk = loader.getNbrLocalIndex(j, k); 
            if (type_old > 0 && m(i, jk) > 0){
                index_old = (type_old-1)*(n_types-1)+m(i, jk)-1;   
                dE_int -= (*interactions[index_old])(j, k);
            }
            if (type_new > 0 && m(i, jk) > 0){
                index_new = (type_new-1)*(n_types-1)+m(i, jk)-1;
                dE_int += (*interactions[index_new])(j, k);
            }
        }
        /* if (world_size==1){
            if (dE_int!=0.0){
                std::cout << dE_int << std::endl;
            }
        } */
        dE += dE_int;
        double dF = dE + mu[type_new] - mu[type_old];

        double prob = std::exp(-dF/kT);
        double p = uniform(gen);

        bool global_acceptance_flag = false;
        if (is_vcsgc){
            std::vector<int> dN(n_types, 0);
            if (p<=prob){
                dN[type_new] = 1;
                dN[type_old] = -1;
            }
            std::vector<int> dN_tot;
            if (world_size > 1) {
                dN_tot.assign(n_types, 0);
                MPI_Allreduce(dN.data(), dN_tot.data(), n_types, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            } else {
                dN_tot = dN;
            }
            double dF_glob = 0;
            for (int k = 1; k<n_types; k++){
                dF_glob += kappa*dN_tot[k]*(dN_tot[k] + 2*(number_of_solutes[k]-number_of_solutes_target[k]))/natoms;
            }
            double prob_glob = std::exp(-dF_glob/kT);
            double p_glob = uniform(gen_shared);
            global_acceptance_flag = (p_glob<=prob_glob);
            if (global_acceptance_flag){
                for (int k = 0; k<n_types; k++){
                    number_of_solutes[k] += dN_tot[k];
                }
            }
        }
        else {
            global_acceptance_flag = true;
        }

        if (p<=prob && global_acceptance_flag){
            m(i, j) = type_new;
            accepted_loc ++;
            number_of_solutes_loc[type_new]++;
            number_of_solutes_loc[type_old]--;
            energy_loc += dE;
        }

        // syncronization of thermo
        if (step%print_each==0 || step == mc_steps)
        {
            if (!is_vcsgc){
                MPI_Reduce(
                    number_of_solutes_loc.data(), 
                    number_of_solutes.data(), 
                    n_types, 
                    MPI_INT, 
                    MPI_SUM, 
                    0, // root
                    MPI_COMM_WORLD
                );
            }

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
                out << std::setw(15) << std::fixed << std::setprecision(4) << energy/natoms << std::endl;
            }
            accepted = 0;
            accepted_loc = 0;
        }

        if (step%dump_each==0)
        {
            Matrix<double> x_loc(cols, n_types, 0.0); 
            for (int jj = 0; jj<cols; jj++){
                for (int ii = 0; ii<rows; ii++){
                    x_loc(jj, m(ii, jj)) ++;
                }
                for (int k=0; k<n_types; k++){
                    x_loc(jj, k) /= rows;
                }
            }

            // Буферы для Root-процесса
            std::vector<int> recv_counts_data, disp_data;
            std::vector<int> recv_counts_idx, disp_idx;
            std::vector<double> temp_x_glob;
            std::vector<int> global_indices;

            int local_data_size = cols * n_types;

            if (world_rank == 0) {
                recv_counts_data.resize(world_size);
                disp_data.resize(world_size);
                recv_counts_idx.resize(world_size);
                disp_idx.resize(world_size);
            }

            MPI_Gather(&cols, 1, MPI_INT, recv_counts_idx.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_data_size, 1, MPI_INT, recv_counts_data.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); // можно пересчитать без отправки через local_data_size = cols * n_types; 

            if (world_rank == 0) {
                int d_data = 0, d_idx = 0;
                for (int i = 0; i < world_size; ++i) {
                    disp_data[i] = d_data;
                    d_data += recv_counts_data[i];
                    disp_idx[i] = d_idx;
                    d_idx += recv_counts_idx[i];
                }
                
                global_indices.resize(d_idx);
                temp_x_glob.resize(d_data); 
                assert(d_data == loader.total_site_types*n_types);// (number of elements in x_glob)
            }

            MPI_Gatherv(partition.partition.data(), cols, MPI_INT,
                        global_indices.data(), recv_counts_idx.data(), disp_idx.data(), MPI_INT,
                        0, MPI_COMM_WORLD);

            MPI_Gatherv(x_loc.data(), local_data_size, MPI_DOUBLE,
                        temp_x_glob.data(), recv_counts_data.data(), disp_data.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);

            if (world_rank == 0) {
                for (size_t i = 0; i < global_indices.size(); ++i) {
                    int target_row = global_indices[i]-1;
                    
                    std::copy(temp_x_glob.begin() + i * n_types, 
                            temp_x_glob.begin() + (i + 1) * n_types, 
                            x_glob.begin() + target_row * n_types);
                }
            // dump
            x_glob.save_to_text(dump_dir+"/x_"+std::to_string(step)+".txt");
            }
        } 

        if (step%restart_each==0)
        {   
            m.save_to_text(restart_dir+"/m_"+std::to_string(world_rank+1)+".txt");
        } 

    }//end MC loop

    if (world_rank==0){
        out.close();
    }

    if (world_size>1){
        exchanger.mpi_finalize();
    }
    MPI_Finalize();

    return 0;
}
