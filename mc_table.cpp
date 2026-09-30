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
#include "DataLoader.h"
#include <algorithm>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <numeric> 
#include <limits.h>
#include <filesystem>

std::string dump_dir = "dump";
std::string restart_dir = "restart";



int main(int argc, char* argv[]) {
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
        std::cerr << err.what() << std::endl;
        std::cerr << program;
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

    std::vector<std::string> eint_filenames((n_types-1)*(n_types-1));
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            int index = I*(n_types-1)+J;
            eint_filenames[index] = "eint_"+std::to_string(I)+"_"+std::to_string(J)+".txt";
        }
    }

    int z_max = 30;
    DataLoader loader(z_max*2+1, (n_types-1)*(n_types-1));
    try {
        loader.loadAndDistribute("neighbors.txt", eint_filenames);

    } catch (const std::exception& e) {
        std::cerr << " Исключение: " << e.what() << std::endl;
    }

    int cols = loader.total_site_types;

    unsigned int seed = 42;
    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> uniform_col(0, cols-1);


    Matrix<int> m_load;
    if (restart){
        try {
            m_load.load_from_text(restart_dir+"/m_"+std::to_string(1)+".txt");
        }
        catch (...) {
            std::cerr << "[ERROR]: cannot open restart file" << std::endl;
            return 1;
        }
        if (rows != m_load.rows()){
            std::cerr << "[WARNING]: Number of rows in restart file differs!!! Number from restart will be used" << std::endl;
        }
        rows = m_load.rows();
    }

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
    Matrix<double> x;
    x = Matrix(loader.total_site_types, n_types, 0.0); 

    Matrix<int> m = Matrix(rows, cols, types[0]); 
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
        es_load.load_from_text("es_"+std::to_string(k)+".txt"); // TODO: заменить число на химический тип
        for (int i = 0; i<cols; ++i){
            es(i, k+1) = es_load(i, 1);
        }
    }
    // interaction matrix
    std::vector<std::unique_ptr<Matrix<double>>> interactions;
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            interactions.push_back(std::make_unique<Matrix<double>>(cols, z_max, 0.0));
            int index = I*(n_types-1)+J;
            for (int i = 0; i<cols; ++i){
                for (int j = 0; j<loader.z[i]; ++j){
                    (*interactions[index])(i, j) = loader.getNbrEint(i, j, index)*interaction_coef;
                }
            }
            
        }
    } 

    double energy = 0.0;
    std::vector<int> number_of_solutes(n_types, 0);
    if (restart){
        for (int i = 0; i<rows; i++){
            for (int j = 0; j<cols; j++){
                energy += es(j, m(i, j));
                if (m(i, j)>0){
                    for (int k = 0; k<loader.z[j]; k++){ // over neighbors of j
                        int jk = loader.getNbrIndex(j, k); 
                        if (m(i, jk) > 0){
                            int index = (m(i, j)-1)*(n_types-1)+m(i, jk)-1;   
                            energy += (*interactions[index])(j, k)/2;
                        }
                    }
                }
                number_of_solutes[m(i, j)] ++;
            }
        }
    }
    else {
        number_of_solutes[types[0]] = rows*cols;
    }
    
    int accepted = 0;

    std::ofstream out;
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
    
    
    ProgressBar bar(mc_steps);

    for (size_t step = 1; step <= mc_steps; step++){      
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
        for (int k = 0; k<loader.z[j]; k++){ // over neighbors of j
            int jk = loader.getNbrIndex(j, k); 
            if (type_old > 0 && m(i, jk) > 0){
                index_old = (type_old-1)*(n_types-1)+m(i, jk)-1;   
                dE_int -= (*interactions[index_old])(j, k);
            }
            if (type_new > 0 && m(i, jk) > 0){
                index_new = (type_new-1)*(n_types-1)+m(i, jk)-1;
                dE_int += (*interactions[index_new])(j, k);
            }
        }
        dE += dE_int;
        double dF = dE + mu[type_new] - mu[type_old];

        double prob = std::exp(-dF/kT);
        double p = uniform(gen);

        bool vcsgc_acceptance_flag = false;
        if (is_vcsgc){
            std::vector<int> dN(n_types, 0);
            if (p<=prob){
                dN[type_new] = 1;
                dN[type_old] = -1;
            }
            double dF_vcsgc = 0;
            for (int k = 1; k<n_types; k++){
                dF_vcsgc += kappa*dN[k]*(dN[k] + 2*(number_of_solutes[k]-number_of_solutes_target[k]))/natoms;
            }
            double prob_vcsgc = std::exp(-dF_vcsgc/kT);
            double p_vcsgc = uniform(gen);
            vcsgc_acceptance_flag = (p_vcsgc<=prob_vcsgc);
            if (vcsgc_acceptance_flag){
                for (int k = 0; k<n_types; k++){
                    number_of_solutes[k] += dN[k];
                }
            }
        }
        else {
            vcsgc_acceptance_flag = true;
        }

        if (p<=prob && vcsgc_acceptance_flag){
            m(i, j) = type_new;
            accepted ++;
            number_of_solutes[type_new]++;
            number_of_solutes[type_old]--;
            energy += dE;
        }

        // thermo
        if (step%print_each==0 || step == mc_steps)
        {
            bar.update(step); 
            //thermo
            double acc = static_cast<double>(accepted)/dump_each;
            out << std::left;
            out << std::setw(14) << step;
            out << std::setw(12) << std::fixed << std::setprecision(6) << acc;
            for (int k = 0; k<n_types; k++){
                out << std::setw(12) << std::fixed << std::setprecision(4) << static_cast<double>(number_of_solutes[k])/natoms;
            }
            out << std::setw(15) << std::fixed << std::setprecision(4) << energy/natoms << std::endl;
            
            accepted = 0;
        }

        if (step%dump_each==0)
        {
            for (int jj = 0; jj<cols; jj++){
                for (int ii = 0; ii<rows; ii++){
                    x(jj, m(ii, jj)) ++;
                }
                for (int k=0; k<n_types; k++){
                    x(jj, k) /= rows;
                }
            }

            x.save_to_text(dump_dir+"/x_"+std::to_string(step)+".txt");
            
        } 

        if (step%restart_each==0)
        {   
            m.save_to_text(restart_dir+"/m_"+std::to_string(1)+".txt");
        } 

    }//end MC loop

    
    out.close();

    return 0;
}
