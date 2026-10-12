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

    program.add_argument("total concentrations")
            .help("list of pairs <element Xtot [in at.%]> (Ni 1.0 Cu 2.5)")
            .nargs(argparse::nargs_pattern::any); 

    program.add_argument("-r", "--restart")
           .help("load restart files")
           .implicit_value(true) 
           .default_value(false);
    
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

    double fgb_trial_fraction;
    program.add_argument("--fgb-trial")
        .help("fraction of fgb trail steps (from 0 to 1)")
        .scan<'g', double>()
        .default_value(0.1)
        .store_into(fgb_trial_fraction);

    double fgb_step;
    program.add_argument("--fgb-step")
        .help("fgb trail step magnitude (from 0 to 1)")
        .scan<'g', double>()
        .default_value(0.1)
        .store_into(fgb_step);

    double fgb;
    program.add_argument("--fgb")
        .help("initial f_gb (from 0 to 1)")
        .scan<'g', double>()
        .default_value(0.2)
        .store_into(fgb);

    double interaction_coef;
    program.add_argument("-w")
        .help("coefficient to multiply interactions (for debbuging)")
        .scan<'g', double>()
        .default_value(1.0)
        .store_into(interaction_coef);

    std::string matrix_el;
    program.add_argument("--matrix")
        .help("matrix element")
        .default_value(std::string("Ag"))
        .store_into(matrix_el);

    double kgamma;
    program.add_argument("-g", "--gamma")
        .help("kgamma - per-atom structural GB energy [kJ/mol]")
        .scan<'g', double>()
        .default_value(9.31581395739516)
        .store_into(kgamma);

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

    if (fgb_trial_fraction < 0.0 || fgb_trial_fraction > 1.0) {
        std::cerr << "ERROR: fraction of fgb trial steps must be from 0 to 1!" << std::endl;
        return 1;
    }

    if (fgb_step < 0.0 || fgb_step > 1.0) {
        std::cerr << "ERROR: fgb trial step magnitude must be from 0 to 1!" << std::endl;
        return 1;
    }

    bool restart = program.get<bool>("--restart");

    auto xtot_arguments = program.get<std::vector<std::string>>("total concentrations");

    if (xtot_arguments.size() % 2 != 0) {
        std::cerr << "ERROR: Odd number of arguments. Every key must have a meaning!" << std::endl;
        return 1;
    }

    std::map<std::string, double> xtot_map;
    for (size_t i = 0; i < xtot_arguments.size(); i += 2) {
        try {
            xtot_map[xtot_arguments[i]] = std::stod(xtot_arguments[i + 1])/100;
            if (xtot_map[xtot_arguments[i]] < 0.0  || xtot_map[xtot_arguments[i]] > 1.0){
                std::cerr << "ERROR: expected a concentration in at. % (from 0.0 to 100.0)!" << std::endl;
                return 1;
            }
        } 
        catch (const std::invalid_argument& e) {
            std::cerr << "ERROR: expected a double value after chemical specie!" << e.what() << std::endl;
            return 1;
        } 
        catch (const std::out_of_range& e) {
            std::cerr << "ERROR: the number is too big for double! " << e.what() << std::endl;
            return 1;
        }
    }

    std::string xtot_str = "";
    std::string element_str = matrix_el + " ";
    std::vector<std::string> elements;
    std::vector<double> Xtot;
    elements.push_back(matrix_el);
    Xtot.push_back(1.0);
    for (const auto& [key, value] : xtot_map) {
        std::string s = std::to_string(value);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos); // Удаляем нули на конце
        if (s.back() == '.') s.pop_back();                       // Удаляем точку, если число целое
        xtot_str += s + " ";
        element_str += key + " ";
        elements.push_back(key);
        Xtot.push_back(value);
        Xtot[0] -= value;
    }
    

    if (Xtot[0] < 0.0){
        std::cerr << "ERROR: Total solute concetration exceeds 100%!" << std::endl;
        return 1;
    }

    std::string s = std::to_string(Xtot[0]);
    s.erase(s.find_last_not_of('0') + 1, std::string::npos); // Удаляем нули на конце
    if (s.back() == '.') s.pop_back();                       // Удаляем точку, если число целое
    xtot_str = s + " " + xtot_str;

    if (fgb<0 || fgb>1){
        std::cerr << "ERROR: fgb must be from 0 to 1!" << std::endl;
        return 1;
    }

    const int n_types = elements.size();
    std::uniform_int_distribution<int> uniform_type(0, n_types-1);

    std::vector<int> types(n_types);
    std::iota(types.begin(), types.end(), 0); // 0 1 2 3 ... n_types - 1

    std::vector<std::string> eint_filenames((n_types-1)*(n_types-1));
    for (int I = 0; I<n_types-1; ++I){
        for (int J = 0; J<n_types-1; ++J){
            int index = I*(n_types-1)+J;
            eint_filenames[index] = "eint_"+elements[I+1]+"_"+elements[J+1]+".txt";
        }
    }
    

    int z_max = 30;
    DataLoader loader(z_max*2+1, (n_types-1)*(n_types-1));
    try {
        loader.loadAndDistribute("neighbors.txt", eint_filenames);

    } catch (const std::exception& e) {
        std::cerr << " ERROR: " << e.what() << std::endl;
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
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "================= MC + Grain Growth ======================" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "================= Input parameters =======================" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "               total site types     : " << loader.total_site_types << std::endl;
    std::cout << "               MC steps             : " << mc_steps << std::endl;
    std::cout << "               types                : " << n_types << std::endl;
    std::cout << "               elements             : " << element_str << std::endl;
    std::cout << "               total concentrations : " << xtot_str << std::endl;
    std::cout << "               fgb trial fraction   : " << fgb_trial_fraction << std::endl;
    std::cout << "=========================================================\n\n" << std::endl;
    if (restart){
    std::cout << "================= RESTART LOADED ========================" << std::endl;
    std::cout << "=========================================================\n\n" << std::endl;
    }
    

    int natoms = loader.total_site_types;

    // occupation matrix
    Matrix<int> m = Matrix(1, cols, types[0]); 
    if (restart){
        for (int j = 0; j<cols; j++){
            m(0, j) = m_load(0, j); // TODO to vectors
        }
    }
    // energy matrix
    Matrix<double> es = Matrix(cols, n_types, 0.0);
    for (int k = 1; k<n_types; k++){
        Matrix<double> es_load;
        es_load.load_from_text("es_"+elements[k]+".txt");
        for (int i = 0; i<cols; ++i){
            es(i, k) = es_load(i, 1);
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
        for (int j = 0; j<cols; j++){
            energy += es(j, m(0, j));
            if (m(0, j)>0){
                for (int k = 0; k<loader.z[j]; k++){ // over neighbors of j
                    int jk = loader.getNbrIndex(j, k); 
                    if (m(0, jk) > 0){
                        int index = (m(0, j)-1)*(n_types-1)+m(0, jk)-1;   
                        energy += (*interactions[index])(j, k)/2;
                    }
                }
            }
            number_of_solutes[m(0, j)] ++;
        }
    }
    else {
        number_of_solutes[types[0]] = cols;
    }
    

    std::vector<double> Xc(n_types, 0);
    for (int k=0; k<n_types; k++){
        Xc[k] = (Xtot[k]-fgb*number_of_solutes[k]/cols)/(1-fgb);
        if (Xc[k] < 0.0 || Xc[k] > 1.0){
            std::cerr << "ERROR: Such fgb, Xgb and Xtot leads to unphysical Xc: " << Xc[k] << " for " << elements[k] << std::endl;
            return 1;
        }
    }
    std::vector<double> Xc_new(n_types, 0);

    double Sc = 0.0;
        for (int k=0; k<n_types; k++){
            if (Xc[k] > 0.0){
                Sc += Xc[k]*log(Xc[k]);
            }
        }
    double Sc_new;

    int accepted1 = 0;
    int accepted2 = 0;

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
    out.open("mc_gg_output.txt");
    if (!out.is_open()) {
        std::cerr << "[ERROR]: error during creating of mc_output.txt file:" << std::endl;
        return 1; 
    }
    out << "MC + Grain Growth. kT: " << kT;
    for (int k = 1; k < n_types; k++){
        out << (" Xtot_" + elements[k] + ": " + std::to_string(Xtot[k]));
    }
    out << std::endl;
    out << std::left; 
    out << std::setw(14) << "step" 
        << std::setw(12) << "acc_1"
        << std::setw(12) << "acc_2"
        << std::setw(12) << "f_gb";
    for (int k = 0; k < n_types; k++) {
        out << std::setw(12) << ("Xgb_" + elements[k]);
    }
    for (int k = 0; k < n_types; k++) {
        out << std::setw(12) << ("Xc_" + elements[k]);
    }
    out << std::setw(15) << "per_site_seg_energy" << std::endl;
    
    
    ProgressBar bar(mc_steps);

    for (size_t step = 0; step < mc_steps; step++){      
        double trial_type = uniform(gen);
        if (trial_type<1-fgb_trial_fraction){// trial step #1
            int j = uniform_col(gen);
            int type_old = m(0, j);
            int type_new = type_old;
            while (type_new == type_old){
                type_new = types[uniform_type(gen)];
            }

            bool auto_reject = false;
            for (int k=0; k<n_types; k++){
                double Xgb_new = number_of_solutes[k];
                if (k==type_new){
                    Xgb_new ++;
                }
                if (k==type_old){
                    Xgb_new --;
                }
                Xgb_new /= cols;

                Xc_new[k] = (Xtot[k]-fgb*Xgb_new)/(1-fgb);
                if (Xc_new[k] < 0.0 || Xc_new[k] > 1.0){
                    auto_reject = true;
                }
            }
            Sc_new = 0.0;
            for (int k=0; k<n_types; k++){
                if (Xc[k] > 0.0){
                    Sc_new += Xc_new[k]*log(Xc_new[k]);
                }
            }

            if (!auto_reject){
                double dE = es(j, type_new) - es(j, type_old);
                double dE_int = 0.0;
                int index_old, index_new;
                for (int k = 0; k<loader.z[j]; k++){ // over neighbors of j
                    int jk = loader.getNbrIndex(j, k); 
                    if (type_old > 0 && m(0, jk) > 0){
                        index_old = (type_old-1)*(n_types-1)+m(0, jk)-1;   
                        dE_int -= (*interactions[index_old])(j, k);
                    }
                    if (type_new > 0 && m(0, jk) > 0){
                        index_new = (type_new-1)*(n_types-1)+m(0, jk)-1;
                        dE_int += (*interactions[index_new])(j, k);
                    }
                }
                dE += dE_int;

                double dF = fgb*dE + cols*(1-fgb)*kT*(Sc_new-Sc);

                double prob = std::exp(-dF/kT);
                double p = uniform(gen);

                if (p<=prob){
                    m(0, j) = type_new;
                    accepted1 ++;
                    number_of_solutes[type_new]++;
                    number_of_solutes[type_old]--;
                    energy += dE; // TODO разделить на полную энергию и энергию гз
                    for (int k=0; k<n_types; k++){
                        Xc[k] = Xc_new[k];
                    }
                    Sc = Sc_new;
                }
            }
        }
        else { // trial step #2
            double dfgb = (uniform(gen)*2.0-1.0)*fgb_step;
            double fgb_new = fgb + dfgb;
            if (fgb_new > 0 && fgb_new < 1){
                bool auto_reject = false;
                for (int k=1; k<n_types; k++){
                    double Xgb = static_cast<double>(number_of_solutes[k])/cols;
                    Xc_new[k] = (Xtot[k]-fgb_new*Xgb)/(1.0-fgb_new);
                }
                Xc_new[0] = 1.0;
                for (int k=1; k<n_types; k++){
                    Xc_new[0] -= Xc_new[k];
                }
                if (Xc_new[0] < 0.0 || Xc_new[0] > 1.0){
                    auto_reject = true;
                }

                if (!auto_reject){
                    Sc_new = 0.0;
                    for (int k=0; k<n_types; k++){
                        if (Xc[k] > 0.0){
                            Sc_new += Xc_new[k]*log(Xc_new[k]);
                        }
                    }
                    double dF = dfgb*(cols*kgamma + energy) + kT*((1-fgb_new)*Sc_new - (1-fgb)*Sc);
                    double prob = std::exp(-dF/kT);
                    double p = uniform(gen);

                    //std::cout << "step 2: " << dF << std::endl;
                    if (p<=prob){
                        accepted2 ++;
                        for (int k=0; k<n_types; k++){
                            Xc[k] = Xc_new[k];
                        }
                        Sc = Sc_new;
                        fgb = fgb_new;
                    }
                }
            }

        }

        // thermo
        if (step%print_each==0 || step == mc_steps-1)
        {
            bar.update(step); 
            //thermo
            double acc1 = static_cast<double>(accepted1)/dump_each/(1-fgb_trial_fraction);
            double acc2 = static_cast<double>(accepted2)/dump_each/fgb_trial_fraction;
            out << std::left;
            out << std::setw(14) << step;
            out << std::setw(12) << std::fixed << std::setprecision(6) << acc1;
            out << std::setw(12) << std::fixed << std::setprecision(6) << acc2;
            out << std::setw(12) << std::fixed << std::setprecision(6) << fgb;
            for (int k = 0; k<n_types; k++){
                out << std::setw(12) << std::fixed << std::setprecision(4) << static_cast<double>(number_of_solutes[k])/natoms;
            }
            for (int k = 0; k<n_types; k++){
                out << std::setw(12) << std::fixed << std::setprecision(4) << Xc[k];
            }
            out << std::setw(15) << std::fixed << std::setprecision(4) << energy/natoms << std::endl;
            
            accepted1 = 0;
            accepted2 = 0;
        }

        if (step%dump_each==0)
        {
            m.save_to_text(dump_dir+"/m_"+std::to_string(step)+".txt");   
        } 

        if (step%restart_each==0)
        {   
            m.save_to_text(restart_dir+"/m.txt");
        } 

    }//end MC loop

    
    out.close();

    return 0;
}
