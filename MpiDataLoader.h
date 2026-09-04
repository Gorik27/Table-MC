#pragma once

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include "matrix.hpp"

class MpiDataLoader {
private:
    int rank;
    int world_size;
    int max_cols;
    int n_solute_pairs;

public:
    std::unordered_map<int, int> local_ind;
    int local_count;
    std::vector<int> local_z;
    std::vector<int> local_nbrs;
    std::vector<double> local_eint;
    int total_site_types;

    // Конструктор инициализирует параметры MPI и задает ширину строки
    MpiDataLoader(int _max_cols = 20, int _n_solutes = 1) 
        : max_cols(_max_cols), n_solute_pairs(_n_solutes)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    // Главный метод для загрузки и распределения данных
    void loadAndDistribute(const std::string& nbr_filename, 
                            std::vector<std::string> eint_filenames, 
                            std::vector<int> partition,
                            std::vector<int> nbrs) 
        {
        std::vector<int> global_ids_n;
        std::vector<double> global_eint;
        int total_rows = 0;

        // 1. Чтение файла процессом Rank 0
        if (rank == 0) {
            /// read new_neighbors.txt
            std::ifstream file1(nbr_filename);
            if (!file1.is_open()) {
                throw std::runtime_error("MpiDataLoader: Не удалось открыть файл " + nbr_filename);
            }

            std::string line;
            while (std::getline(file1, line)) {
                    int number;
                    int id_c;
                    std::stringstream ss(line);
                    if (!(ss >> id_c)) {
                        continue; // Если строка пустая, просто пропускаем её
                    }

                    int count = 0;

                    while (ss >> number) {
                        if (count < max_cols) {
                            global_ids_n.push_back(number);
                            count++;
                        }
                    }
                    // Выравнивание строки нулями
                    while (count < max_cols) {
                        global_ids_n.push_back(0);
                        count++;
                    }
                
                total_rows++;
            }
            file1.close();
            total_site_types = total_rows;

            /// read eint.txt
            for (size_t k = 0; k < n_solute_pairs; ++k) {
                int total_rows2 = 0;
                std::string eint_filename = eint_filenames[k];
                std::ifstream file2(eint_filename);
                if (!file2.is_open()) {
                    throw std::runtime_error("MpiDataLoader: Не удалось открыть файл " + eint_filename);
                }

                while (std::getline(file2, line)) {
                        std::stringstream ss(line);
                        double energy;
                        int id_c;
                        if (!(ss >> id_c)) {
                            continue; // Если строка пустая, просто пропускаем её
                        }

                        int count = 0;

                        while (ss >> energy) {
                            if (count < max_cols) {
                                global_eint.push_back(energy);
                                count++;
                            }
                        }
                        // Выравнивание строки нулями
                        while (count < max_cols) {
                            global_eint.push_back(0.0);
                            count++;
                        }
                    
                    total_rows2++;
                }
                file2.close();
                if (total_rows2!=total_rows){
                    throw std::runtime_error("MpiDataLoader: number of rows in " + nbr_filename + " and " + eint_filename + " does not match!");
                }
            }
        }        

        MPI_Bcast(&total_site_types, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_size == 1) {
            local_count = total_rows;

            local_z.resize(local_count);
            local_nbrs.resize(local_count * max_cols);
            local_eint.resize(local_count * max_cols * n_solute_pairs);
            for (int i = 0; i < local_count; i++) {
                int z = 0;
                local_ind[i+1] = i;
                for (int j = 0; j < max_cols; j++) {
                    int index = i * max_cols + j;
                    local_nbrs[index] = global_ids_n[index];
                    if (global_ids_n[index]!=0){
                        z++;
                    }
                    for (int k = 0; k < n_solute_pairs; k++){
                        local_eint[index * n_solute_pairs + k] = global_eint[index * n_solute_pairs + k];
                    }
                }
                local_z[i] = z;
            }
        }

        // ====================================================================
        // КОД НИЖЕ ВЫПОЛНЯЕТСЯ ТОЛЬКО ЕСЛИ ПРОЦЕССОВ >= 2
        // ====================================================================
        else {
            if (n_solute_pairs>1){
                throw std::runtime_error("[ERROR] Does not support parallel version for n_solute_pairs>1!");
            }
            local_count = partition.size();

            local_z.resize(local_count);
            local_nbrs.resize(local_count * max_cols);
            local_eint.resize(local_count * max_cols);
            // Создаем окно памяти на процессе 0
            MPI_Win win1;
            MPI_Win win2;
            int disp_unit1 = sizeof(int);
            int disp_unit2 = sizeof(double);
            if (rank == 0) {
                MPI_Win_create(global_ids_n.data(), total_rows * max_cols * sizeof(int), disp_unit1, MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
                MPI_Win_create(global_eint.data(),  total_rows * max_cols * sizeof(double), disp_unit2, MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
            } else {
                MPI_Win_create(NULL, 0, disp_unit1, MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
                MPI_Win_create(NULL, 0, disp_unit2, MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
            }

            std::vector<int> block_lengths(local_count, max_cols);
            
            std::vector<int> memory_offsets(local_count);
            for (int i = 0; i < local_count; i++) {
                memory_offsets[i] = (partition[i]-1) * max_cols;
            }

            MPI_Datatype target_indexed_type_int;
            // Создаем тип на основе массива смещений
            MPI_Type_indexed(
                local_count, 
                block_lengths.data(), 
                memory_offsets.data(),  
                MPI_INT, 
                &target_indexed_type_int
            );
            MPI_Type_commit(&target_indexed_type_int);

            MPI_Datatype target_indexed_type_double;
            // Создаем тип на основе массива смещений
            MPI_Type_indexed(
                local_count, 
                block_lengths.data(), 
                memory_offsets.data(),  
                MPI_DOUBLE, 
                &target_indexed_type_double
            );
            MPI_Type_commit(&target_indexed_type_double);

            // --- Передача данных ---
            MPI_Win_fence(0, win1);
            MPI_Win_fence(0, win2);

            if (rank == 0) {
                // Локальное и очень быстрое копирование для самого себя
                for (int i = 0; i < local_count; i++) {
                    int index_needed = partition[i]-1;
                    for (int j = 0; j < max_cols; j++) {
                        local_nbrs[i*max_cols + j] = global_ids_n[index_needed*max_cols + j];
                    }
                }
            } else {
                // Сетевое одностороннее чтение для остальных процессов
                MPI_Get(local_nbrs.data(), local_count*max_cols, MPI_INT, 
                        0, 0, 1, target_indexed_type_int, win1);
            }

            if (rank == 0) {
                // Локальное и очень быстрое копирование для самого себя
                for (int i = 0; i < local_count; i++) {
                    int index_needed = partition[i]-1;
                    for (int j = 0; j < max_cols; j++) {
                        local_eint[i*max_cols + j] = global_eint[index_needed*max_cols + j];
                    }
                }
            } else {
                // Сетевое одностороннее чтение для остальных процессов
                MPI_Get(local_eint.data(), local_count*max_cols, MPI_DOUBLE, 
                        0, 0, 1, target_indexed_type_double, win2);
            }

            MPI_Win_fence(0, win1);
            MPI_Win_fence(0, win2);

            // Освобождаем ресурсы
            MPI_Type_free(&target_indexed_type_int);
            MPI_Win_free(&win1);
            MPI_Type_free(&target_indexed_type_double);
            MPI_Win_free(&win2);

            // calculate coodination (z)
            for (int i = 0; i < local_count; i++) {
                local_ind[partition[i]] = i;
                int z = 0;
                for (int j = 0; j < max_cols; j++) {
                    if (local_nbrs[i * max_cols + j]!=0){
                        z++;
                    }
                }
                local_z[i] = z;
            }

            std::unordered_set<int> seen;
            int cnt = 0;
            for (int i = 0; i < nbrs.size(); i++){
                if (seen.count(nbrs[i])) {
                    continue;
                }
                else {
                    seen.insert(nbrs[i]);
                    local_ind[nbrs[i]] = local_count + cnt;
                    cnt ++;
                }
            } 
        }
    }

    

    int getNbrID(int row, int col) const {
        if (row >= local_count){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        if (col >= local_z[row]){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        return local_nbrs[row * max_cols + col];
    }

    int getNbrLocalIndex(int row, int col) const {
        if (row >= local_count){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        if (col >= local_z[row]){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        int id = local_nbrs[row * max_cols + col];
        if (id==0){
            throw std::runtime_error("Попытка обратиться к несуществующему соседу");
        }
        int res;
        try {
            res = local_ind.at(id);
        }
        catch (const std::exception& err) {
            if (rank==0){
                std::cout << "ERROR at id: " << id << std::endl;
                std::cerr << err.what() << std::endl;
            }
            throw err;
        }
        return res;
    }

    double getNbrEint(int row, int col, int type) const {
        if (row >= local_count){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        if (col >= local_z[row]){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        if (type >= n_solute_pairs){
            throw std::runtime_error("Solute type number exceeds maximum!");
        }
        return local_eint[(row * max_cols + col) * n_solute_pairs + type];
    }
};
