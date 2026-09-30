#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "matrix.hpp"

class DataLoader {
private:
    int max_cols;
    int n_solute_pairs;

public:
    std::vector<int> ids_n;
    std::vector<int> z;
    std::vector<double> eint;
    int total_site_types;

    DataLoader(int _max_cols = 20, int _n_solutes = 1) 
        : max_cols(_max_cols), n_solute_pairs(_n_solutes)
    {
    }

    // Главный метод для загрузки и распределения данных
    void loadAndDistribute(const std::string& nbr_filename, 
                            std::vector<std::string> eint_filenames) 
        {
        int total_rows = 0;

        /// read new_neighbors.txt
        std::ifstream file1(nbr_filename);
        if (!file1.is_open()) {
            throw std::runtime_error("DataLoader: Не удалось открыть файл " + nbr_filename);
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
                        ids_n.push_back(number);
                        count++;
                    }
                }
                // Выравнивание строки нулями
                while (count < max_cols) {
                    ids_n.push_back(0);
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
                throw std::runtime_error("DataLoader: Не удалось открыть файл " + eint_filename);
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
                            eint.push_back(energy);
                            count++;
                        }
                    }
                    // Выравнивание строки нулями
                    while (count < max_cols) {
                        eint.push_back(0.0);
                        count++;
                    }
                
                total_rows2++;
            }
            file2.close();
            if (total_rows2!=total_rows){
                throw std::runtime_error("DataLoader: number of rows in " + nbr_filename + " and " + eint_filename + " does not match!");
            }
        }
        

        z.resize(total_site_types);
        for (int i = 0; i < total_site_types; i++) {
            z[i] = 0;
            for (int j = 0; j < max_cols; j++) {
                int index = i * max_cols + j;
                if (ids_n[index]!=0){
                    z[i]++;
                }
            }
        }
        
    }

    

    int getNbrID(int row, int col) const {
        if (row >= total_site_types){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        if (col >= z[row]){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        return ids_n[row * max_cols + col];
    }

    int getNbrIndex(int row, int col) const {
        int id = getNbrID(row, col);
        if (id==0){
            throw std::runtime_error("Попытка обратиться к несуществующему соседу");
        }
        return id-1;
    }

    double getNbrEint(int row, int col, int type) const {
        if (row >= total_site_types){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        if (col >= z[row]){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        if (type >= n_solute_pairs){
            throw std::runtime_error("Solute type number exceeds maximum!");
        }
        return eint[type * total_site_types * max_cols + row * max_cols + col];
    }
};
