#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <stdexcept>

class PartitionLoader {
private:
    int cols;
    int rows;

public:
    int z_max;
    int z_tot;
    int nsites;
    std::vector<int> data;
    std::vector<int> partition;
    std::vector<int> z;
    std::vector<int> srcs;
    std::vector<int> nbrs;
    std::vector<int> blks;

    PartitionLoader()
    {
        //pass
    }

    void load(const std::string& filename) {
        int total_rows = -1;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("PartitionLoader: Не удалось открыть файл " + filename);
        }

        z_tot = 0;
        std::string line;
        cols = 0;
        while (std::getline(file, line)) {
            if (total_rows >= 0){
                std::stringstream ss(line);
                int number;
                int count = 0;

                while (ss >> number) {
                    data.push_back(number);
                    count++;
                    if (number!=0){
                        z_tot ++; 
                    }
                }
                if (cols == 0 && count>0)
                {
                    cols = count;
                    assert((cols-1)%2==0);
                    z_max = (cols-1)/2;
                }
            }
    
            total_rows++;
        }
        rows = total_rows;
        z_tot = z_tot/2; // double counting (nbrID + blockID)
        file.close();

        for (int i = 0; i < rows; i++){
            int src = getID(i);
            partition.push_back(src);
            int _z = 0;
            int id;
            int block;
            for (int j = 0; j < z_max; j++){
                id = getNbrID(i, j);
                block = getNbrBlock(i, j);
                if (id!=0){
                    _z ++;
                    srcs.push_back(src);
                    nbrs.push_back(id);
                    blks.push_back(block);
                }
            }
            z.push_back(_z);
        }
        nsites = partition.size();
    }

    int getID(int row) const {
        if (row >= rows){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        return data[row * cols];
    }

    int getNbrID(int row, int col) const {
        if (col >= z_max){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        if (row >= rows){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        return data[row * cols + col + 1];
    }

    int getNbrBlock(int row, int col) const {
        if (col >= z_max){
            throw std::runtime_error("Neighbor number exceeds maximum!");
        }
        if (row >= rows){
            throw std::runtime_error("Central atom number exceeds maximum!");
        }
        return data[row * cols + col + z_max + 1];
    }


};
