#pragma once // Гарантирует, что файл подключится только один раз

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

// Функция для записи массива в текстовом формате
bool save_vector(const std::string& filename, const std::vector<double>& array) {
    std::ofstream out(filename); // Открываем файл для записи
    if (!out.is_open()) return false;

    // Сначала записываем размер массива, чтобы потом было удобно читать
    out << array.size() << "\n";

    // Записываем элементы через пробел
    for (size_t i = 0; i < array.size(); ++i) {
        out << array[i] << (i == array.size() - 1 ? "" : " ");
    }
    out << "\n";
    return true;
}


template <typename T>
class Matrix {
private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<T> data_;

public:
    // Конструкторы
    Matrix() = default;
    Matrix(int rows, int cols, T default_value = T()) 
        : rows_(rows), cols_(cols), data_(rows * cols, default_value) {
        assert(rows > 0 && cols > 0 && "Размеры должны быть больше нуля!");
    }

    // Доступ к элементам
    T& operator()(int r, int c) {
        assert(r >= 0 && r < rows_ && c >= 0 && c < cols_ && "Индекс строк/столбцов вышел за границы!");
        return data_[r * cols_ + c];
    }

    const T& operator()(int r, int c) const {
        assert(r >= 0 && r < rows_ && c >= 0 && c < cols_ && "Индекс строк/столбцов вышел за границы!");
        return data_[r * cols_ + c];
    }

    // Вспомогательные методы
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    // Работа с файлами (Текстовый формат)
    bool save_to_text(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) return false;

        out << rows_ << " " << cols_ << "\n";
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                out << (*this)(r, c) << (c == cols_ - 1 ? "" : " ");
            }
            out << "\n";
        }
        return true;
    }

    bool load_from_text(const std::string& filename) {
        std::ifstream in(filename);
        if (!in.is_open()) return false;

        int r = 0, c = 0;
        in >> r >> c;
        if (r <= 0 || c <= 0) return false;

        rows_ = r;
        cols_ = c;
        data_.assign(rows_ * cols_, T());

        for (int i = 0; i < rows_ * cols_; ++i) {
            in >> data_[i];
        }
        return true;
    }

    // Работа с файлами (Бинарный формат)
    bool save_to_binary(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary); 
        if (!out.is_open()) return false;

        out.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
        out.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
        out.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(T));
        return true;
    }

    bool load_from_binary(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;

        in.read(reinterpret_cast<char*>(&rows_), sizeof(rows_));
        in.read(reinterpret_cast<char*>(&cols_), sizeof(cols_));

        data_.assign(rows_ * cols_, T());
        in.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(T));
        return true;
    }

    Matrix<T> sum_axis_0() const {
    assert(rows_ > 0 && cols_ > 0 && "Матрица пуста!");
    
    // Результат — матрица из 1 строки и cols_ столбцов
    Matrix<T> result(1, cols_, T()); 
    
    // Суммируем элементы каждого столбца
    for (int c = 0; c < cols_; ++c) {
        T sum = T();
        for (int r = 0; r < rows_; ++r) {
            sum += (*this)(r, c);
        }
        result(0, c) = sum;
    }
    
    return result;
}
};
