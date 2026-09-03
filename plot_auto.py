import glob
import os
import re
import numpy as np


def process_mc_data():
    # Регулярное выражение для поиска val (включая целые, дробные и отрицательные числа)
    pattern = re.compile(r"mc_output_mu_(-?\d+\.?\d*)\.txt")

    data_list = []

    # Находим все txt файлы в текущей директории
    all_files = glob.glob("*.txt")

    for file_path in all_files:
        match = pattern.match(file_path)
        if match:
            
            # Извлекаем значение val и преобразуем во float
            val = float(match.group(1))
            print(val)
            try:
                # Читаем только названия колонок (первую строку), чтобы узнать их индексы
                with open(file_path, "r") as f:
                    f.readline() # пропуск первой строки
                    header = f.readline().split()

                if not header:
                    print(f"Пропущен файл {file_path}: пустой заголовок.")
                    continue

                # Находим индексы нужных колонок
                try:
                    idx_x1 = header.index("X_1")
                    idx_energy = header.index("per_site_energy")
                except ValueError:
                    print(
                        f"Пропущен файл {file_path}: отсутствуют колонки 'X_1' или 'per_site_energy'."
                    )
                    continue

                # Загружаем данные с помощью np.loadtxt
                # skiprows=1 пропускает строку с заголовками
                # usecols считывает только нужные нам столбцы
                data = np.loadtxt(
                    file_path, skiprows=2, usecols=(idx_x1, idx_energy)
                )

                # Если в файле была всего одна строка данных, numpy сделает массив одномерным
                if data.ndim == 1:
                    data = np.atleast_2d(data)

                num_rows = data.shape[0]
                if num_rows == 0:
                    print(f"Пропущен файл {file_path}: нет данных.")
                    continue

                # Вычисляем количество строк для последних 20%
                tail_size = int(num_rows * 0.2)
                if tail_size == 0:
                    tail_size = 1  # Берем минимум одну строку, если файл очень короткий

                # Срезаем последние 20% строк
                tail_data = data[-tail_size:]

                # Считаем среднее арифметическое по колонкам (ось 0)
                mean_values = np.mean(tail_data, axis=0)
                mean_x1 = mean_values[0]
                mean_energy = mean_values[1]

                # Сохраняем результат в список
                data_list.append([val, mean_x1, mean_energy])

            except Exception as e:
                print(f"Ошибка при обработке файла {file_path}: {e}")

    if not data_list:
        print("Файлы формата 'mc_output_mu_val.txt' не найдены.")
        return

    # Превращаем список в numpy массив
    result_array = np.array(data_list)

    # Сортируем массив по первой колонке (по возрастанию val)
    result_array = result_array[result_array[:, 0].argsort()]

    # Сохраняем итоговые зависимости в файл с помощью np.savetxt
    output_filename = "mc_auto_output.txt"
    header_text = "val\tmean_X_1\tmean_per_site_energy"

    np.savetxt(
        output_filename,
        result_array,
        fmt="%.6f",
        delimiter="\t",
        header=header_text,
        comments="",
    )

    print(f"\nУспешно сохранено! Результаты записаны в: {output_filename}")
    print(header_text)
    print(result_array)


if __name__ == "__main__":
    process_mc_data()
