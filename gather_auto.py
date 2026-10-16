#!/home/user/miniconda3/bin/python
"""
TODO надо проверить, правильность кода, особенно совпадает ли порядок mu с порядком Х
"""
import glob
import os
import re
import numpy as np
from tqdm import tqdm


def process_mc_data():
    # Регулярное выражение для поиска val (включая целые, дробные и отрицательные числа)
    pattern = re.compile(r"mc_output_mu((_-?\d+\.?\d*)+)\.txt")

    data_list = []

    # Находим все txt файлы в текущей директории
    all_files = glob.glob("*.txt")
    nfiles = 0
    mc_output_files = []
    matches = []

    for file_path in all_files:
        match = pattern.match(file_path)
        if match:
            nfiles += 1
            mc_output_files.append(file_path)
            matches.append(match.group(1))

    read_metadata = False

    for cnt, file_path in enumerate(tqdm(mc_output_files)):
        # Извлекаем значение val и преобразуем во float
        vals_fname = [float(item) for item in matches[cnt].lstrip('_').split('_')]
        
        try:
            # Читаем только названия колонок (первую строку), чтобы узнать их индексы
            with open(file_path, "r") as f:
                metadata = f.readline().split(";")  # mc_output; elements Ag Cu Ni ; mu 0 -7.7551 -20 ; kT 3
                metadata = [e.strip() for e in metadata]

                header = f.readline().split()

            if not header:
                print(f"Пропущен файл {file_path}: пустой заголовок.")
                continue

            kT = float([e for e in metadata if e.startswith("kT")][0].split()[1])

            # Находим все колонки вида X_* динамически
            x_cols = [col for col in header if col.startswith("X_")]
            elements = [col.replace("X_", "") for col in x_cols]

            elements_mu = [e for e in metadata if e.startswith("elements")][0].split()[1:]
            mu = list(map(float, [e for e in metadata if e.startswith("mu")][0].split()[2:]))
            if mu:
                vals = mu
                assert set(elements)==set(elements_mu)
            else:
                vals = vals_fname
                elements_mu = elements
                if not read_metadata:
                    print("[WARNING]: Can not found mu in file's metadata, use values from filename. The order of mu columns may be incorrect!")           


            if not read_metadata:
                elements0 = elements
                elements_mu0 = elements_mu
                kT0 = kT
                read_metadata = True

            else:
                assert elements==elements0
                assert elements_mu==elements_mu0
                assert kT == kT0


            # Если нет колонок типа X_*, пропускаем файл
            if not x_cols:
                print(
                    f"Пропущен файл {file_path}: отсутствуют колонки 'X_*'."
                )
                continue

            # Находим индексы ВСЕХ колонок включая per_site_energy
            indices = {}
            for i, col in enumerate(header):
                if col.startswith("X_") or col == "per_site_energy":
                    indices[col] = i

            energy_idx = indices["per_site_energy"]
            
            all_indices = x_cols + ["per_site_energy"]  # строки
            int_indices = np.array([indices[col] for col in all_indices])

            # Загружаем данные с помощью np.loadtxt
            # skiprows=1 пропускает строку с заголовками
            data = np.loadtxt(
                file_path, skiprows=2, usecols=int_indices
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
            
            # x_means - это все столбцы кроме последнего (energy)
            n_x_cols = len(x_cols)
            x_means = mean_values[:n_x_cols]
            mean_energy = mean_values[-1]

            # Формируем список для одного файла: [mu, mean_X_1, ..., mean_per_site_energy]
            data_list.append(vals + list(x_means) + [mean_energy])
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    if not data_list:
        print("Файлы формата 'mc_output_mu_val.txt' не найдены.")
        return

    # Превращаем список в numpy массив
    result_array = np.array(data_list)

    # Сортируем массив по первой колонке (по возрастанию vals)
    # lexsort требует фиксированное число ключей, поэтому сортируем только по first column
    sorted_indices = np.lexsort(result_array.T[:len(vals)][::-1])
    result_array = result_array[sorted_indices]
    
    # Сохраняем итоговые зависимости в файл с помощью np.savetxt
    output_filename = "mc_auto_output.txt"
    
    # Формируем заголовок динамически на основе найденных элементов
    # Первые две колонки: mu_1, mu_2 (значения из имени файла)
    # Далее: mean_X_*, ..., mean_per_site_energy
    header_parts = [f"mu_{elements_mu0[i+1]}" for i in range(len(elements_mu0)-1)]
    
    # Добавляем mean_ для каждого элемента X_*
    for col_name in x_cols:
        header_parts.append(f"mean_{col_name}")
    
    # Добавляем энергию в конце
    header_parts.append("mean_per_site_energy")
    header_text = f"kT {kT0};\n"+"\t".join(header_parts)
    
    np.savetxt(
        output_filename,
        result_array,
        fmt="%.6f",
        delimiter="\t",
        header=header_text,
        comments="",
    )

    print(f"\nУспешно сохранено! Результаты записаны в: {output_filename}")


if __name__ == "__main__":
    process_mc_data()
