import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.cluster import SpectralClustering
import os
import networkx as nx
import nxmetis


file_path = "new_neighbors.txt"

# Инициализируем списки для координат связей (строка, колонка)
sources = []
targets = []

with open(file_path) as f:
    for line in f.readlines():
        args = line.replace(' \n', '').replace('\n', '').split(' ')
        id_c = int(args[0])
        ids_n = list(map(int, args[1:]))
        ids_c = [id_c]*len(ids_n)
        sources.extend(ids_c)
        targets.extend(ids_n)

print("Файл прочитан. Формируем разреженную матрицу...")
num_nodes = max(max(sources), max(targets)) 
sources = np.array(sources)-1
targets = np.array(targets)-1

# Собираем разреженную матрицу CSR. Веса связей принудительно делаем равными 1
# (так как для кластеризации веса не нужны, оцениваем только факт связи)
sparse_matrix = sp.coo_matrix(
    (np.ones(len(sources)), (sources, targets)), 
    shape=(num_nodes, num_nodes)
).tocsr()

# Считаем максимальное координационное число z_max для разреженной матрицы
# (Вместо sum(axis=0) для разреженной матрицы берем количество элементов в строках)
degrees = np.bincount(sources)
z_max = degrees.max()
print('Maximum coordination number:', z_max)
print(f"Всего вершин: {sparse_matrix.shape[0]}, Всего связей: {sparse_matrix.nnz}")
print("-" * 50)

# --- НАСТРОЙКА И ЗАПУСК КЛАСТЕРИЗАЦИИ ---

n_clusters = 20  # Ваше количество блоков

# Превращаем вашу разреженную матрицу в граф NetworkX
G = nx.from_scipy_sparse_array(sparse_matrix)
G.node = G.nodes 

print("Запуск METIS для разбиения...")
# Указываем целевое количество блоков (20)
# METIS гарантирует практически 100% равномерность по числу вершин
edge_cut, parts = nxmetis.partition(G, n_clusters)

# Конвертируем результат в массив labels, как у вас было в sklearn
labels = np.zeros(sparse_matrix.shape[0], dtype=int)
for cluster_id, nodes_list in enumerate(parts):
    labels[nodes_list] = cluster_id

print("Разбиение METIS завершено.")

""" clustering = SpectralClustering(
    n_clusters=n_clusters,
    affinity='precomputed',
    assign_labels='kmeans',
    eigen_solver='arpack',  # КРИТИЧЕСКИ ВАЖНО: включает быстрый алгоритм для разреженных матриц
    random_state=42
)

# Передаем разреженную матрицу напрямую в sklearn
print("Запуск спектральной кластеризации...")
labels = clustering.fit_predict(sparse_matrix)
print("Кластеризация завершена!") """

def find_boundary_nodes_sparse(sparse_matrix, labels):
    """
    Быстро находит межблочные связи для разреженной матрицы.
    """
    # Получаем координаты всех связей (строка, колонка) напрямую из CSR матрицы
    # sparse_matrix.tocoo() делает это мгновенно без копирования данных
    coo = sparse_matrix.tocoo()
    rows, cols = coo.row, coo.col
    
    # Исключаем петли (связи вершины с самой собой), если они есть
    valid = rows != cols
    rows, cols = rows[valid], cols[valid]
    
    # Находим связи, у которых блоки на концах не совпадают
    is_inter_block = labels[rows] != labels[cols]
    
    # Возвращаем массивы: (откуда_id, куда_id) для внешних связей
    return rows[is_inter_block], cols[is_inter_block]

    
partition_path = 'partitions'
os.makedirs(partition_path, exist_ok=True)
# Получаем массивы межблочных связей с помощью новой функции
boundary_sources, boundary_targets = find_boundary_nodes_sparse(sparse_matrix, labels)

# Для быстрого доступа сгруппируем цели (targets) по исходным вершинам (sources)
from collections import defaultdict
foreign_neighbors_dict = defaultdict(list)
for src, tgt in zip(boundary_sources, boundary_targets):
    foreign_neighbors_dict[src].append(tgt)

zs = list(map(len, foreign_neighbors_dict.values()))
z_max = np.max(zs)
print('calculated z_max for interblock bonds:', z_max)

# Количество знаков для красивого форматирования savetxt
ndigits = int(np.ceil(np.log10(1 + sparse_matrix.shape[0])))

for cluster_id in range(n_clusters):
    nodes = np.where(labels == cluster_id)[0]
    
    # Создаем пустую матрицу для выгрузки (строк = сколько узлов в блоке, колонок = 1 + 2*z_max)
    out = np.zeros((len(nodes), 2 * z_max + 1), dtype=int)
    out[:, 0] = nodes + 1  # Первый столбец — ID текущей вершины (1-based)

    for i, node in enumerate(nodes):
        # Достаем "чужих" соседей из нашего быстрого словаря
        foreign_neighbors = foreign_neighbors_dict.get(node, [])
        
        if foreign_neighbors:
            foreign_neighbors = np.array(foreign_neighbors)
            
            # Заполняем ID чужих соседей (переводим в 1-based индекс)
            out[i, 1 : 1 + len(foreign_neighbors)] = foreign_neighbors + 1
            
            # Заполняем ID блоков, в которых они сидят (переводим в 1-based индекс)
            foreign_blocks = labels[foreign_neighbors] + 1
            out[i, 1 + z_max : 1 + z_max + len(foreign_blocks)] = foreign_blocks

    # Сохраняем файл для текущего блока
    np.savetxt(
        os.path.join(partition_path, f'{cluster_id + 1}.txt'), 
        out, 
        fmt=f'%-{ndigits}d', 
        header='site_id neighbor_ids ... neighbor_blocks ...'
    )

print("Файлы блоков успешно сохранены!")


def check_cluster_connections_sparse(sparse_matrix, labels):
    """
    Считает количество связей внутри блоков и между ними без перевода в плотную матрицу.
    """
    # Общее число уникальных ребер в разреженном графе
    # (так как граф неориентированный, делим общее число ненулевых элементов без учета диагонали на 2)
    # На всякий случай обнуляем диагональ у копии для корректного подсчета связей
    diag_mask = sparse_matrix.diagonal() != 0
    total_edges = int((sparse_matrix.nnz - np.sum(diag_mask)) / 2)
    
    print(f"Всего уникальных связей в графе: {total_edges}")
    print("-" * 50)
    
    unique_labels = np.unique(labels)
    edges_inside_total = 0
    
    # Переводим в CSR для быстрой нарезки строк и столбцов
    csr_mat = sparse_matrix.tocsr()
    
    # 1. Считаем связи внутри каждого блока
    for cluster_id in unique_labels:
        # Находим индексы вершин текущего блока
        node_indices = np.where(labels == cluster_id)[0]
        nodes_count = len(node_indices)
        
        # Магия scipy: извлекаем подматрицу только для нужных строк и столбцов
        sub_matrix = csr_mat[node_indices, :][:, node_indices]
        
        # Считаем ребра внутри подматрицы (вычитаем её собственную диагональ, если она есть)
        sub_diag_sum = np.sum(sub_matrix.diagonal() != 0)
        edges_inside = int((sub_matrix.nnz - sub_diag_sum) / 2)
        
        edges_inside_total += edges_inside
        print(f"Блок {cluster_id + 1} ({nodes_count} верш.): связей внутри = {edges_inside}")
        
    # 2. Считаем связи МЕЖДУ разными блоками
    edges_between = total_edges - edges_inside_total
    
    print("-" * 50)
    print(f"МЕЖДУ РАЗНЫМИ БЛОКАМИ осталось связей: {edges_between}")
    print(f"Процент «разрезанных» связей: {(edges_between / total_edges) * 100:.2f}%")

# Вызов проверки
check_cluster_connections_sparse(sparse_matrix, labels)
