import numpy as np
from copy import deepcopy as dc


ids = []
nbrs = []

with open("neighbors.txt") as f:
    cnt = 0
    for line in f:
        if cnt == 0:
            cnt += 1
            continue
        args = line.replace(' \n', '').replace('\n', '').split(' ')
        id = int(args[0])
        ids.append(id)
        nbrs.append(list(map(int, args[2:])))

nbrs_c = dc(nbrs)
ids_arr = np.array(ids)

ids_eint = []
eint = []
with open("GBEs_int.txt") as f:
    cnt = 0
    for line in f:
        if cnt == 0:
            cnt += 1
            continue
        args = line.replace(' \n', '').replace('\n', '').split(' ')
        id = int(float(args[0]))
        ids_eint.append(id)
        eint.append(list(map(float, args[1:])))

"""
восстанавливаем симметричные связи, которые были отброшены для избежания двойного расчета в МД
"""
for i in range(len(nbrs_c)):
    for j in range(len(nbrs_c[i])):
        e = nbrs_c[i][j]
        i1 = np.where(ids_arr==e)[0][0]
        n0 = len(nbrs[i1])
        nbrs[i1].append(ids[i])
        if len(eint[i1])==n0:
            eint[i1].append(eint[i][j])
        else:
            eint[i1][n0] = eint[i][j]


srt = np.argsort(ids)
ids_arr = np.array(ids)[srt]
nbrs = np.array(nbrs, dtype=object)[srt]

srt = np.argsort(ids_eint)
ids_eint = np.array(ids_eint)[srt]
eint = np.array(eint, dtype=object)[srt]

assert np.all(ids_arr==ids_eint) # проверяем, что порядок индексов в файле с энергией и в файле с соседями совпадает

df = np.loadtxt("GBEs.txt")
ids_e = df[:, 0]
es = df[:, 1]

srt = np.argsort(ids_e)
ids_e = np.array(ids_e)[srt]
es = np.array(es)[srt]

assert np.all(ids_arr==ids_e) # проверяем, что порядок индексов в файле с энергией и в файле с соседями совпадает

"""
создаем новые индексы от 1 до N_sites
"""
new_ids = np.arange(len(ids))+1
new = dict(zip(ids, new_ids)) # (keys, values)
new_nbrs = []

out = 'central atom. neighbor_ids\n'
out_eint = '# id [Es]\n'
out_e = '# id E\n'
for i in range(len(ids)):
    new_nbr = []
    out += f"{i+1}"
    out_e += f"{i+1} {es[i]}"
    out_eint += f"{i+1}"
    for j, id in enumerate(nbrs[i]):
        new_nbr.append(new[id])
        out += f" {new[id]}"
        out_eint += f" {eint[i][j]}"
    new_nbrs.append(new_nbr)
    if i < len(ids)-1:
        out += "\n"
        out_e += "\n"
        out_eint += "\n"    

with open("new_neighbors.txt", 'w') as f:
    f.write(out)

with open("new_es.txt", 'w') as f:
    f.write(out_e)

with open("new_eint.txt", 'w') as f:
    f.write(out_eint)
