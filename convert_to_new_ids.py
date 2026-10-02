import numpy as np
from copy import deepcopy as dc

eV2kJmol = 96.485
suffix = ''

ids = []
nbrs = []

with open(f"neighbors{suffix}.txt") as f:
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

df = np.loadtxt(f"GBEs{suffix}.txt")
ids_e = df[:, 0]
e1s = df[:, 1]
df = np.loadtxt("bulkEs.txt")
if len(df.shape)==1:
    e0 = df[1]
else:
    e0 = np.mean(df[:, 1])

es = (e1s-e0)*eV2kJmol

es_srt = []
ids_e_srt = []

for i in range(len(es)):
    i0 = np.where(ids_e==ids_arr[i])[0][0]
    es_srt.append(es[i0])
    ids_e_srt.append(ids_e[i0])

ids_e = np.array(ids_e_srt)
es = np.array(es_srt)

assert np.all(ids_arr==ids_e) # проверяем, что порядок индексов в файле с энергией и в файле с соседями совпадает


ids_eint = []
eint = []
Epure = np.loadtxt("pureE.txt")

with open(f"GBEs_int{suffix}.txt") as f:
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
        eint[i][j] = (eint[i][j]+Epure)*eV2kJmol-(es[i]+es[i1]+2*e0*eV2kJmol)
        n0 = len(nbrs[i1])
        nbrs[i1].append(ids[i])
        if len(eint[i1])==n0:
            eint[i1].append(eint[i][j])
        else:
            eint[i1][n0] = eint[i][j]

ids_eint = np.array(ids_eint)

assert np.all(ids_arr==ids_eint) # проверяем, что порядок индексов в файле с энергией и в файле с соседями совпадает
"""
создаем новые индексы от 1 до N_sites
"""
new_ids = np.arange(len(ids))+1
new = dict(zip(ids, new_ids)) # (keys, values)
new_nbrs = []

out = 'central atom. neighbor_ids\n'
out_eint = '# id [Es]\n'
out_e = f'{len(ids)} 2\n'
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

with open(f"new_neighbors{suffix}.txt", 'w') as f:
    f.write(out)

with open(f"new_es{suffix}.txt", 'w') as f:
    f.write(out_e)

with open(f"new_eint{suffix}.txt", 'w') as f:
    f.write(out_eint)
