import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

fname = "data.txt"

eAA = None
e0B = None
e0C = None
blk_idx = None
with open(fname, "r") as f:
    for line in f:
        # Если дошли до строк без решетки (начались данные), прекращаем парсить шапку
        if not line.startswith("#"):
            break
        
        # 1. Ищем Eaa
        if "Eaa =" in line:
            match = re.search(r"Eaa\s*=\s*([-\d.]+)", line)
            if match:
                eAA = float(match.group(1))
                
        # 2. Ищем E(Ni...)
        elif "E(Ni@" in line:
            match = re.search(r"=\s*([-\d.]+)$", line)
            if match:
                e0C = float(match.group(1))
                
        # 3. Ищем E(Cu...)
        elif "E(Cu@" in line:
            match = re.search(r"=\s*([-\d.]+)$", line)
            if match:
                e0B = float(match.group(1))
                
        # 4. Ищем blk_idx
        elif "blk_idx =" in line:
            match = re.search(r"blk_idx\s*=\s*(\d+)", line)
            if match:
                blk_idx = int(match.group(1))

df = pd.read_csv(fname, header=0, sep='\s+', comment='#')

ids_c = df["i"].values.astype(int)
ids_n = df["j"].values.astype(int)

eBC = df["WCuNi"].values
eCB = df["WNiCu"].values
eBB = df["WCuCu"].values
eCC = df["WNiNi"].values

eBi1s = df["Eseg_i(Cu)"].values
eCj1s = df["Eseg_j(Ni)"].values
eBj1s = df["Eseg_j(Cu)"].values
eCi1s = df["Eseg_i(Ni)"].values

eBi2s = df["Eseg_i(Cu).1"].values
eCj2s = df["Eseg_j(Ni).1"].values
eBj2s = df["Eseg_j(Cu).1"].values
eCi2s = df["Eseg_i(Ni).1"].values

eBis = (eBi1s + eBi2s)/2
eCis = (eCi1s + eCi2s)/2
eBjs = (eBj1s + eBj2s)/2
eCjs = (eCj1s + eCj2s)/2


ids1 = np.unique(ids_c)
ids2 = np.unique(ids_n)
ids = np.unique(np.concatenate((ids1, ids2)))
ids_nbr = [[] for _ in range(len(ids))]
nbr_out = "central_atom neighbor_ids...\n"
eseg_out = f"{len(ids)} 3\n"
eBseg_out = f"{len(ids)} 2\n"
eCseg_out = f"{len(ids)} 2\n"

eBs, eCs = np.zeros(len(ids)), np.zeros(len(ids))

new_ids = np.arange(len(ids))
new = dict(zip(ids, new_ids))

inds = np.arange(len(ids_c))

wBB_out = "# id [Es]\n"
wBC_out = "# id [Es]\n"
wCB_out = "# id [Es]\n"
wCC_out = "# id [Es]\n"
wBB = [[] for _ in range(len(ids))]
wCC = [[] for _ in range(len(ids))]
wBC = [[] for _ in range(len(ids))]
wCB = [[] for _ in range(len(ids))]


for id in ids:
    mask = (ids_c == id)
    inds_i = inds[mask]
    new_ind = new[id]
    old_ids_nbr = ids_n[mask]
    ids_nbr[new_ind] = [new[old_ids_nbr[j]] for j in range(len(old_ids_nbr))]
    wBB[new_ind] = [eBB[inds_i[j]] for j in range(len(old_ids_nbr))]
    wBC[new_ind] = [eBC[inds_i[j]] for j in range(len(old_ids_nbr))]
    wCB[new_ind] = [eCB[inds_i[j]] for j in range(len(old_ids_nbr))]
    wCC[new_ind] = [eCC[inds_i[j]] for j in range(len(old_ids_nbr))]

    mask_n = (ids_n == id)
    for e in ids_c[mask_n]:
        ids_nbr[new_ind].append(new[e])

    for j in inds[mask_n]:
        wBB[new_ind].append(eBB[j])
        wBC[new_ind].append(eCB[j])
        wCB[new_ind].append(eBC[j])
        wCC[new_ind].append(eCC[j])

    eBs[new_ind] = np.mean(np.concatenate((eBis[mask], eBjs[mask_n])))
    eCs[new_ind] = np.mean(np.concatenate((eCis[mask], eCjs[mask_n])))

    nbr_out += f"{1+new_ind} "+" ".join(str(n + 1) for n in ids_nbr[new_ind])+"\n"
    eseg_out += f"{1+new_ind} {eBs[new_ind]} {eCs[new_ind]}\n"
    eBseg_out += f"{1+new_ind} {eBs[new_ind]}\n"
    eCseg_out += f"{1+new_ind} {eCs[new_ind]}\n"

    wBB_out += f"{1+new_ind} "+" ".join(list(map(str, wBB[new_ind])))+"\n"
    wBC_out += f"{1+new_ind} "+" ".join(list(map(str, wBC[new_ind])))+"\n"
    wCB_out += f"{1+new_ind} "+" ".join(list(map(str, wCB[new_ind])))+"\n"
    wCC_out += f"{1+new_ind} "+" ".join(list(map(str, wCC[new_ind])))+"\n"

with open('es.txt', 'w') as f:
    f.write(eseg_out.rstrip('\n'))

with open('es_0.txt', 'w') as f:
    f.write(eBseg_out.rstrip('\n'))

with open('es_1.txt', 'w') as f:
    f.write(eCseg_out.rstrip('\n'))

with open('neighbors.txt', 'w') as f:
    f.write(nbr_out.rstrip('\n'))
    
with open('eint_0_0.txt', 'w') as f:
    f.write(wBB_out.rstrip('\n'))

with open('eint_0_1.txt', 'w') as f:
    f.write(wBC_out.rstrip('\n'))

with open('eint_1_0.txt', 'w') as f:
    f.write(wCB_out.rstrip('\n'))

with open('eint_1_1.txt', 'w') as f:
    f.write(wCC_out.rstrip('\n'))




