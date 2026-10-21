#!/home/user/miniconda3/bin/python
import numpy as np
from matplotlib import pyplot as plt
import re
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("elements", type=str, nargs="+", help='<chemical_specie_1> <chemical_specie_2> ...')
parser.add_argument("-f", "--fname", type=str, default="data.txt", help='data file')
args = parser.parse_args()

fname = args.fname
elements = args.elements

if len(elements)!=2:
    raise ValueError("currently supports only 2 solute specie")

element_B = elements[0]
element_C = elements[1]


skiprows = 0
with open(fname, "r") as f:
    for line in f:
        if not line.startswith("#"):
            break
        else:
            skiprows+=1

        if "Columns:" in line:
            columns = line.replace('#', '').replace('Columns:', '').strip().split()

df = np.loadtxt(fname, skiprows=skiprows)

ids_c = df[:, columns.index("i")].astype(int)
ids_n = df[:, columns.index("j")].astype(int)

eBC = df[:, columns.index(f"W{element_B}{element_C}")]
eCB = df[:, columns.index(f"W{element_C}{element_B}")]
eBB = df[:, columns.index(f"W{element_B}{element_B}")]
eCC = df[:, columns.index(f"W{element_C}{element_C}")]

eBis = df[:, columns.index(f"Eseg_i({element_B})")]
eCjs = df[:, columns.index(f"Eseg_j({element_C})")]
eBjs = df[:, columns.index(f"Eseg_j({element_B})")]
eCis = df[:, columns.index(f"Eseg_i({element_C})")]


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

with open(f'es_{element_B}.txt', 'w') as f:
    f.write(eBseg_out.rstrip('\n'))

with open(f'es_{element_C}.txt', 'w') as f:
    f.write(eCseg_out.rstrip('\n'))

with open('neighbors.txt', 'w') as f:
    f.write(nbr_out.rstrip('\n'))
    
with open(f'eint_{element_B}_{element_B}.txt', 'w') as f:
    f.write(wBB_out.rstrip('\n'))

with open(f'eint_{element_B}_{element_C}.txt', 'w') as f:
    f.write(wBC_out.rstrip('\n'))

with open(f'eint_{element_C}_{element_B}.txt', 'w') as f:
    f.write(wCB_out.rstrip('\n'))

with open(f'eint_{element_C}_{element_C}.txt', 'w') as f:
    f.write(wCC_out.rstrip('\n'))




