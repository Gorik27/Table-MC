import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dc
from numba import njit, prange
import numba as nb
from itertools import chain, repeat, zip_longest

@nb.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape, dtype=nb.boolean).ravel()
    ar = a.ravel()
    b = set(b)
    for i in prange(out.shape[0]):
        if ar[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out.reshape(a.shape)

eV2kJmol = 96.485
z_max = 20

"""
load data
"""
path = '.'

data = np.loadtxt(f'{path}/new_es.txt', skiprows=1)
data = data[data[:, 0]!=0]
ids = data[:, 0]
Eseg = data[:, 1]

ids_n = []
ids_c = []
with open(f'{path}/new_neighbors.txt', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            df = line.replace('\n', '').split(' ')
            ids_c.append(int(df[0])) 
            ids_n.append(np.array(df[1:]).astype(int))

w = []
with open(f'{path}/new_eint.txt', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            df = line.replace('\n', '').split(' ')
            assert ids_c[i-1] == int(df[0]) 
            w.append(np.array(df[1:]).astype(float))
                
def to_np(arr):
    padded = list(zip_longest(*arr, fillvalue=0))
    return np.array(padded).T

ids_c = np.array(ids_c)
ids_n = to_np(ids_n)
w = to_np(w)

wavg = np.sum(w, axis = 1)
corr = np.mean(np.corrcoef(wavg,Eseg)[0,1])
print('correlation between wavg and Eseg: ',  corr)

"""
simple ordering
"""

Eselected = dc(Eseg)
wselected = dc(w)
ids_c_selected = dc(ids_c)
ids_n_selected = dc(ids_n)

Er = dc(Eseg)
print('ordering...')

@njit
def renorm(Er, wselected, Eselected, ids_c_selected, ids_n_selected):
    for i in range(0, len(Er)):
        if i>0:
            ids_filled = ids_c_selected[i-1]
            msk = (ids_n_selected[i:]==ids_filled)
            O = np.sum(wselected[i:]*msk, axis=1)
        else:
            O = np.sum(wselected[i:]*np.zeros_like(wselected), axis=1)
        Eselected[i:] = Eselected[i:] + O
        i0 = np.argmin(Eselected[i:])
        t0 = Eselected[i:][i0]
        Er[i] = t0
        i0 = i+i0 
        wselected[i0], wselected[i] = wselected[i].copy(), wselected[i0].copy()
        Eselected[i0], Eselected[i] = Eselected[i], Eselected[i0]
        ids_c_selected[i0], ids_c_selected[i] = ids_c_selected[i], ids_c_selected[i0]
        ids_n_selected[i0], ids_n_selected[i] = ids_n_selected[i].copy(), ids_n_selected[i0].copy()
    return
    

renorm(Er, wselected, Eselected, ids_c_selected, ids_n_selected)
"""
conglomerate ordering
"""
plot_each = 5000

E_s = dc(Er)
F_s = np.ones(Er.shape, dtype=int)
w_n = dc(wselected) 


ids_c_s = -np.ones((len(Er), len(Er)), dtype=int)
ids_c_s[:, 0] = ids_c_selected

@njit
def conglomerate_interaction(ids_c_s, i0):
    O = 0
    lst = ids_c_s[i0-1]
    lst = lst[lst!=-1]
    for ind in lst:
        pind = np.where(ids_c_selected==ind)
        mask = isin(ids_n_selected[pind], ids_c_s[i0]) # is neighbors of [i-1] the member of conglomerate [i]?
        O += np.sum(w_n[pind]*mask) # bonds with current site
    return O

cnt = 0
@njit
def permutation(E_s, F_s, ids_c_s, cnt):
    change_flag = False
    for i in range(len(Er)-1-cnt, 0, -1): # reverse order
        if E_s[i] < E_s[i-1]:
            change_flag = True
            
            O = conglomerate_interaction(ids_c_s, i)
            
            if E_s[i] - O/F_s[i] < E_s[i-1] + O/F_s[i-1]: # case when solutes does not form a cluster
                # replace it with corresponding changes in bonds energy
                
                # E_s
                t = E_s[i]
                E_s[i] = E_s[i-1] + O/F_s[i-1]
                E_s[i-1] = t - O/F_s[i]
                
                # F_s
                t = F_s[i]
                F_s[i] = F_s[i-1]
                F_s[i-1] = t
                
                # ids_c_s
                t = ids_c_s[i].copy()
                ids_c_s[i] = ids_c_s[i-1]
                ids_c_s[i-1] = t
                
            else: # solutes form a cluster
                E2 = (E_s[i-1]*F_s[i-1] + E_s[i]*F_s[i])/(F_s[i-1]+F_s[i])
                cnt += 1
                # combine elements into one and shift right side of array
                
                # Combine
                E_s[i-1] = E2 # E_s
                F_s[i-1] += F_s[i] # F_s
                
                # ids_c_s
                ids3 = np.array(list(set(ids_c_s[i-1]).union(set(ids_c_s[i]))))
                ids3 = ids3[ids3!=-1]
                ids_c_s[i-1, :len(ids3)] = ids3 
                
                # Shift
                
                # E_s
                E_s[i:] = np.roll(E_s[i:], -1)
                E_s[-1] = 0
                
                # F_s
                F_s[i:] = np.roll(F_s[i:], -1)
                F_s[-1] = 0
                
                # ids_c_s
                for j in range(i, len(ids_c_s)-1):
                    ids_c_s[j] = ids_c_s[j+1]
                ids_c_s[-1] = -np.ones(len(Er))
                break
        elif change_flag:
            i0 = i
            break
    return i0, change_flag, cnt
    
change_flag = True
iteration = 0
i0 = len(Er)
print('conglomerate ordering...')

while change_flag:
    iteration += 1
    i0, change_flag, cnt = permutation(E_s, F_s, ids_c_s, cnt)
    if iteration % plot_each == 0:
        print(f'iteration #{iteration}, last step at {i0}/{len(Er)} site')
        plt.plot(E_s)
        plt.xlabel('site number')
        plt.ylabel('seg. energy')
        plt.title('conglomerate ordering')
        plt.savefig(f'plot_ordering.png')
        plt.clf()
print('done')
#%%
Ehist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(F_s, E_s)))
Fhist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(F_s, F_s)))

i = 0
j = 0
ids_hist = []
while i<len(E_s):
    if ids_c_s[i, j] == -1:
        i += 1
        j = 0
    else:
        ids_hist.append(ids_c_s[i, j])
        j += 1
        if j==ids_c_s.shape[1]:
            j = 0
            i += 1

out = np.array([ids_hist, Ehist]).transpose()
np.savetxt(f'es_mod.txt', out, header='id e')

plt.hist(Ehist, bins=50, density=True, alpha=1, label='modified spectrum')
plt.hist(Eseg, bins=50, alpha=0.4, density=True, label='original spectrum')
plt.xlabel('$E_{seg}$')
plt.ylabel('probability density')
plt.gca().set_yticklabels([])
plt.legend()
plt.show()

msk = F_s!=0
out = np.array([E_s[msk], F_s[msk]]).transpose()
np.savetxt(f'modified_spectrum.txt', out, header='E F')
out = dc(ids_c_s)
N = (out[:,0]!=-1).sum()
out = out[:N, :]
cmax = (out!=-1).sum(axis=1).max()
out = out[:, :cmax]
np.savetxt(f'ids_c.txt', out)
