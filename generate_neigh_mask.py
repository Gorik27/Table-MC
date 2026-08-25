import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


N = 10000 # number of site types
Z = 11 # average number of neighbors (and mode of distributuion)

def generate(Z, N):
    mode = 0
    while mode!=Z:
        m = np.zeros((N, N))
        Z_max = int(np.ceil(Z/2)*2)*2
        k = Z_max//2
        
        for i in range(N):
            for j in range(i+1, min(i+1+k, N)):
                s = np.random.choice([1, 0], p=[Z/Z_max, 1-Z/Z_max])
                m[i,j] = s
                m[j,i] = s
            if i<k:
                for j in range(N-k+i, N):
                    s = np.random.choice([1, 0], p=[Z/Z_max, 1-Z/Z_max])
                    m[i,j] = s
                    m[j,i] = s
                
        
        st = m.sum(axis=0)
        mode = stats.mode(st, keepdims=True).mode[0]
    return m

m = generate(Z, N)
np.savetxt("N_mask.txt", m, header=f'{N} {N}', comments='', fmt='%d')
st = m.sum(axis=0)
mode = stats.mode(st, keepdims=True).mode[0]

""" plt.subplot(121)
plt.imshow(m)
plt.subplot(122)
plt.hist(st)
plt.title(f'mode {mode}')
plt.gcf().tight_layout()
plt.show() """