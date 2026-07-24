import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from tqdm import tqdm


ls = natsorted(glob("dump/m_*.txt"))
df = np.loadtxt('es.txt', skiprows=1)
e = df[:,1]-df[:,0]


N = len(ls)
n0 = int(round(N*0.8))
m0 = np.loadtxt(ls[-1], skiprows=1)
x = np.zeros(m0.shape[1])
cnt = 0

for file in tqdm(ls[n0::10]):
    m = np.loadtxt(file, skiprows=1)
    xi = m.mean(axis=0)
    x += xi
    cnt += 1

x = x/cnt

srt = np.argsort(x)[::-1]
srt = np.argsort(e)
xs = x[srt]
es = e[srt]

kT = 3
mu = -10
x_fermi = 1/(1+np.exp((es-mu)/kT))
print('mean x_fermi: ', x_fermi.mean())
print('mean x_mc: ', x.mean())

plt.figure(dpi=300)
plt.plot(xs, color='black', label='MC')
plt.plot(x_fermi, color='red', label='fermi')
plt.legend()
plt.savefig("plot_x.png")

