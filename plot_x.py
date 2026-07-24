import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from tqdm import tqdm

modified = True
ls = natsorted(glob("dump/x_*.txt"))

if not modified:
    df = np.loadtxt('es.txt', skiprows=1)
    e = df[:,1]-df[:,0]
else:
    df = np.loadtxt('modified_spectrum.txt', skiprows=1)
    e = np.repeat(df[:, 0], df[:,1].astype(int))

N = len(ls)
n0 = int(round(N*0.8))
m0 = np.loadtxt(ls[-1], skiprows=1)
x = np.zeros(m0.shape[0])
cnt = 0

for file in tqdm(ls[n0::]):
    xi = np.loadtxt(file, skiprows=1)
    x += xi
    cnt += 1

x = x/cnt

srt = np.argsort(e)
xs = x[srt]
es = e[srt]

kT = 3
mu = -10
x_fermi = 1/(1+np.exp((es-mu)/kT))
print('mean x_fermi: ', x_fermi.mean())
print('mean x_mc   : ', x.mean())

plt.figure(dpi=300)
plt.plot(es, xs, color='black', label='MC')
plt.plot(es, x_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.savefig("plot_x.png")

