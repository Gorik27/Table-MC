import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from tqdm import tqdm


rows = 10

df = np.loadtxt('es.txt', skiprows=1)
e0 = df[:,1]-df[:,0]

df = np.loadtxt('modified_spectrum.txt', skiprows=1)
em = np.repeat(df[:, 0], df[:,1].astype(int))

w_table = np.loadtxt('w11.txt', skiprows=1)
w = w_table.sum(axis=0)

ls = natsorted(glob("dump/x_*.txt"))
N = len(ls)
if N>1:
    n0 = int(round(N*0.8))
else:
    n0 = 0
m0 = np.loadtxt(ls[-1], skiprows=1)
x = np.zeros(m0.shape[0])
cnt = 0

for file in tqdm(ls[n0::10]):
    xi = np.loadtxt(file, skiprows=1)[:, 1]
    x += xi
    cnt += 1

x = x/cnt

srt = np.argsort(x)[::-1]
xmc = x[srt]

srt0 = np.argsort(e0)
xs0 = x[srt0]
es0 = e0[srt0]

srtm = np.argsort(em)
xsm = x[srtm]
esm = em[srtm]




kT = 3
mu = -60
x0_fermi = 1/(1+np.exp((es0-mu)/kT))

xgb = x0_fermi.mean()
for _ in range(1000):
    xr_fermi = 1/(1+np.exp((e0+w*xgb/2-mu)/kT))
    eps = xr_fermi.mean()-xgb
    xgb = xr_fermi.mean()
    if abs(eps) > 0.00001:
        break
print(eps)

er = e0 + w*xgb/2
srtr = np.argsort(er)
xsr = x[srtr]
esr = er[srtr]
xr_fermi = 1/(1+np.exp((esr-mu)/kT))

xm_fermi = 1/(1+np.exp((esm-mu)/kT))

print('mean x_fermi: ', x0_fermi.mean())
print('mean x_r_fermi: ', xr_fermi.mean())
print('mean x_mod  : ', xm_fermi.mean())
print('mean x_mc   : ', x.mean())

plt.figure(dpi=300)
plt.subplot(131)
plt.plot(es0, xmc, color='black', label='MC')
plt.plot(es0, x0_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('original spectrum')

plt.subplot(132)
plt.plot(esr, xmc, color='black', label='MC')
plt.plot(esr, xr_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('NCRS')

plt.subplot(133)
plt.plot(esm, xmc, color='black', label='MC')
plt.plot(esm, xm_fermi, color='red', label='fermi')


plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('modified spectrum')



plt.gcf().tight_layout()
plt.savefig("plot_x.png")

print("\noriginal energy:", np.sum(es0*x0_fermi)*rows)
print("NCRS energy:", np.sum(esr*xr_fermi)*rows)
print("mod energy:", np.sum(esm*xm_fermi)*rows)

print("mod energy per solute atom:", np.sum(esm*xm_fermi)*rows/xm_fermi.mean())