import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from tqdm import tqdm


kT = 3
mu_mc = 40


mu = -mu_mc # in MC codes chemical potential is often inverted to classical definition (mu = dG/dX)
"""
READING
"""
df = np.loadtxt('new_es.txt', skiprows=1)
e0 = df[:,1]

df = np.loadtxt('es_mod.txt')
ids_m = df[:, 0]
em = df[:, 1]
srt = np.argsort(ids_m)
em = em[srt]
ids_m = ids_m[srt]
assert np.all(ids_m.astype(int)==np.arange(len(ids_m))+1)

w = np.zeros(len(e0))
with open('new_eint.txt') as f:
    for i, line in enumerate(f.readlines()[1:]):
        args = line.replace(' \n', '').replace('\n', '').split(' ')
        ws = list(map(float, args[1:]))
        w[i] = np.sum(ws)


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

"""
ORDERING (SORT)
"""

srt = np.argsort(x)[::-1]
xmc = x#[srt]

srt0 = np.argsort(e0)
xs0 = x[srt0]
es0 = e0[srt0]

srtm = np.argsort(em)
xsm = x[srtm]
esm = em[srtm]

"""
CALCULATIONS
"""
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

"""
OUTPUT
"""

print('mean x_fermi: ', x0_fermi.mean())
print('mean x_r_fermi: ', xr_fermi.mean())
print('mean x_mod  : ', xm_fermi.mean())
print('mean x_mc   : ', x.mean())

plt.figure(dpi=300)
cs = np.linspace(0,1)

plt.subplot(231)
plt.plot(es0, xmc[srt0], color='black', label='MC')
plt.plot(es0, x0_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('original spectrum')

plt.subplot(234)
plt.plot(cs, cs, linestyle='--', color='grey')
plt.scatter(xmc[srt0], x0_fermi)
plt.xlabel('x MC')
plt.ylabel('x orig')
plt.gca().set_box_aspect(1)

plt.subplot(232)
plt.plot(esr, xmc[srtr], color='black', label='MC')
plt.plot(esr, xr_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('NCRS')

plt.subplot(235)
plt.plot(cs, cs, linestyle='--', color='grey')
plt.scatter(xmc[srtr], xr_fermi)
plt.xlabel('x MC')
plt.ylabel('x NCRS')
plt.gca().set_box_aspect(1)

plt.subplot(233)
plt.plot(esm, xmc[srtm], color='black', label='MC')
plt.plot(esm, xm_fermi, color='red', label='fermi')
plt.xlabel('energy')
plt.ylabel('X')
plt.legend()
plt.gca().set_box_aspect(1)
plt.title('modified spectrum')

plt.subplot(236)
plt.plot(cs, cs, linestyle='--', color='grey')
plt.scatter(xmc[srtm], xm_fermi)
plt.xlabel('x MC')
plt.ylabel('x mod')
plt.gca().set_box_aspect(1)

plt.gcf().tight_layout()
plt.savefig("plot_x.png")

print("\noriginal energy:", np.mean(es0*x0_fermi))
print("NCRS energy:", np.mean(esr*xr_fermi))
print("mod energy:", np.mean(esm*xm_fermi))

print("\noriginal energy per solute atom:", np.mean(es0*x0_fermi)/x0_fermi.mean())
print("NCRS energy per solute atom:", np.mean(esr*xr_fermi)/xr_fermi.mean())
print("mod energy per solute atom:", np.mean(esm*xm_fermi)/xm_fermi.mean())