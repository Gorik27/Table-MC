#!/home/user/miniconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", '--offset', type=float, default=0, help='fraction of window to offset')
args = parser.parse_args()
of = args.offset 

fname = "mc_gg_output.txt"
df = np.loadtxt(fname, skiprows=2)

elements = []
with open(fname) as f:
    line = f.readlines()[1]
    for item in line.split(" "):
        if "Xgb_" in item:
            elements.append(item.replace("Xgb_", ""))

t = df[:, 0]
nsteps = len(t)
if of>0 and of<1:
    n0 = int(round(of*nsteps))
    t = t[n0:]
else:
    n0 = 0

n_types = (df.shape[1]-4)//2
acc1 = df[n0:, 1]
acc2 = df[n0:, 2]
fgbs = df[n0:, 3]
Ngb0 = df[n0:, 4]
Ngbs = np.zeros((Ngb0.shape[0], n_types-1))
for k in range(n_types-1):
    Ngbs[:, k] = df[n0:, 5+k]
Nc0 = df[n0:, 5+n_types]
Ncs = np.zeros((Nc0.shape[0], n_types-1))
for k in range(n_types-1):
    Ncs[:, k] = df[n0:, 5+n_types+k]
e = df[n0:, df.shape[1]-1]


plt.figure(dpi=300)

plt.subplot(231)
plt.plot(t, acc1, label='step 1')
plt.plot(t, acc2, label='step 2')
plt.legend()
plt.xlabel('MC steps')
plt.title('acceptance')

plt.subplot(232)
xgbs = np.einsum("ij,i->ij", Ngbs, 1/(Ngb0+np.sum(Ngbs, axis=1)))
for k in range(n_types-1):
    plt.plot(t, xgbs[:,k], label=f'{elements[k+1]}')
plt.legend()
plt.xlabel('MC steps')
plt.title('GB concentration')

plt.subplot(233)
xcs = np.einsum("ij,i->ij", Ncs, 1/(Nc0+np.sum(Ncs, axis=1)))
for k in range(n_types-1):
    plt.plot(t, xcs[:,k], label=f'{elements[k+1]}')
plt.legend()
plt.xlabel('MC steps')
plt.title('C concentration')

plt.subplot(236)
plt.plot(t, e)
plt.xlabel('MC steps')
plt.title('energy')

plt.subplot(235)
plt.plot(t, fgbs)
plt.xlabel('MC steps')
plt.title('GB fraction')

plt.gcf().tight_layout()
plt.savefig('plot_mc_gg.png')

for k in range(n_types-1):
    print(f"final GB conc {elements[k+1]}:", xgbs[-1, k])
print("")
for k in range(n_types-1):
    print(f"final C conc {elements[k+1]}:", xcs[-1, k])
print("")
print("final fgb:", fgbs[-1])
print("final energy:", e[-1])

for k in range(n_types-1):
    print(f"mean GB conc {elements[k+1]}:", xgbs[:, k].mean())
print("")
for k in range(n_types-1):
    print(f"mean C conc {elements[k+1]}:", xcs[:, k].mean())
print("")
print("mean fgb:", fgbs.mean())
print("mean energy:", e.mean())