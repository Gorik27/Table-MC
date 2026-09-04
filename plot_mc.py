import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", '--offset', type=float, default=0, help='fraction of window to offset')
args = parser.parse_args()
of = args.offset 

df = np.loadtxt("mc_output.txt", skiprows=2)
t = df[:, 0]
nsteps = len(t)
if of>0 and of<1:
    n0 = int(round(of*nsteps))
    t = t[n0:]
else:
    n0 = 0

n_types = df.shape[1]-3
acc = df[n0:, 1]
N0 = df[n0:, 2]
Ns = np.zeros((N0.shape[0], n_types-1))
for k in range(n_types-1):
    Ns[:, k] = df[n0:, 3+k]
e = df[n0:, df.shape[1]-1]


plt.figure(dpi=300)

plt.subplot(221)
plt.plot(t, acc)
plt.xlabel('MC steps')
plt.title('acceptance')

plt.subplot(222)
xs = np.einsum("ij,i->ij", Ns, 1/(N0+np.sum(Ns, axis=1)))
for k in range(n_types-1):
    plt.plot(t, xs[:,k], label=f'{k}')
plt.legend()
plt.xlabel('MC steps')
plt.title('concentration')

plt.subplot(223)
plt.plot(t, e)
plt.xlabel('MC steps')
plt.title('energy')

plt.gcf().tight_layout()
plt.savefig('plot_mc.png')

for k in range(n_types-1):
    print(f"final conc {k}:", xs[-1, k])
print("final energy:", e[-1])