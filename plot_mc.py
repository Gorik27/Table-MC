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
acc = df[n0:, 1]
N0 = df[n0:, 2]
N1 = df[n0:, 3]
e = df[n0:, 4]


plt.figure(dpi=300)

plt.subplot(221)
plt.plot(t, acc)
plt.xlabel('MC steps')
plt.title('acceptance')

plt.subplot(222)
x = N1/(N0+N1)
plt.plot(t, x)
plt.xlabel('MC steps')
plt.title('concentration')

plt.subplot(223)
plt.plot(t, e)
plt.xlabel('MC steps')
plt.title('energy')

plt.gcf().tight_layout()
plt.savefig('plot_mc.png')

print("final conc:", x[-1])
print("final energy:", e[-1])
print("final energy per solute atom:", e[-1]/x[-1])