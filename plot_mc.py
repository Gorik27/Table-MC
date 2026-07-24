import numpy as np
import matplotlib.pyplot as plt


df = np.loadtxt("mc_output.txt", skiprows=2)
t = df[:, 0]
acc = df[:, 1]
N0 = df[:, 2]
N1 = df[:, 3]
e = df[:, 4]

plt.figure(dpi=300)

plt.subplot(221)
plt.plot(t, acc)
plt.xlabel('MC steps')
plt.title('acceptance')

plt.subplot(222)
plt.plot(t, N1/(N0+N1))
plt.xlabel('MC steps')
plt.title('concentration')

plt.subplot(223)
plt.plot(t, e)
plt.xlabel('MC steps')
plt.title('energy')

plt.gcf().tight_layout()
plt.savefig('plot_mc.png')
