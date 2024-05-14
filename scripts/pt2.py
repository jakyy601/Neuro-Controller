import matplotlib.pyplot as plt

yn = []
yn2 = []
t = []
T = 0.1
K = 1.0
u = 1

yn.append(0)
yn2.append(0)

for i in range(100):
    yn2p1 = yn[i] + K * u * T
    yn2.append(yn2p1)
    ynp1 = yn[i] + K * yn2[i+1] * T
    yn.append(ynp1)
    t.append(i)

del yn2[-1]
plt.plot(t, yn2)
plt.grid()
plt.show()
