import numpy as np
import matplotlib.pyplot as plt

action_potentials = np.load('data_files/action_potentials.npy', allow_pickle=True)
firing_samples = np.load('data_files/firing_samples.npy', allow_pickle=True).T


action_trains = np.zeros((8, 200000), dtype=int)
action_potential_trains = np.zeros((8, 200000), dtype=float)
sample_counts = np.zeros((8), dtype=int)

for i in range(len(firing_samples)):
    for sample in firing_samples[i]:
        action_trains[i][sample] = 1
        sample_counts[i] += 1

        for j in range(len(action_potentials[i])):
            action_potential_trains[i][sample + j] = action_potentials[i][j]


for k in range(8):
    print(sum(action_trains[k]))

#Prints för att kolla data:
   
#print(np.size(firing_samples[0]))    
#print(sum(action_trains[0]))
#print(firing_samples)
#print(action_potentials)
#print(firing_samples.shape)

#Plottningar
plt.plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)

plt.show()


plt.plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)
plt.xlim(10, 10.5)
plt.show()

plt.plot(np.linspace(0, 20, 200000, dtype=float), list(map(sum, action_potential_trains.T)))
plt.xlim(10, 10.5)
plt.show()

