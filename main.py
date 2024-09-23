import numpy as np
import matplotlib.pyplot as plt

action_potentials = np.load('data_files/action_potentials.npy', allow_pickle=True)
firing_samples = np.load('data_files/firing_samples.npy', allow_pickle=True)


action_trains = np.zeros((8, 200000), dtype=int)
action_potential_trains = np.zeros((8, 200000), dtype=float)

for i in range(len(firing_samples)):
    for sample in firing_samples[i]:
        action_trains[i][sample] = 1

        for j in range(len(action_potentials[i])):
            action_potential_trains[i + j] = action_potentials[j]


plt.plot(range(200000), action_potential_trains)
plt.show()