#Imports
import math
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


#for k in range(8):
#    print(sum(action_trains[k]))

#Prints för att kolla data:
   
#print(np.size(firing_samples[0]))    
#print(sum(action_trains[0]))
#print(firing_samples)
#print(action_potentials)
#print(firing_samples.shape)

#Plottningar

figure, axis = plt.subplots(2, 2)
figure.suptitle("Q1")

axis[0, 0].set_title("d) Action potential train 3")
axis[0, 0].set_xlabel("Time (s)")
axis[0, 0].set_ylabel("A.U")
axis[0, 0].plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)

axis[0, 1].set_title("d) Action potential train 3 (10-10.5s)")
axis[0, 1].set_xlabel("Time (s)")
axis[0, 1].set_ylabel("A.U")
axis[0, 1].plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)
axis[0, 1].set_xlim(10, 10.5)

axis[1, 0].set_title("f) Sum of all action potential trains (10-10.5)")
axis[1, 0].set_xlabel("Time (s)")
axis[1, 0].set_ylabel("A.U")
axis[1, 0].plot(np.linspace(0, 20, 200000, dtype=float), list(map(sum, action_potential_trains.T)))
axis[1, 0].set_xlim(10, 10.5)

figure.tight_layout()


### Q2
# a)
hanning_window = np.hanning(10000)
filtered_action_trains = [np.convolve(hanning_window, action_train) for action_train in action_trains]

figure, axis = plt.subplots(2, 2)
figure.suptitle("Q2")

# c)
for i, f in enumerate(filtered_action_trains):
    axis[0, 0].plot(np.linspace(-0.5, 20.5, 200000 + 10000 - 1, dtype=float), f, label=f'{i+1}')

axis[0, 0].set_title("c) Filtered signals of all action trains")
axis[0, 0].set_xlabel("Time (s)")
axis[0, 0].set_ylabel("A.U")
axis[0, 0].legend(ncol=4)

# d)
axis[1, 0].set_title("d) Unit 4: Action train and filtered action train")
axis[1, 0].set_xlabel("Time (s)")
axis[1, 0].set_ylabel("A.U")
axis[1, 0].plot(np.linspace(0, 20, 200000), action_trains[3], linewidth='0.5')
axis[1, 0].plot(np.linspace(-0.5, 20.5, 209999), filtered_action_trains[3])


# e)
axis[1, 1].set_title("e) Unit 7: Action train and filtered action train")
axis[1, 1].set_xlabel("Time (s)")
axis[1, 1].set_ylabel("A.U")
axis[1, 1].plot(np.linspace(0, 20, 200000), action_trains[6], linewidth='0.5')
axis[1, 1].plot(np.linspace(-0.5, 20.5, 209999), filtered_action_trains[6])
axis[1, 1].set_ylim(axis[1, 0].get_ylim())

figure.tight_layout()
plt.show()


