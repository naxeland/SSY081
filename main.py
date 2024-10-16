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

plt.plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)
plt.title("Q1 d) Action potential train 3")
plt.xlabel("Time (s)")
plt.ylabel("A.U")

plt.figure()
plt.plot(np.linspace(0, 20, 200000, dtype=float), action_potential_trains[2], linewidth=0.5)
plt.title("Q1 d) Action potential train 3 (10-10.5s)")
plt.xlabel("Time (s)")
plt.ylabel("A.U")
plt.xlim(10, 10.5)

plt.figure()
plt.plot(np.linspace(0, 20, 200000, dtype=float), list(map(sum, action_potential_trains.T)))
plt.xlim(10, 10.5)
plt.title("Q1 f) Sum of all action potential trains (10-10.5)")
plt.xlabel("Time (s)")
plt.ylabel("A.U")



### Q2
# a)
hanning_window = np.hanning(10000)
filtered_action_trains = [np.convolve(hanning_window, action_train) for action_train in action_trains]



# c)
plt.figure()
for i, f in enumerate(filtered_action_trains):
    plt.plot(np.linspace(-0.5, 20.5, 200000 + 10000 - 1, dtype=float), f, label=f'{i+1}')

plt.title("Q2 c) Filtered signals of all action trains")
plt.xlabel("Time (s)")
plt.ylabel("A.U")
plt.legend(ncol=4)

# d)
plt.figure()
plt.title("Q2 d) Unit 4: Action train and filtered action train")
plt.xlabel("Time (s)")
plt.ylabel("A.U")
plt.plot(np.linspace(0, 20, 200000), action_trains[3], linewidth='0.5')
plt.plot(np.linspace(-0.5, 20.5, 209999), filtered_action_trains[3])


# e)
plt.figure()
plt.title("Q2 e) Unit 7: Action train and filtered action train")
plt.xlabel("Time (s)")
plt.ylabel("A.U")
plt.plot(np.linspace(0, 20, 200000), action_trains[6], linewidth='0.5')
plt.plot(np.linspace(-0.5, 20.5, 209999), filtered_action_trains[6])

plt.show()


