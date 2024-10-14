import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

f = np.load('data_files/f.npy', allow_pickle=True)

figure, axis = plt.subplots(2)
### Q3 
# a)
axis[0].plot(f.T, 'b', linewidth=0.5, label = 'Without interference')
axis[0].set_xlabel("Time(s)")
axis[0].set_ylabel("A.U")

# Added interference manually
#f50 = [value + 0.15 * math.cos(50.0 * 2.0 * math.pi * i / 1024.0) for i, value in enumerate(f.T)]
#axis[1, 0].plot(f50)
#axis[1, 0].set_xlabel("Time(s)")
#axis[1, 0].set_ylabel("A.U")



# Added interference with fft
fft_values = fft.rfft(f[0])#/x.size
#print(fft_values)
interf = fft_values.copy()
interf[50] += 0.15 * 1024 / 2

axis[1].plot(np.abs(interf)    , 'r' , linewidth=0.5, label = 'With interference')
axis[1].plot(np.abs(fft_values), 'b' , linewidth=0.5, label = 'Without interference')
axis[1].set_xlabel("Hz")
axis[1].set_ylabel("A.U")
axis[1].legend(ncol=4)



axis[0].plot(fft.irfft(interf), 'r', linewidth=0.5, label = 'With interference')
axis[0].legend(ncol=4)

plt.show()