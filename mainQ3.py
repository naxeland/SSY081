import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

f = np.load('data_files/f.npy', allow_pickle=True)
### Q3 
# Added interference with fft
fft_values = fft.rfft(f[0])#/x.size
interf = fft_values.copy()
interf[50] += 0.15 * 1024 / 2
# a)
plt.title("Q3 a) Signal comparison")
plt.plot(fft.irfft(interf), 'r', linewidth=0.5, label = 'With interference')
plt.plot(f.T, 'b', linewidth=0.5, label = 'Without interference')
plt.xlabel("Time(s)")
plt.ylabel("A.U")
plt.legend(ncol=4)

plt.figure()
plt.title("Q3 c) Absolute value of DFT")
plt.plot(np.abs(interf)    , 'r' , linewidth=0.5, label = 'With interference')
plt.plot(np.abs(fft_values), 'b' , linewidth=0.5, label = 'Without interference')
plt.xlabel("Hz")
plt.ylabel("A.U")
plt.legend(ncol=4)
plt.show()
