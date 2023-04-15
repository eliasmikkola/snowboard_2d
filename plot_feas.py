# This is a sample Python script.
import numpy as np
import time
# import plot
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sys import getsizeof
import seaborn as sns

steepness_lower_bound = 0.1
steepness_upper_bound = 0.6
amplitude_lower_bound = 0.0
amplitude_upper_bound = 4.0
frequency_lower_bound = 1
frequency_upper_bound = 10

param_space_dim = [10j,10j,10j]

dim_stack = np.stack(np.mgrid[
steepness_lower_bound:steepness_upper_bound:param_space_dim[0],
amplitude_lower_bound:amplitude_upper_bound:param_space_dim[1],
frequency_lower_bound:frequency_upper_bound:param_space_dim[2]]).T.reshape(-1, 3)
print("STACK SHAPE", dim_stack.shape[0])
feasibility_truth = np.load("feasibility_arr.npy")
print("FEAS TRUTH", feasibility_truth)

print("DIM STACK", dim_stack)


# add feasibility truth to end of each dim stack row
dim_with_feas = np.hstack((dim_stack, feasibility_truth.reshape(-1,1)))
print(getsizeof(dim_with_feas))
# print in dim_with_feas formated sa "--S={0} --A={1} --F={2} {3}
print("---FEASIBILITY---")
# for i in range(len(dim_with_feas)):
# # round to 2
#     dim_steep, dim_amp, dim_freq, feas = dim_with_feas[i]
#     print(f"--S={round(dim_steep,2)} --A={round(dim_amp,2)} --F={round(dim_freq,2)} {feas}")

# plot dim_with_feas in 3d
fig = plt.figure()
# increse figure size
fig.set_size_inches(10.5, 7.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dim_with_feas[:,0], dim_with_feas[:,1], dim_with_feas[:,2], c=dim_with_feas[:,3], cmap='winter', linewidth=0.5, alpha=0.5, edgecolors='black')

# 3d solid plot
ax = fig.add_subplot(111, projection='3d')
# make the colors red and green
ax.scatter(dim_with_feas[:,0], dim_with_feas[:,1], dim_with_feas[:,2], c=dim_with_feas[:,3], linewidth=0.5, alpha=0.3, s=300, cmap='RdYlGn', marker='s', edgecolors='black')
# make the plot bigger
plt.rcParams['figure.figsize'] = [10, 7]

ax.set_xlabel('Steepness')
ax.set_ylabel('Amplitude')
ax.set_zlabel('Frequency')
plt.show()