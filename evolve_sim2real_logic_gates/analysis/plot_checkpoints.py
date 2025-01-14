import pickle
import os 
import sys
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath('')))
from afpo import AFPO

seeds = 10
gen = 100

gate = "XOR"
encoding = "stiffness"

file = open(f"results.csv", "a")
# file.write("seed,gate,encoding,stiffness,size,fitness_gen_100\n")


for stiffness in [True]:
    for size in [False,True]:
        if stiffness and size:
            label = 'size change, stiffness change' 
            color = 'red'
        elif size: label = 'size change';color='blue'
        else: label = 'stiffness change';color='orange'

        for seed in range(seeds):
            checkpoint_file = f'../checkpoints/{encoding}_encoding/{gate}/run{seed}_gen{gen}_change_stiffness_{stiffness}_change_size_{size}_gate{gate}_encoding{encoding}.p'
            with open(checkpoint_file, 'rb') as f:
                afpo, rng_state, np_rng_state = pickle.load(f)
            best_mat = afpo.return_best()
            if seed==0:
                plt.plot(np.max(afpo.fitness_data[:gen,:, 0], axis=1), color=color, label=label)
            plt.plot(np.max(afpo.fitness_data[:gen,:, 0], axis=1), color=color)
            print(np.max(afpo.fitness_data[gen,:, 0]))
            # quit()
            file.write(f"{seed},{gate},{encoding},{stiffness},{size},{np.max(afpo.fitness_data[:gen,:, 0])}")
            file.write("\n")
plt.ylim([0,20])
plt.xlabel('Generation')
plt.ylabel(f'{gate}ness')
plt.grid()
plt.legend()
plt.savefig(f"{gate}_encoding_{encoding}")