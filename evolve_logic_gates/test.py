import sys
import numpy as np
import matplotlib.pyplot as plt

from MD_functions import FIRE_VL, FIRE_FixedTopForce_VL, ForceWall
from PlotFunctions import ConfigPlot_DiffSize

# inflate_id = int(sys.argv[1])
# inflate_d = float(sys.argv[2])
# bc_type = int(sys.argv[3])

# inflate_id = 1
# inflate_d = 0.2
# material = self.children[material_index]
n_col = 4
n_row = 3
N = (n_col - 1) * n_row + int(np.floor(n_row / 2.0)) # total number of particles

d0 = 1.
y_top_disp = 0.01 ## compression amount
Lx = d0 * n_col
Ly = (n_row - 1) * np.sqrt(3) / 2 * d0 + d0

D = np.ones(N, dtype = np.float64) * d0


x = np.zeros(N, dtype = np.float64)
y = np.zeros(N, dtype = np.float64)

ind = -1
for i_row in range(n_row):
    if i_row % 2 == 1:
        n_col_now = n_col
    else:
        n_col_now = n_col - 1
    for i_col in range(n_col_now):
        ind += 1
        if i_row % 2 == 1:
            x[ind] = (i_col + 0.5) * d0
        else:
            x[ind] = (i_col + 1.) * d0
        y[ind] = i_row * 0.5 * np.sqrt(3) * d0
y = y + 0.5 * d0


mass = np.ones(N, dtype = np.float64)
k_list = np.ones(N, dtype = np.float64) * 1.
hard_grains = [1,4,5,7,8,9]
k_list[hard_grains] = 10
# for particle_index, particle in enumerate(material.particles): #fill in genome 
#     D[particle_index] = d0 * (1. + particle.expansion) # inflate this specific particle
#     k_list[particle_index] = particle.stiffness 

Ly -= y_top_disp * d0
Lx -= y_top_disp * d0

FIRE_VL(N, x, y, D, Lx, Ly, k_list)


# Fx_w, Fy_w = ForceWall(N, x, y, D, Lx, Ly, k_list)
# return material_index, Fy_w


Fx_w, Fy_w = ForceWall(N, x, y, D, Lx, Ly, k_list)
print(Fy_w)
# print(Fy_w)

ConfigPlot_DiffSize(N, x, y, D, Lx, Ly, k_list, 1, 0)
plt.title(f'{Fy_w[2]}')
plt.savefig('configuration_v')
