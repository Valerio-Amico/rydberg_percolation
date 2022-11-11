import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, ArtistAnimation
from rydbperc import ryperc

N_points = 5000
radius_MOT = 40
shape = [80,80,100]
density = N_points/(shape[0]*shape[1]*shape[2]*10**-2) # atoms/cm^3

C6 = 1.2*10**6 # van der waals coefficient [MHz * um^6]
gamma_def = 4 # dephasing rate [MHz]
Delta = 10 # detuning [MHz]
gamma_spontaneous_emission_70s = 0.01 # [MHz]

radius_shell = (C6/(Delta))**(1/6)
delta_radius_shell = radius_shell*(gamma_def/(6*Delta))

dt = 0.1 # [us]
evo_time = 100 # [us]
N_steps = int(evo_time/dt)

p_spont_exc = 2/(N_points) * dt/evo_time
p_spont_emi = gamma_spontaneous_emission_70s * dt
p_facilitation = 0.1

print("density = ",density)
print("fac. shell radius and delta = ", radius_shell, delta_radius_shell)


clu = ryperc.cluster3D(N_points, shape, distribution="gaussian", MOT_radius=radius_MOT)

clu.set_evolution_parameters(
                            shell_radius=radius_shell, 
                            shell_delta=delta_radius_shell, 
                            p_spont_exct=p_spont_exc, 
                            p_emission=p_spont_emi, 
                            p_fac=p_facilitation
                            )

clu.evolve(steps=N_steps)
print(len(clu.cluster_excited))
clu.show()



