import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, ArtistAnimation
from rydbperc import ryperc

N_points = 2000
radius_MOT = 40
shape = [80,80,100]
density = N_points/(shape[0]*shape[1]*shape[2]*10**-2) # atoms/cm^3

C6 = 1.2*10**6 # van der waals coefficient [MHz * um^6]
gamma_def = 4 # dephasing rate [MHz]
Delta = 10 # detuning [MHz]
gamma_spontaneous_emission_70s = 0.5 # [MHz]

radius_shell = (C6/(Delta))**(1/6)
delta_radius_shell = radius_shell*(gamma_def/(6*Delta))

dt = 0.1 # [us]
evo_time = 100 # [us]
N_steps = int(evo_time/dt)

p_spont_exc = 2/(N_points) * dt/evo_time
p_spont_emi = gamma_spontaneous_emission_70s * dt
p_facilitation_range = np.linspace(0, 0.06, 10)

print("density = ",density)
print("fac. shell radius and delta = ", radius_shell, delta_radius_shell)

just_repetitions = 10
N_exct_means = []
N_exct_stds = []

for i, p_facilitation in enumerate(p_facilitation_range):
    reps = []
    for _ in range(just_repetitions):
        clu = ryperc.cluster3D(N_points, shape, distribution="gaussian", MOT_radius=radius_MOT, is_2D=True)

        clu.set_evolution_parameters(
                                    shell_radius=radius_shell, 
                                    shell_delta=delta_radius_shell, 
                                    p_spont_exct=p_spont_exc, 
                                    p_emission=p_spont_emi, 
                                    p_fac=p_facilitation
                                    )

        clu.evolve(steps=N_steps, excitation_steps=int(N_steps/10))
        reps.append(len(clu.cluster_excited))
        
    print("status: %d %%"%((i+1)/len(p_facilitation_range)*100))
    N_exct_means.append(np.mean(reps))
    N_exct_stds.append(np.std(reps))

plt.errorbar(p_facilitation_range, N_exct_means, N_exct_stds, linestyle="", marker=".")
plt.show()

clu.show()
