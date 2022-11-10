import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, ArtistAnimation
from rydbperc import ryperc

N_points = 1000
radius_MOT = 10
shape = [80,80,80]
density = N_points/(shape[0]*shape[1]*shape[2]*10**-2) # atoms/cm^3
print("density = ",density)
clu = ryperc.cluster3D(N_points, shape, distribution="gaussian", MOT_radius=radius_MOT)

clu.show()