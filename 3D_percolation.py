import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, ArtistAnimation
from rydbperc import ryperc

N_points = 1000
shape = [1,1,10]
clu = ryperc.cluster3D(N_points, shape)
clu.show()