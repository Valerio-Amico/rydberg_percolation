import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=1):
    """ 
    retruns a truncated normal distribution.
    args:
        mean (float): mean
        sd (float): standard deviation
        low (float): lower bound
        upp (float): upper bound
    returns:
        scipy.stats.truncnorm object
    """
    return truncnorm((low - mean)/sd, (upp - mean)/sd, loc=mean, scale=sd)

class cluster3D:
    def __init__(self, size, shape=[1,1,1], distribution="uniform", MOT_radius=None) -> None:
        """ 
        defines the cluster and creates a KDTree to optimize the NN seach. The interaction volume is a cube with width equals to "width".
        args:
            size (int): number of points (or atoms) in the cloud.
            distribution (string): "uniform" or "gaussian", is the sample distribution.
            shape ([float,float,float]): is the shape of the interaction volume the the 3 coordinates.
            MOT_radius (float): if distribution is "gaussian" the radius is the standard deviation of the distribution.
            
        """
        self.size = size
        self.shape = shape
        self.distribution = distribution
        if distribution == "gaussian": 
            if MOT_radius == None or MOT_radius <= 0:
                print("\"MOT_radius\" must be a positive float.")
                raise
            self.MOT_radius = MOT_radius
        self.KDT = self.get_KDT()
        self.R_shell = None
        self.dR_shell = None
        pass

    def get_KDT(self):
        """ 
        generates the point positions and creates the KDtree.
        """
        if self.distribution == "uniform":
            positions = np.array([
                                np.random.random(self.size)*self.shape[0]-self.shape[0]/2, 
                                np.random.random(self.size)*self.shape[1]-self.shape[1]/2, 
                                np.random.random(self.size)*self.shape[2]-self.shape[2]/2
                                ])
        elif self.distribution == "gaussian":
            positions = np.array([
                                truncated_normal(0, self.MOT_radius, -self.shape[0]/2, self.shape[0]/2).rvs(self.size), 
                                truncated_normal(0, self.MOT_radius, -self.shape[1]/2, self.shape[1]/2).rvs(self.size), 
                                truncated_normal(0, self.MOT_radius, -self.shape[2]/2, self.shape[2]/2).rvs(self.size)
                                ])
        else: 
            print("distribution must be \"uniform\" or \"gaussian\".")
            raise
        return KDTree(positions.T)

    def show(self):
        """
        shows the 3D cluster
        """
        plt.figure(figsize=(15,12))
        ax = plt.axes(projection ="3d")
        ax.scatter(self.KDT.data[:,0],self.KDT.data[:,1],self.KDT.data[:,2], marker=".", alpha=1, s=100)
        plt.show()

    def point_connections(self, point_index):
        """ 
        retruns set of points (indeces) in the facilitation shell of the point with index "point_index".
        the facilitation shell is the shell with R in [R-dR/2, R+dR/2]. 
        """
        big_ball = set(KDTree.query_ball_point(KDTree.data[point_index],self.R_shell+self.dR_shell/2))
        small_ball = set(KDTree.query_ball_point(KDTree.data[point_index],self.R_shell-self.dR_shell/2))
        return big_ball.difference(small_ball)