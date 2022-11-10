import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import truncnorm
from rydbperc import utilities

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
        self.cluster_excited = []
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

    def get_points_connections(self, point_indeces):
        """ 
        retruns set of points (indeces) in the facilitation shell of the point with index "point_index".
        the facilitation shell is the shell with R in [R-dR/2, R+dR/2]. 
        """
        big_ball = set(KDTree.query_ball_point(KDTree.data[point_indeces],self.R_shell+self.dR_shell/2))
        small_ball = set(KDTree.query_ball_point(KDTree.data[point_indeces],self.R_shell-self.dR_shell/2))
        return big_ball.difference(small_ball)
    
    def set_evolution_parameters(self, shell_radius, shell_delta, p_spont_exct, p_fac, p_emission):
        """ 
        args:
            shell_radius (float): radius of the facilitation shell
            shell_delta (float): width of the facilitation shell
                                 the facilitation shell is the shell with R in [R-dR/2, R+dR/2]
            p_spont_exct (float in [0,1]): probability of spontaneous excitations in one step
            p_fac (float in [0,1]): probability of facilitated excitation in one step, if 
                                    facilitation condition is respected (i.e. the point is in the facilitiation shell of another point)
            p_emission (float in [0,1]): probability of spontaneous emission in one step
        """
        self.R_shell = shell_radius
        self.dR_shell = shell_delta
        self.p_spont_exct = p_spont_exct
        self.p_fac = p_fac
        self.p_emission = p_emission
        return

    def evolve(self, steps):
        """ 
        evolves the system
        args:
            steps (int): evolution steps
        """
        for step in range(steps):
            ####### spontaneus excitation ###################
                # the number of spontaneous excitation is exctracted from a poissonian, because
                # the single excitations are indipendent events and since p<<1 and N_atoms>>N_spontaneous_exct
                # the binomial distribution tends to a poissonian.
            N_spontaneous_exct = np.random.poisson(self.p_emission*self.size)
            self.cluster_excited = list(set(self.cluster_excited + list(np.random.choice(np.range(self.size), N_spontaneous_exct))))
            ####### END spontaneous excitation ##############
            ####### facilitation excitation #################
                # first it computes the possible points witch can be excited by facilitation,
                # than from a binomial distribution are extracted the number of them will be excited
            facilitable_points = self.get_points_connections(self.cluster_excited)
            N_fac_exct = np.random.binomial(len(facilitable_points),self.p_fac)
                # one by one the points are extracted from the facilitable_points,
                # to be sure that the facilitation constraint in respected,
                # if not less point will be excited.
            for _ in range(N_fac_exct):
                if 
                self.cluster_excited = list(set(self.cluster_excited + list(np.random.choice(facilitable_points, 1))))
            ####### END facilitation excitation #############
            ####### spontaneous emission ####################
            N_spontaneous_emission = np.random.binomial(len(self.cluster_excited), self.p_emission)
            list(np.random.choice(np.range(self.size), N_spontaneous_emission))
            self.cluster_excited = list(set(self.cluster_excited + )))
            ####### END spontaneous emission ################
            
        return 

