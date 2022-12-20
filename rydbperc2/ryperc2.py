import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import truncnorm
#from rydbperc import utilities

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

def get_excitation_rate(omega_rabi, dephasing, detuning, potential_term):
    #  ATTENZIONE, DA VERIFICARE SE LA FREQUENZA DI RABI è QUESTA O DIVISO 2. 
    rate = ((omega_rabi/2)**2)*dephasing/((dephasing/2)**2 + (-detuning+potential_term)**2)
    return rate

class cluster3D:
    def __init__(self, size, shape=[1,1,1], distribution="uniform", MOT_radius=None, is_2D = False) -> None:
        """ 
        defines the cluster and creates a KDTree to optimize the NN seach. The interaction volume has shape "shape".
        args:
            size (int): number of points (or atoms) in the cloud.
            distribution (string): "uniform" or "gaussian", is the sample distribution.
            shape ([float,float,float]): is the shape of the interaction volume the the 3 coordinates.
            MOT_radius (float): if distribution is "gaussian" the radius is the standard deviation of the distribution.
            is_2D (bool): if True a 2D cluster is created (i.e. the z-axis in a np.zeros(size))
        
        all the energies, times and distances are in MHz, us and um.
        #################
        Memory organization
            KD-Tree is used to optimize the seach in the cluster.
            the data are stored in a matrix with shape [3,size], the first index refers to x,y,z.
            the excited atoms are stored in a list (cluster_excited).

            all the point are divided in:
             - internal_points: all atoms whose distance from a Rydberg atom is less than the blockade radius
             - cluster_excited: all the atoms in the rydberg state
             - external_points: all the atoms that are not in internal_points o cluster_excited
             - leaves: all the atoms that are in the facilitiation shell of some other atom, note: leaves are also internal_points.
        """
        self.size = size
        self.shape = shape
        self.distribution = distribution
        self.is_2D = is_2D
        if distribution == "gaussian": 
            if MOT_radius == None or MOT_radius <= 0:
                print("\"MOT_radius\" must be a positive float.")
                raise
            self.MOT_radius = MOT_radius
        self.KDT = self.get_KDT()
        self.cluster_excited = set([])
        self.internal_points = set([])
        self.external_points = set(np.arange(self.size))
        pass

    def get_KDT(self):
        """ 
        generates the point positions and creates the KDtree.
        """
        if self.distribution == "uniform":
            positions = np.array([
                                np.random.random(self.size)*self.shape[0]-self.shape[0]/2, 
                                np.random.random(self.size)*self.shape[1]-self.shape[1]/2, 
                                np.random.random(self.size)*self.shape[2]-self.shape[2]/2 if self.is_2D==False else np.zeros(self.size)
                                ], dtype=object)
        elif self.distribution == "gaussian":
            positions = np.array([
                                truncated_normal(0, self.MOT_radius, -self.shape[0]/2, self.shape[0]/2).rvs(self.size), 
                                truncated_normal(0, self.MOT_radius, -self.shape[1]/2, self.shape[1]/2).rvs(self.size), 
                                truncated_normal(0, self.MOT_radius, -self.shape[2]/2, self.shape[2]/2).rvs(self.size) if self.is_2D==False else np.zeros(self.size)
                                ], dtype=object)
        else: 
            print("distribution must be \"uniform\" or \"gaussian\".")
            raise
        return KDTree(positions.T)

    def show(self):
        """
        shows the 3D cluster
        """
        if self.is_2D == True:
            self.show2D()
            return
        plt.figure(figsize=(15,12))
        ax = plt.axes(projection ="3d")
        ax.scatter(self.KDT.data[:,0],self.KDT.data[:,1],self.KDT.data[:,2], marker=".", c="b",  alpha=1, s=10)
        ax.scatter(self.KDT.data[self.cluster_excited,0],self.KDT.data[self.cluster_excited,1],self.KDT.data[self.cluster_excited,2], marker=".", c="r", alpha=1, s=100)
        plt.show()
        return
    
    def show2D(self):
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.plot(self.KDT.data[:,0],self.KDT.data[:,1], linestyle="", marker=".", c="b",  alpha=1)
        ax.plot(self.KDT.data[list(self.cluster_excited),0],self.KDT.data[list(self.cluster_excited),1], linestyle="",marker="o", c="r", alpha=1)
        ax.plot(self.KDT.data[list(self.internal_points),0],self.KDT.data[list(self.internal_points),1], linestyle="",marker=".", c="grey", alpha=1)
        for index in self.cluster_excited:
            circ = plt.Circle((self.KDT.data[index,0],self.KDT.data[index,1]), (self.facilitation_radius-self.facilitation_delta_radius/2), color='b', fill=False)
            ax.add_patch(circ)
            circ = plt.Circle((self.KDT.data[index,0],self.KDT.data[index,1]), (self.facilitation_radius+self.facilitation_delta_radius/2), color='b', fill=False)
            ax.add_patch(circ)
        lieves = list(self.get_points_connections(self.cluster_excited))
        ax.plot(self.KDT.data[lieves,0], self.KDT.data[lieves,1], linestyle="", marker=".", c="lime", markersize=8,  alpha=1)
        plt.show()
        return

    def get_points_connections(self, point_indeces):
        """ 
        retruns set of points (indeces) in the facilitation shell of the point with index "point_index".
        the facilitation shell is the shell with R in [R-dR/2, R+dR/2]. 
        in other words it returns the leaves of a given set of points.
        """
        if point_indeces == [] or len(point_indeces) == 0:
            return set([])
        
        big_ball = set(list(np.concatenate(self.KDT.query_ball_point(self.KDT.data[list(point_indeces)], self.facilitation_radius+self.facilitation_delta_radius/2)).flat))
        small_ball = set(list(np.concatenate(self.KDT.query_ball_point(self.KDT.data[list(point_indeces)], self.facilitation_radius-self.facilitation_delta_radius/2)).flat))
        return big_ball.difference(small_ball)
    
    def set_evolution_parameters(self, rabi_frequency, dephasing, detuning, vdw_C6, spontaneous_emission_rate):
        """ 
        args:
            rabi_frequency (float): the rabi frequency, in MHz
            dephasing (float): dephasing, in MHz
            detuning (float): detuning of the excitation laser, in MHz
            vdw_C6 (float): van der waals dispersion coefficient, in GHz*um^6
            spontaneous_emission_rate (float): single atom spontaneous emission rate, in MHz
        """
        self.rabi_freq = rabi_frequency
        self.dephasing = dephasing
        self.vdw_c6 = vdw_C6
        self.detuning = detuning
        self.spont_emission_rate = spontaneous_emission_rate
        self.blockade_radius = (vdw_C6/dephasing)**(1/6)
        self.facilitation_radius = 0
        self.facilitation_delta_radius = 0
        self.facilitation_rate = 0
        if detuning>0:
            self.facilitation_radius = (vdw_C6/abs(detuning))**(1/6)
            self.facilitation_delta_radius = 2*self.facilitation_radius*dephasing/(6*abs(detuning))
            if vdw_C6 != 0:
                self.facilitation_rate = get_excitation_rate(rabi_frequency, dephasing, detuning, self.vdw_c6/self.facilitation_radius**6)
        self.spont_exct_rate = get_excitation_rate(rabi_frequency, dephasing, detuning, 0)
        if self.blockade_radius < self.facilitation_radius:
            print("blockade radius (%.1f um) < facilitation radius (%.1f um)" %(self.blockade_radius, self.facilitation_radius))
        return

    def evolve(self, time, steps, excitation_steps=None, seeds = 0):
        """ 
        evolves the system
        args:
            steps (int): evolution steps
            excitation_steps (int): number of steps (from the start) in witch the sponaneous excitation are possible
        """
        self.p_spont_exct = self.spont_exct_rate * time/steps
        self.p_facilitation = self.facilitation_rate * time/steps
        self.p_spont_emission = self.spont_emission_rate * time/steps
        for _ in range(seeds):
            he_want_be_excited = np.random.choice(list(self.external_points))
            # checks the blockade contraint, and excites only who respects it
            self.excite_atom(he_want_be_excited)

        self.facilitated_points_counts = 0
        if excitation_steps is None:
            excitation_steps = steps
        p_spont_exct_aus = self.p_spont_exct
        evolution = np.zeros(steps)
        facilitated_points = np.zeros(steps)
        for i in range(steps):
            if i == excitation_steps:
                self.p_spont_exct = 0
            self.evolution_step()
            evolution[i] = len(self.cluster_excited)
            facilitated_points[i] = self.facilitated_points_counts
        self.p_spont_exct = p_spont_exct_aus
        return evolution, facilitated_points
    
    def evolution_step(self, delta_time=None):
        if delta_time is not None:
            self.p_spont_exct = self.spont_exct_rate * delta_time
            self.p_facilitation = self.facilitation_rate * delta_time
            self.p_spont_emission = self.spont_emission_rate * delta_time

        ####### spontaneus excitation ###################
            # the number of spontaneous excitation is exctracted from a poissonian, because
            # the single excitations are indipendent events and since p<<1 and N_atoms>>N_spontaneous_exct
            # the binomial distribution tends to a poissonian.
        N_spontaneous_exct = np.random.poisson(self.p_spont_exct*len(self.external_points))
        for _ in range(N_spontaneous_exct):
            he_want_be_excited = np.random.choice(list(self.external_points))
            # checks the blockade contraint, and excites only who respects it
            self.excite_atom(he_want_be_excited)

        ####### END spontaneous excitation ##############

        ####### facilitation excitation #################
            # first it computes the possible points witch can be excited by facilitation,
            # than from a binomial distribution are extracted the number of them will be excited
        facilitable_points = list(self.get_points_connections(self.cluster_excited))
        self.facilitated_points_counts = len(facilitable_points)

        if len(facilitable_points) != 0:
            N_fac_exct = np.random.binomial(len(facilitable_points),self.p_facilitation)
                # one by one the points are extracted from the facilitable_points,
                # to be sure that the facilitation constraint in respected,
                # if not less point will be excited.
            if N_fac_exct>0:
                self.excite_atom(facilitable_points[0])
                for _ in range(N_fac_exct-1):
                    facilitable_points = list(self.get_points_connections(self.cluster_excited))
                    if len(facilitable_points)>0:
                        self.excite_atom(facilitable_points[0])
        ####### END facilitation excitation #############

        ####### spontaneous emission ####################
        N_spontaneous_emission = np.random.binomial(len(self.cluster_excited), self.p_spont_emission)
        spontaneous_emissions = list(np.random.choice(list(self.cluster_excited), N_spontaneous_emission, replace=False))
        self.deexcite_atoms(spontaneous_emissions)

        ####### END spontaneous emission ################
        return 

    def excite_atom(self, he_want_be_excited):
        # add the point to the excited atoms
        self.cluster_excited.add(he_want_be_excited)
        # include all the points within the blockade radious in the internal points set
        blockade_ball = self.KDT.query_ball_point(self.KDT.data[he_want_be_excited], self.blockade_radius)
        #print("il nuovo atomo è nella sua stessa palla? :", he_want_be_excited in blockade_ball)
        self.internal_points.update(blockade_ball)
        self.external_points.difference_update(blockade_ball)
        return

    def deexcite_atoms(self, they_want_be_deexcited):
        # add the point to the excited atoms
        self.cluster_excited.difference_update(they_want_be_deexcited)
        # include all the points within the blockade radious in the internal points set
        if len(self.cluster_excited)>1:
            new_internal_points = set(list(np.concatenate(self.KDT.query_ball_point(self.KDT.data[list(self.cluster_excited)], self.blockade_radius)).flat))
        elif len(self.cluster_excited)==1:
            #print(self.KDT.query_ball_point(self.KDT.data[list(self.cluster_excited)], self.blockade_radius))
            new_internal_points = set(list(np.concatenate(self.KDT.query_ball_point(self.KDT.data[list(self.cluster_excited)], self.blockade_radius)).flat))
        else: 
            new_internal_points = set([])
        self.external_points.update(self.internal_points.difference(new_internal_points))
        self.internal_points = new_internal_points
        return

def common_member(a, b):
    """ 
    check if two lists (a, b) has no elements in common
    """
    if len(a.intersection(b)) > 0:
        return False
    return True
