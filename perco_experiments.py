#!/usr/bin/python3
from perco_lattice import percoLattice
from perco_functions import fermi_dist
from multiprocessing import Pool, cpu_count
import numpy as np
import networkx as nx
import numpy.random as nr
import scipy.optimize as spo
import scipy.stats as sps
from functools import partial
import matplotlib.pyplot as plt


def quick_perc(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    res = M.assess_percolation()
    del M
    return res


def quick_average(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    M.connect_lattice()
    res = M.average_cluster()
    del M
    return res


class percoExp:
    def __init__(self, xd, yd=None, zd=None, n_iter=250, p_step=0.01,
                 p_start=0.1, p_end=0.9, parallel=False, ncpus=None):
        if yd == None:
            yd = xd
        if zd == None:
            zd = xd
        self.__xd, self.__yd, self.__zd = xd, yd, zd
        self.__p_range = np.arange(p_start, p_end, p_step)
        self.__parallel = parallel
        self.__ncpus = cpu_count() if ncpus == None else ncpus
        

    def find_quantity(self, quantity='threshold', random_state=22, 
                      fitting=False, fit_func=fermi_dist,
                      plot=False, plotfile=None):
        """
        Determines the threshold probability of percolation of a lattice of 
        given dimensions
    
        Parameters
        ----------
        """
        # Need to have some dictionary to find quantities
        quickfunc = quick_perc
        nr.seed(random_state)
        vals = []
        dims = ( self.__xd, self.__yd, self.__zd)
        for p in self.__p_range:
            if self.__parallel:
                procs = cpu_count() if self.__ncpus == None else self.__ncpus
                P = Pool(procs)
                func = partial(quickfunc, *dims)
                outcomes = P.map(func, p*np.ones(self.__n_iter))
                outcomes = [float(oc) for oc in outcomes]
            else:
                outcomes = []
                for i in range(self.__n_iter):
                    outcomes.append(float(quickfunc(*dims, p)))
            vals.append(np.average(outcomes))
        if fitting:
            fits=spo.curve_fit(fitting_func, self.__p_range, vals, p0=[0.5,1])
        ##############################3
        ########## contain plot in something else
        if plot:
            fig = plt.figure()
            hard_range= np.arange(0.0, 1.0, 0.005)
            ax = fig.add_subplot(111)
            ax.plot(self.__p_range, perc_vals, 'ro')
            ax.plot(hard_range, fitting_func(hard_range, fits[0][0], fits[0][1]),'k-')
            if plotfile != None:
                fig.savefig(plotfile)
                
                
                
        return [p_range, vals]

    
def average_clusters(xd, yd=None, zd=None, n_iter=100, p_step=0.01,p_start=0.1,
                     p_end=0.9, parallel=False, ncpus=None, proper_errors=True,
                     random_state=22, plot=False, plotfile=None):
    """Gives the average cluster size that an arbitrary node belongs to"""
    averages, err_ = [], []
    p_range = np.arange(p_start, p_end, p_step)
    nr.seed(random_state)
    for p in p_range:
        if parallel:
            procs = cpu_count() if ncpus == None else ncpus
            P = Pool(procs)
            func = partial(quick_average, xd, yd, zd)
            outcomes = P.map(func, p*np.ones(n_iter))
        else:
            outcomes = []
            for i in range(n_iter):
                outcomes.append(quick_average(xd, yd, zd, p))
        means = [out[0] for out in outcomes]
        if proper_errors:
            stderr = [out[1]/np.sqrt(out[2]) for out in outcomes]
            se_sq_inv = [1./se**2 for se in stderr]
            ssei = sum(se_sq_inv)
            mu_over_se_sq = sum([means[i]*se_sq_inv[i] for i in range(n_iter)])
            full_mean, full_err= mu_over_se_sq/ssei, np.sqrt(ssei/n_iter)
        else:
            full_mean = np.mean(means)
            full_err = sps.sem(means)
        err_.append(full_err)
        averages.append(full_mean)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(p_range, averages, 'ro')
        ax.errorbar(p_range, averages, yerr=err_)
        if plotfile != None:
            fig.savefig(plotfile)
    return [p_range, averages, err_]
    

def count_forward_bonds(xdim, ydim, zdim, reps = 300, prob=0.75):
    """Calculates the percentage of the bonds in the lattice which are forward in the x direction"""
    averages = []
    for rep in range(reps):
        M = build_lattice(xdim, ydim, zdim, prob)
        assess_percolation(M)
        FWD = 0
        for edge in M.edges():
            if M.node[edge[0]]['X'] != M.node[edge[1]]['X']:
                FWD += 1
            else:
                continue
        if len(M.edges()) == 0.0:
            averages.append(0)
        else:
            averages.append(float(FWD)/(len(M.edges())))
    return [np.mean(averages), sps.sem(averages)]


        
def planar_bonds(xdim, ydim, zdim, reps = 300, prob=0.75):
    """Calculates the percentage of bonds in the lattice which are sideways and in the y-z plane"""
    averages = []
    for rep in range(reps):
        M = build_lattice(xdim, ydim, zdim, prob)
        assess_percolation(M)
        planar = 0
        for edge in M.edges():
            if M.node[edge[0]]['X'] == M.node[edge[1]]['X']:
                planar += 1
            else:
                continue
        if len(M.edges()) == 0.0:
            averages.append(0)
        else:
            averages.append(float(planar)/(len(M.edges())))
    return [np.mean(averages), sps.sem(averages)]
    

    
def average_path_length(xdim, ydim, zdim, prob, reps=400):
    """Calculates the average length of a spanning path, given that it exists"""
    lengths = []
    for rep in range(reps):
        M = build_lattice(xdim ,ydim, zdim, prob)
        Y = assess_percolation(M)
        M.add_node('Back')
        M.add_node('Front')
        if Y == True:
            for y in range(1, ydim +1):
                for z in range(1, zdim +1):
                    M.add_edge(tau(1, y, z), 'Back')
                    M.add_edge(tau(xdim, y, z), 'Front')
            length = len(nx.shortest_path(M, 'Back', 'Front')) - 2
            lengths.append(length)
        else:
            continue
    return [np.mean(lengths), sps.sem(lengths)]
        
            

def largest_cluster(xdim, ydim, zdim, prob, reps=400):
    """Calculates the largest typical cluster size"""
    maxes = []
    for rep in range(reps):
        M = build_lattice(xdim, ydim, zdim, prob)
        assess_percolation(M)
        cluster_sizes = []
        for node in M.nodes():
            if len(nx.node_connected_component(M, node)) != 1:
               cluster_sizes.append(len(nx.node_connected_component(M, node)))
            else: continue
        maxes.append(max(cluster_sizes))
    return np.mean(maxes)
            
    
