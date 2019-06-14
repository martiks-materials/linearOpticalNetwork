#!/usr/bin/python3
from perco_lattice import percoLattice
from perco_functions import fermi_dist
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.random as nr
import scipy.optimize as spo
import scipy.stats as sps
from functools import partial
import matplotlib.pyplot as plt


class percoExperiment:
    def __init__(self, xd, yd=None, zd=None, n_iter=250, p_step=0.01,
                 p_start=0.1, p_end=0.9, parallel=False, ncpus=None,
                 random_state=42):
        if yd == None:
            yd = xd
        if zd == None:
            zd = xd
        self.__xd, self.__yd, self.__zd = xd, yd, zd
        self.__n_iter = n_iter
        self.__p_range = np.arange(p_start, p_end, p_step)
        self.__parallel = parallel
        self.__random_state = random_state
        self.__ncpus = cpu_count() if ncpus == None else ncpus
        

    def find_quantity(self, quickfunc, fit=False, proper_stats=True,
                      fit_func=fermi_dist, plot=False, plotfile=None):
        """
        Determines the threshold probability of percolation of a lattice of 
        given dimensions
    
        Parameters
        ----------
        """
        # Need to have some dictionary to find quantities
        nr.seed(self.__random_state)
        vals, err_ = [], []
        xd, yd, zd= self.__xd, self.__yd, self.__zd
        procs = cpu_count() if self.__ncpus == None else self.__ncpus
        P = Pool(procs)
        for p in self.__p_range:
            if self.__parallel:
                func = partial(quickfunc, xd, yd, zd)
                outcomes = P.map(func, p*np.ones(self.__n_iter))
                #outcomes = [float(oc) for oc in outcomes]
            else:
                outcomes = []
                for i in range(self.__n_iter):
                    outcomes.append(quickfunc(xd, yd, zd, p))
            processed = process_outcomes(outcomes, self.__n_iter, proper_stats)
            vals.append(processed[0])
            err_.append(processed[1])
            del outcomes
        if fit:
            fitting=spo.curve_fit(fit_func, self.__p_range, vals, p0=[0.5,1])
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.__p_range, vals, 'ro')
            ax.errorbar(self.__p_range, vals, yerr=err_)
            if fit:
                hard_range= np.arange(0.0, 1.0, 0.005)
                ax.plot(hard_range, fit_func(hard_range, fitting[0][0], \
                                             fitting[0][1]),'k-')
            if plotfile != None:
                fig.savefig(plotfile)
        return [self.__p_range, vals, err_]
    
    
    
def func_perc(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    res = M.assess_percolation(), 1, 1
    del M
    return res

def func_average_cluster(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    M.connect_lattice()
    res = M.average_cluster()
    del M
    return res

def func_forward_bonds(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    M.connect_lattice()
    res = M.directed_bonds(planar=False), 1, 1
    del M
    return res

def func_planar_bonds(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    M.connect_lattice()
    res = M.directed_bonds(planar=True), 1, 1
    del M
    return res

def func_shortest_path(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    res = M.shortest_path(), 1, 1
    del M
    return res

def func_max_cluster(xd, yd, zd, p0):
    M = percoLattice(xd, yd, zd, p0)
    M.connect_lattice()
    res = M.largest_cluster(distribution=False), 1, 1
    del M
    return res

def process_outcomes(outcomes, n_iter, proper_stats):
    means = [out[0] for out in outcomes]
    if proper_stats:
        stderr = [out[1]/np.sqrt(out[2]) for out in outcomes]
        se_sq_inv = [1./se**2 for se in stderr]
        ssei = sum(se_sq_inv)
        mu_over_se_sq = sum([means[i]*se_sq_inv[i] for i in range(n_iter)])
        full_mean, full_err= mu_over_se_sq/ssei, np.sqrt(ssei/n_iter)
    else:
        full_mean, full_err = np.mean(means), sps.sem(means)
    return full_mean, full_err