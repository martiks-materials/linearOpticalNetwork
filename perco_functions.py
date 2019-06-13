#!/usr/bin/python3
import numpy as np

def tau(x, y, z):
    """Qubit Naming Function: Converts coordinates into node keys."""
    return "{}:{}:{}".format(x, y, z)

def fermi_dist(x, a, b):
    return 1/(np.exp((a-x)/b)+1)