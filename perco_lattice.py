#!/usr/bin/python3
# Percolation Lattice
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from perco_functions import tau

class percoLattice:
    def __init__(self, xdim, ydim=None, zdim=None, p=0.75):
        """
        Forms microclusters for all points in the graph.
    
        Parameters
        ----------
        p : float
            Probability of fusion of two photons.
        """
        if ydim == None and zdim == None:
            ydim, zdim = xdim, xdim
        elif zdim == None:
            zdim = ydim
        self.__xdim, self.__ydim, self.__zdim = xdim, ydim, zdim
        self.__p = p
        M = nx.Graph()
        M.graph['xdim'], M.graph['ydim'], M.graph['zdim'] = xdim, ydim, zdim
        M.graph['Prob'] = p
        probs = np.array([p**2, p**2+p*(1-p), p**2+2*(1-p)*p, 1.0])
        nvals = [3, 2, 1, 0]
        for x in range(1, xdim+1):
            for y in range(1, ydim+1):
                for z in range(1, zdim+1):
                    M.add_node(tau(x, y, z))
                    M.node[tau(x, y, z)]['X'] = x
                    M.node[tau(x, y, z)]['Y'] = y
                    M.node[tau(x, y, z)]['Z'] = z
                    # Marker for when connecting the nodes so 
                    # to avoid double-counting branches.
                    M.node[tau(x, y, z)]['Checked'] = False
                    # Now simulates stochastic microcluster formation success.
                    rand = nr.random()
                    accept = list(rand < probs)
                    try:
                        nv = nvals[accept.index(False)]
                    except ValueError:
                        nv = 0
                    # Returns the first False where rand falls in prob range,
                    # instead of extra lines of if-else statements.
                    M.node[tau(x, y, z)]['value'] = nv
        self.__latt = M
        self.__connected = False
    
    
    def connect_lattice(self):
        for node in self.__latt.nodes():
            self.connect(node)
        self.__connected = True
            
        
    def connect(self, node):
        """
        Connects microclusters at adjacent nodes, probability p of a successful
        connection. If the fusion is unsuccessful then connection is ignored 
        and not considered again. If an edge already exists between two nodes, 
        the connection is not repeated.
        """
        self.__latt.node[node]['Checked'] = True
        self.determine_shell(node)
        for mate in self.__latt.node[node]['shell']:
            # For every node in the 'shell' of nearest neighbouring nodes.
            self.determine_shell(mate)
            # For any sort of fusion, need to consider the probability.
            if node in self.__latt.node[mate]['shell']:
                # This condition tests for a direct (non-diagonal) fusion.
                if not self.__latt.node[mate]['Checked']:
                    rand = nr.random()
                    if rand <= self.__p:
                        self.__latt.add_edge(node, mate)
                continue
            else:        
                # Not found a suiting node to connect to yet.
                connected = False
                prior, posterior, jump_count = node, mate, 1
                while not connected:
                    jump_count += 1
                    next_node, to_continue = self.wise_chain(prior, posterior)
                    # Decides the next node to consider for a fusion to in
                    # the event of a diagonal connection.
                    if not to_continue or next_node == None:
                        """Hit a boundary, no diagonal fusion possible."""
                        connected = True
                        break
                    else:
                        prior, posterior = posterior, next_node
                        self.determine_shell(posterior)
                        if prior in self.__latt.node[posterior]['shell']:
                            connected = True
                            if not self.__latt.node[posterior]['Checked']:
                                if nr.random() <= self.__p**jump_count:
                                    self.__latt.add_edge(node, posterior)
                            break
                        else: 
                            continue
                        
                        
    def determine_shell(self, node):
        """
        Instantiates a shell of immediately neighbouring points that
        correspond to the fusion connections in a brickwork pattern. This is 
        done by a positive and negative x-direction connection, and then
        alternation each consecutive node in any of three directions, between
        positive and negative connections in y-direction and z-direction.
        """
        shell = []
        nodex = self.__latt.node[node]['X']
        nodey = self.__latt.node[node]['Y']
        nodez = self.__latt.node[node]['Z']
        value = self.__latt.node[node]['value']
        # Factor to determine the alternating brickwork lattice connections.
        factor = (-1)**(nodey+nodez+nodex)
        # The x-dimension is not affected by alternating factor.
        if 1 < nodex < self.__xdim:
            if value in [3, 2]:
                alt_ = 1 if value == 3 else -1
                shell.append(tau(nodex+alt_, nodey, nodez))
                shell.append(tau(nodex-alt_, nodey, nodez))
        elif nodex == self.__xdim:
            if value in [3, 2]:
                shell.append(tau(nodex-1, nodey, nodez))
        else:
            if value in [3, 2]:
                shell.append(tau(nodex+1, nodey, nodez))
        # The y-dimension is affected by alternating factor.
        if 1 < nodey < self.__ydim:
            if value in [3, 1]:
                shell.append(tau(nodex, nodey+factor, nodez))
        elif nodey == self.__ydim:
            if factor == -1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey-1, nodez))
        else:
            if factor == +1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey+1, nodez))
        # The z-dimension is affected by alternating factor
        if 1 < nodez < self.__zdim:
            if value in [3, 1]:
                shell.append(tau(nodex, nodey, nodez+factor))
        elif nodez == self.__zdim:
            if factor == -1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey, nodez-1))
        else:
            if factor == 1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey, nodez+1))
        self.__latt.node[node]['shell'] = shell
        
            
        
    def wise_chain(self, node, mate):
        """
        Procedure to determine next node in the series of lattice points to
        consider, given the diamond/brickwork connection convention.
        """
        # Considers the neighbouring 'mate' of the current node.
        xm = self.__latt.node[mate]['X']
        ym = self.__latt.node[mate]['Y']
        zm = self.__latt.node[mate]['Z']
        mfactor = (-1)**(xm + ym + zm)
        # There is no next unless convinced otherwise.
        to_continue, next_node = True, None
        if xm == self.__latt.node[node]['X'] + 1:
            # If +ve 1 in the x-direction
            if xm == self.__xdim:
                to_continue = False
            else:
                next_node = tau(xm + 1, ym, zm)
        elif xm == self.__latt.node[node]['X'] - 1:
            # If -ve 1 in the x-direction
            if xm == 1:
                # If hitting the edge of the lattice
                to_continue = False
            else:
                next_node = tau(xm - 1, ym, zm)
        elif int(np.abs(self.__latt.node[node]['Y'] - ym)) == 1:
            # If a positive of negative difference in the y-direction.
            if mfactor == 1:
                if zm == self.__zdim:
                    to_continue = False
                else:
                    next_node = tau(xm, ym, zm+1)
            elif mfactor == -1:
                if zm == 1:
                    to_continue = False
                else:
                    next_node = tau(xm, ym, zm-1)
        elif int(np.abs(self.__latt.node[node]['Z'] - zm)) == 1:
            # If a positive of negative difference in the z-direction.
            if mfactor == 1:
                if ym == self.__ydim:
                    to_continue = False
                else:
                    next_node = tau(xm, ym +1, zm)
            elif mfactor == -1:
                if ym == 1:
                    to_continue = False
                else:
                    next_node = tau(xm, ym-1, zm)
        return next_node, to_continue

        
    def edge_count(self):
        # edge_counts = [x, y, z, diagonal]
        edge_counts = [0, 0, 0, 0]
        for edge in self.__latt.edges():
            n1, n2 = edge[0], edge[1]
            xdiff = np.abs(self.__latt.node[n1]['X']-self.__latt.node[n2]['X'])
            ydiff = np.abs(self.__latt.node[n1]['Y']-self.__latt.node[n2]['Y'])
            zdiff = np.abs(self.__latt.node[n1]['Z']-self.__latt.node[n2]['Z'])
            metric = [int(xdiff), int(ydiff), int(zdiff)]
            if sum(metric) == 1:
                indx = metric.index(1)
                edge_counts[indx] += 1
            else:
                edge_counts[3] += 1
        return edge_counts
    
    
    def average_cluster(self):
        clusters = 0
        clusters_sq = 0
        total_no = 0.0
        for node in self.__latt.nodes():
            len_clust = len(nx.node_connected_component(self.__latt, node))
            if len_clust != 1:
                clusters += 1
                clusters_sq += len_clust
                total_no += 1./len_clust
        if clusters == 0:
            return [0, 1, 1]
        else:
            mean = clusters/total_no
            mean_sq = clusters_sq/total_no
            std = np.sqrt(np.abs(mean_sq-mean**2))
            return mean, std, total_no
        
        
    def directed_bonds(self, planar=False):
        bonds = 0
        for edge in self.__latt.edges():
            x1,x2=self.__latt.node[edge[0]]['X'],self.__latt.node[edge[1]]['X']
            condition = (x1 == x2) if planar else (x1 != x2)
            if condition:
                bonds += 1
        if self.__latt.edges() == 0:
            return 0
        else:
            return float(bonds)/self.__latt.edges()
        
    def shortest_path(self):
        percolate = self.assess_percolation()
        if percolate:
            for y in range(1, self.__ydim +1):
                for z in range(1, self.__zdim +1):
                    self.__latt.add_edge(tau(1, y, z), 'Back')
                    self.__latt.add_edge(tau(self.__xdim, y, z), 'Front')
            length = len(nx.shortest_path(self.__latt, 'Back', 'Front')) - 2
            self.__latt.remove_node('Back')
            self.__latt.remove_node('Front')
            return length
        else:
            return 0
        
    def largest_cluster(self, distribution=False):
        sizes = []
        for node in self.__latt.nodes():
            if len(nx.node_connected_component(self.__latt, node)) != 1:
               sizes.append(len(nx.node_connected_component(self.__latt,node)))
        mx = max(sizes) if len(sizes)>0 else 0
        if not distribution:
            del sizes
            return mx
        else:
            return sizes
    
    
    def assess_percolation(self):
        """Not for use with extendable lattices or overlap windows"""
        # Create two panels of nodes at the front and back of lattice
        front_nodes, back_nodes = self.panels()
        if not self.__connected:
            for node in self.__latt.nodes():
                self.connect(node)
            self.__connected = True
        for node1, node2 in product(back_nodes, front_nodes):
            # Detects any percolating path between nodes at the front or back.
            if node2 in nx.node_connected_component(self.__latt, node1):
                return True
        # Return False if no connections found.
        return False
    
    
    def panels(self):
        """Create two panels of nodes at the front and back of lattice."""
        front, back = [], []
        for y, z in product(range(1, self.__ydim+1), range(1, self.__zdim+1)):
            front.append(tau(self.__xdim, y , z))
            back.append(tau(1, y, z))
        return (back, front)
    
        
    def visualise(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xxs, xxh, xxf = [], [], []
        yys, yyh, yyf = [], [], []
        zzs, zzh, zzf = [], [], []
        for node in self.__latt.nodes():
            if self.__latt.node[node]['value'] == 3:
                xxs.append(self.__latt.node[node]['X'])
                yys.append(self.__latt.node[node]['Y'])
                zzs.append(self.__latt.node[node]['Z'])
            elif self.__latt.node[node]['value'] in [2, 1]:
                xxh.append(self.__latt.node[node]['X'])
                yyh.append(self.__latt.node[node]['Y'])
                zzh.append(self.__latt.node[node]['Z'])
            else:
                xxf.append(self.__latt.node[node]['X'])
                yyf.append(self.__latt.node[node]['Y'])
                zzf.append(self.__latt.node[node]['Z'])
        Axes3D.scatter(ax, xxs, yys, zzs, s = 12, color = 'k')
        Axes3D.scatter(ax, xxh, yyh, zzh, s = 12, color = 'g')
        Axes3D.scatter(ax, xxf, yyf, zzf, s = 12, color = 'r')
        ax.set_xlim((1, self.__xdim))
        ax.set_ylim((1, self.__ydim))
        ax.set_zlim((1, self.__zdim))
        for edge in self.__latt.edges():
            x1,x2=self.__latt.node[edge[0]]['X'],self.__latt.node[edge[1]]['X']
            y1,y2=self.__latt.node[edge[0]]['Y'],self.__latt.node[edge[1]]['Y']
            z1,z2=self.__latt.node[edge[0]]['Z'],self.__latt.node[edge[1]]['Z']
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'm-')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()