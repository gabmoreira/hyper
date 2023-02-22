"""
    hyperbolic/plot.py
    Feb 21 2022
    Gabriel Moreira
"""

import numpy as np
import matplotlib.pyplot as plt
from .functional import mobius_inverse, mobius_transform

theme1 = {'fig_size'         : (14,5),
          'circle_fill'      : True,
          'circle_color'     : [253/255, 240/255, 213/255],
          'circle_linewidth' : 0.7,
          'node_fill_color'  : [102/255, 155/255, 188/255, 0.75],
          'node_edge_color'  : [0.0, 0.0, 0.0],
          'node_size'        : 100,
          'node_linewidth'   : 0,
          'edge_color'       : [0, 48/255, 73/255],
          'edge_linewidth'   : 0.3}


theme2 = {'fig_size'         : (14,5),
          'circle_fill'      : True,
          'circle_color'     : [253/255, 240/255, 213/255],
          'circle_linewidth' : 0.7,
          'node_fill_color'  : [193/255, 18/255, 31/255, 0.75],
          'node_edge_color'  : [0.0, 0.0, 0.0],
          'node_size'        : 100,
          'node_linewidth'   : 0,
          'edge_color'       : [0, 48/255, 73/255],
          'edge_linewidth'   : 0.3}


def poincare_figure(figsize=(14,5),
                    k=-1,
                    circle_color=[0.0, 0.0, 0.0, 1.0],
                    circle_fill=False,
                    circle_linewidth=0.7):

    r = 1.0 / np.sqrt(-k)
    
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)

    circle = plt.Circle((0, 0),
                        r,
                        color=circle_color,
                        fill=circle_fill,
                        linewidth=circle_linewidth)
    ax.add_patch(circle)

    ax.set_xlim([-0.05-r,0.05+r])
    ax.set_ylim([-0.05-r,0.05+r])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.locator_params(axis='x', nbins=4)

    ax.set_aspect('equal')

    return fig, ax


class PoincareSegment:
    def __init__(self, p, q, k=-1):
        """
            Geodesic segment in Poincare 2D disk model

            Parameters 
            ----------
            p - start point
            q - end point
        """
        self.k = k
        self.p = p
        self.q = q
        self.q_transformed = mobius_transform(self.p, self.k)(self.q)
        
    def points(self, n):
        inverted_pts = np.array([self.q_transformed * i/n for i in range(0,n+1)])
        pts = mobius_inverse(self.p, self.k)(inverted_pts)
        return pts

    def draw(self, ax, n=200, color=[0,0,0], linewidth=.5):
        """
        """
        pts = self.points(n)
        ax.plot(np.real(pts), np.imag(pts), '-', linewidth=linewidth, color=color, zorder=5)


def poincare_graph_draw(graph,
                        root_id,
                        embeddings,
                        ax,
                        node_size=100,
                        node_fill_color=[1.0, 1.0, 1.0, 1.0],
                        node_edge_color=[0.0, 0.0, 0.0],
                        node_linewidth=0.5,
                        edge_color=[0.0, 0.0, 0.0],
                        edge_linewidth=0.3,
                        beta=0.9):
    """
        Draws Networkx graph in Poincaré 2D disk model of the hyperbolic plane
    """
    if root_id is not None:
        transform  = mobius_transform(embeddings[root_id])
        embeddings = {n : np.array([np.real(transform(embeddings[n])),
                                    np.imag(transform(embeddings[n]))]) for n in graph.nodes()}

    if edge_linewidth > 0:
        for e in graph.edges():
            segment = PoincareSegment(embeddings[e[0]],embeddings[e[1]])
            segment.draw(ax, 100, edge_color, edge_linewidth)
        
    size = [node_size * (1-beta*np.linalg.norm(pos)) for _, pos in embeddings.items()]

    plt.scatter([pos[0] for _, pos in embeddings.items()],
                [pos[1] for _, pos in embeddings.items()],
                 color=node_fill_color,
                 s=size,
                 edgecolors=node_edge_color,
                 linewidths=node_linewidth,
                 zorder=2)



def hyperboloid_plot(pos,
                     ax,
                     node_fill_color=[1.0, 1.0, 1.0, 1.0],
                     node_size=40,
                     edge_color=[0.0, 0.0, 0.0],
                     edge_width=0.3):
    """
    """
    assert pos.shape[1] == 3

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],
                 color=node_fill_color, s=node_size, edgecolors=edge_color, linewidths=.7) 


class HyperboloidSegment:
    def __init__(self, p, q):
        """
            Geodesic segment in Hyperboloid in 3-Minkowski space

            Parameters 
            ----------
            p - start point
            q - end point
        """
        self.p = p
        self.q = q

        self.v = (self.q - self.p * np.cosh(1.0)) / np.sinh(1.0)

    def draw(self, ax, color=[.8,.8,.8], linewidth=.5):
        """
        """
        step = 0.01
        t = np.arange(0, 1.0 + step, step)
        pts = self.p * np.cosh(t[:,np.newaxis]) + self.v * np.sinh(t[:,np.newaxis])
        ax.plot3D(pts[:,0], pts[:,1], pts[:,2], '-', linewidth=linewidth, color=color, zorder=1)



def hyperboloid_graph_draw(graph,
                           embeddings,
                           ax,
                           node_color=[1.0, 1.0, 1.0, 1.0],
                           node_size=40,
                           edge_color=[0.0, 0.0, 0.0],
                           edge_width=0.3):
    """
        Draws Networkx graph in hyperboloid model of the hyperbolic plane
    """
    for e in graph.edges():
        segment = HyperboloidSegment(embeddings[e[0]], embeddings[e[1]])
        segment.draw(ax, edge_color, edge_width)
        
    ax.scatter3D([pos[0] for _, pos in embeddings.items()],
                 [pos[1] for _, pos in embeddings.items()],
                 [pos[2] for _, pos in embeddings.items()],
                  color=node_color, s=node_size, edgecolors='black', linewidths=.7, zorder=2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

