"""
    poincare.py
    Jul 24 2022
    Gabriel Moreira
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def complex_mobius_transform(p):
    """
        Returns an hyperbolic isometry that sends p to the origin
    """
    transform = lambda z : (z - p) / (1 - np.conj(p) * z)
    return transform


def complex_mobius_inverse(p):
    """
        Inverse hyperbolic isometry of the Mobius transform
        Maps the origin back to p
    """
    transform = lambda w : (w + p) / (1 + np.conj(p) * w)
    return transform



def delaunay_tree_embedding(graph, root_id, tau):
    """
        Computes the Poincare embeddings of a graph
        Parameter epsilon controls the distortion
    """
    id2num = {node : i for i, node in enumerate(graph.nodes)}
    num2id = {i : node for i, node in enumerate(graph.nodes)}

    scale = (np.exp(tau) - 1) / (np.exp(tau) + 1)

    # Start by embedding the root and its children
    embeddings = {root_id : np.array([0, 0]).astype(np.float64)}
    queue      = np.zeros([graph.number_of_nodes(), 2])

    j = 0
    for child_id in nx.neighbors(graph, root_id):
        embedding = scale * np.exp(2*np.pi*j*1j / nx.degree(graph,root_id))
        embeddings[child_id] = np.array([np.real(embedding), np.imag(embedding)]).astype(np.float64)
        queue[j,:] = np.array([id2num[root_id], id2num[child_id]])
        j = j + 1
    
    k = 0
    while k < graph.number_of_nodes()-1:
        # For all nodes:
        # 1) Mobius transform to place the node at the origin
        # 2) Add children evenly spaced
        # 3) Revert Mobius transform
        grandparent_num = queue[k,0]
        grandparent_id  = num2id[grandparent_num]
        parent_num      = queue[k,1]
        parent_id       = num2id[parent_num]

        grandparent_embedding   = complex(embeddings[grandparent_id][0], embeddings[grandparent_id][1])
        parent_embedding        = complex(embeddings[parent_id][0],      embeddings[parent_id][1])
        grandparent_transformed = mobius_transform(parent_embedding)(grandparent_embedding)

        theta = np.angle(grandparent_transformed)

        n = 0
        for child_id in nx.neighbors(graph, parent_id):
            if id2num[child_id] != id2num[grandparent_id]:
                transformed_embedding = scale * np.exp((theta + (2*np.pi*n/nx.degree(graph,parent_id)))*1j)
                embedding = mobius_inverse(parent_embedding)(transformed_embedding)
                embedding = embedding if np.abs(embedding) <= 1 else embedding / np.abs(embedding)
                embeddings[child_id] = np.array([np.real(embedding), np.imag(embedding)]).astype(np.float64)
                queue[j,:] = np.array([id2num[parent_id], id2num[child_id]])
                j = j + 1
            n = n + 1
        
        k = k + 1
        if len(queue) == 0:
            break

    return embeddings
