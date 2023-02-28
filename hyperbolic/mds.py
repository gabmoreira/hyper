"""
    hyperbolic/mds.py
    Aug 12 2022
    Gabriel Moreira
"""

import torch
from tqdm import tqdm

import functional as hf

def euclidean_mds(cdist, x_init, lr=1.0, num_iter=500, verbose=True):
    """
        Euclidean multidimensional scaling (MDS)

        Parameters 
        ----------
        cdist    - torch.Tensor with shape (n, n) pairwise distances
        x_init   - torch.Tensor with shape (n, dim_embedding) initial estimate
        lr       - float learning rate
        num_iter - int number iterations
        verbose  - bool
            
        Output
        ---------
        X - torch.Tensor with shape of Xinit (n, dim_embedding)
    """
    assert cdist.shape[0] == cdist.shape[1]
    assert cdist.shape[0] == x_init.shape[0]
    assert x_init.device  == cdist.device
    
    device = x_init.device
    n      = x_init.shape[0]
    dim    = x_init.shape[1]

    with torch.no_grad():
        x = torch.empty((n, dim), requires_grad=True, dtype=torch.float64, device=device)
        x.data.copy_(x_init)

    optimizer = torch.optim.Adam([x], lr=lr)

    bar = tqdm(total=num_iter, dynamic_ncols=True, desc="Embedding in E^{}".format(dim)) 
    for _ in range(num_iter):
        embedding_distances = torch.cdist(x, x, p=2)
        distortion = torch.sum(torch.square(embedding_distances - cdist)) / (n * (n-1))
                    
        distortion.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            x[0,:] = x[0,:]*0
            
        bar.set_postfix(distortion="{:0.5f}".format(distortion))
        bar.update()
    bar.close()

    x = x.detach()
    
    return x


def hyperboloid_mds(cdist, x_init, k=-1.0, lr=1.0, num_iter=10000, verbose=True):
    """
        Hyperbolic multidimensional scaling (MDS)

        Parameters 
        ----------
        cdist    - torch.Tensor with shape (n, n) pairwise distances
        x_init   - torch.Tensor with shape (n, dim_manifold + 1) initial estimate
        lr       - float learning rate
        num_iter - int number inerations
        verbose  - bool
            
        Output
        ---------
        X - torch.Tensor with shape of Xinit (n, dim_manifold + 1)
    """
    assert cdist.shape[0] == cdist.shape[1]
    assert cdist.shape[0] == x_init.shape[0]
    assert x_init.device  == cdist.device
    
    device = x_init.device
    n      = x_init.shape[0]
    dim    = x_init.shape[1] - 1

    with torch.no_grad():
        x = torch.empty((n, dim + 1), requires_grad=True, dtype=torch.float64, device=device)
        x.data.copy_(x_init)

    cdist = cdist.to(device)
    
    bar = tqdm(total=num_iter, dynamic_ncols=True, desc="Embedding in H^{}".format(dim)) 
    for _ in range(num_iter):
        embedding_distances = hf.lorentz_cdist(x, x, k)
        embedding_distances.fill_diagonal_(0.0)
        
        distortion = torch.sum(torch.square(embedding_distances - cdist)) / (n * (n-1))
        distortion.backward()

        with torch.no_grad():
            l_grad = lorentz_grad(x)
            
            if l_grad.isnan().any():
                print("Lorentz grad NaN")
                break
            x.data.copy_(hf.lorentz_exp(x, -lr * l_grad, k=k).data)
            hyperbolic_grad_norm = torch.norm(l_grad, 'fro', dim=None)
            
            if x.isnan().any():
                print("x NaN")
                break

            x.grad.fill_(0.0)

        with torch.no_grad():
            x.data.copy_(lorentz_inclusion(x[...,:-1], k=k))
            
        bar.set_postfix(distortion="{:0.5f}".format(distortion),
                        hyperbolic_grad="{:1.2e}".format(hyperbolic_grad_norm))
        bar.update()
    bar.close()

    x = x.detach()
    
    return x






def embedding_map(graph : nx.classes.graph.Graph,
                  embedding : dict,
                  metric : Callable) -> float:
    mean_avg_precision = 0
    # Iterate over all nodes in the graph
    for node_a, deg_a in tqdm(graph.degree()):
        # Neighbors of the node
        nbrs_node_a = set(node_b for node_b in graph.neighbors(node_a))
        # Distances from node to all other nodes
        dist_node_a = {n : metric(embedding[node_a], embedding[n]) for n in graph.nodes()}

        node_map = 0
        for node_b in nbrs_node_a:
            # distance from the node to the neighbor
            dist_a_b = dist_node_a[node_b]

            smallest_set = set(node for node, dist in dist_node_a.items() if dist <= dist_a_b)
            smallest_set.remove(node_a)
            
            node_map += len(nbrs_node_a.intersection(smallest_set)) / len(smallest_set)
            
        #print("{} : neighbors: {}   smallest_set: {}".format(node_a, nbrs_node_a, smallest_set))
        mean_avg_precision += node_map / deg_a

    mean_avg_precision /= graph.number_of_nodes()
    return mean_avg_precision