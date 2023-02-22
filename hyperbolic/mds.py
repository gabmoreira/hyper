"""
    hyperbolic/mds.py
    Aug 12 2022
    Gabriel Moreira
"""

import torch
from tqdm import tqdm
from hyperbolic.hyperboloid import Hyperboloid


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

        bar.set_postfix(distortion="{:0.5f}".format(distortion))
        bar.update()
    bar.close()

    x = x.detach()
    
    return x


def hyperboloid_mds(cdist, x_init, lr=1.0, num_iter=10000, verbose=True):
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

    hyperboloid = Hyperboloid(dim, device=device)

    with torch.no_grad():
        x = torch.empty((n, dim + 1), requires_grad=True, dtype=torch.float64, device=device)
        x.data.copy_(x_init)

    cdist = cdist.to(device)
    
    bar = tqdm(total=num_iter, dynamic_ncols=True, desc="Embedding in H^{}".format(dim)) 
    for _ in range(num_iter):
        inner_products = torch.clamp(-torch.einsum('k,ik,jk->ij', hyperboloid.metric, x, x), min=1.0)
        inner_products.fill_diagonal_(1.0)

        embedding_distances = torch.acosh(inner_products)
        embedding_distances.fill_diagonal_(0.0)
        
        distortion = torch.sum(torch.square(embedding_distances - cdist)) / (n * (n-1))
        distortion.backward()

        with torch.no_grad():
            x.hyperboloid_grad = hyperboloid.grad(x)
            if x.hyperboloid_grad.isnan().any():
                print("Euclidean grad NaN")
                break
            x.data.copy_(hyperboloid.expmap(x, -lr * x.hyperboloid_grad).data)
            hyperbolic_grad_norm = torch.norm(x.hyperboloid_grad, 'fro', dim=None)
            if x.isnan().any():
                print("Hyperbolic grad NaN")
                break

            x.grad.fill_(0.0)

        bar.set_postfix(distortion="{:0.5f}".format(distortion),
                        hyperbolic_grad="{:1.2e}".format(hyperbolic_grad_norm))
        bar.update()
    bar.close()

    x = x.detach()
    
    return x
