# gradoptorch
Gradient based optimizers for python with a PyTorch backend. This package intends to allow more fine-grain control of optimizers without getting lost in the PyTorch high-level API.

Note that this toolbox is meant for prototyping and built-in PyTorch optimizers will almost certainly have better performance.

## Included optimizers:
    'grad_exact' : exact gradient optimization
    'conj_grad_fr' : conjugate gradient descent using Fletcher-Reeves search direction
    'conj_grad_pr' : conjugate gradient descent using Polak-Ribiere search direction
    'newton_exact' : exact newton optimization
    'bfgs' : approximate newton optimization using bfgs

## Included line-search methods:
    'back_tracking' : backing tracking based line-search
    'quad_search' : quadratic line-search
    'constant' : no line search, constant step size used

## Setup
```bash
pip install git+https://github.com/coursekevin/mpbatch.git
```
