<div align="center">

# GradOpTorch

#### Classical gradient based optimization in PyTorch.

</div>

- [Installation](#installation)
- [Usage](#usage)

## What is GradOpTorch?

GradOpTorch is a suite of classical gradient-based optimization tools for
PyTorch. The toolkit includes conjugate gradients, BFGS, and some
methods for line-search.

## Why not [torch.optim](https://pytorch.org/docs/stable/optim.html)?

Not every problem is high-dimensional, nonlinear, with noisy gradients.  
For such problems, classical optimization techniques
can be more efficient.

## Installation

GradOpTorch can be installed from PyPI:

```bash
pip install gradoptorch
```

## Usage

There are two primary interfaces for making use of the library.

1. The standard PyTorch object oriented interface:

```python
from gradoptorch import optimize_module
from torch import nn

class MyModel(nn.Module):
    ...

model = MyModule()

def loss_fn(model):
    ...

hist = optimize_module(model, loss_fn, opt_method="bfgs", ls_method="back_tracking")
```

2. The functional interface:

```python
from gradoptorch import optimizer

def f(x):
    ...

x_guess = ...

x_opt, hist = optimizer(f, x_guess, opt_method="conj_grad_pr", ls_method="quad_search")
```

Newton's method is only available in the functional interface

### Included optimizers:

    'grad_exact' : exact gradient optimization
    'conj_grad_fr' : conjugate gradient descent using Fletcher-Reeves search direction
    'conj_grad_pr' : conjugate gradient descent using Polak-Ribiere search direction
    'newton_exact' : exact newton optimization
    'bfgs' : approximate newton optimization using bfgs

### Included line-search methods:

    'back_tracking' : backing tracking based line-search
    'quad_search' : quadratic line-search
    'constant' : no line search, constant step size used
