import torch
from torch import nn
from gradoptorch import optimize_module

import matplotlib.pyplot as plt  # type: ignore

dim = 2

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class SomeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(-0.5))
        self.y = nn.Parameter(torch.tensor(3.0))


def loss_fn(model):
    a = 1.0
    b = 100.0
    return (a - model.x).pow(2) + b * (model.y - model.x.pow(2)).pow(2)


def main():
    # ---------------------------------------------------------------------------------
    # optimizing using all of the different methods
    opt_method = ["conj_grad_pr", "conj_grad_fr", "grad_exact", "bfgs"]
    histories = []
    for method in opt_method:
        # using quadratic line search
        model = SomeModule()
        hist = optimize_module(
            model, loss_fn, opt_method=method, ls_method="quad_search"
        )
        for n, p in model.named_parameters():
            print(n, p)
        histories.append(hist)

    # ---------------------------------------------------------------------------------
    # making some plots
    _, axs = plt.subplots(1, 2)

    ax1 = axs[0]
    ax2 = axs[1]

    for hist in histories:
        ax1.plot(torch.tensor(hist.f_hist).log10().detach())
        x_hist = torch.stack(hist.x_hist, dim=0).detach()
        ax2.plot(x_hist[:, 0], x_hist[:, 1], "x-")
    ax1.legend(opt_method)

    X, Y = torch.meshgrid(
        torch.linspace(-2.5, 2.5, 100), torch.linspace(-1.5, 3.5, 100), indexing="ij"
    )

    # Define test objective function
    def f(x):
        a = 1.0
        b = 100.0
        return (a - x[0]).pow(2) + b * (x[1] - x[0].pow(2)).pow(2)

    ax2.contourf(X, Y, f(torch.stack([X, Y], dim=0)), 50)

    plt.show()


if __name__ == "__main__":
    main()
