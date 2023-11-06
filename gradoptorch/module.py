from typing import Callable, Any

import torch
from torch import nn, Tensor
from jaxtyping import Float

from .gradoptorch import optimizer, OptimLog, default_opt_settings, default_ls_settings


def update_model_params(model: nn.Module, new_params: Tensor) -> None:
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = new_params[pointer : pointer + num_param].view_as(param).data
        pointer += num_param
        param.requires_grad = True


def optimize_module(
    model: nn.Module,
    f: Callable[[nn.Module], Float[Tensor, ""]],
    opt_method: str = "conj_grad_pr",
    opt_params: dict[str, Any] = default_opt_settings,
    ls_method: str = "back_tracking",
    ls_params: dict[str, Any] = default_ls_settings,
) -> OptimLog:
    """
    Optimizes the parameters of a given nn.Module using classical optimizer.

    See the `optimizer` function for more details on the optimization methods
    available.

    INPUTS:
        model < nn.Module > : The torch.nn model to be optimized.
        f < Callable[[nn.Module], Float[Tensor, ""]] > : The loss function to be minimized
        opt_method < str > : The optimization method to be used.
        opt_params < dict[str, Any] > : The parameters to be used for the optimization method.
        ls_method < str > The line search method to be used.
        ls_params < dict[str, Any] > : The parameters to be used for the line search method.

    OptimLog: the log of the optimization process.
    """
    if opt_method == "newton_exact":
        raise NotImplementedError(
            "Exact Newton's method is not implemented for optimize_module."
        )

    # Flatten the model parameters and use them as an initial guess if not provided
    params = torch.cat([param.view(-1) for param in model.parameters()])

    def f_wrapper(params: Float[Tensor, " d"]) -> Float[Tensor, ""]:
        update_model_params(model, params)
        return f(model)

    def grad_wrapper(params: Float[Tensor, " d"]) -> Float[Tensor, " d"]:
        update_model_params(model, params)
        model.zero_grad()
        with torch.enable_grad():
            loss = f(model)
        loss.backward()
        return torch.cat(
            [
                param.grad.view(-1)
                if param.grad is not None
                else torch.zeros_like(param).view(-1)
                for param in model.parameters()
            ]
        )

    final_params, hist = optimizer(
        f=f_wrapper,
        x_guess=params,
        g=grad_wrapper,
        opt_method=opt_method,
        opt_params=opt_params,
        ls_method=ls_method,
        ls_params=ls_params,
    )
    update_model_params(model, final_params)
    return hist
