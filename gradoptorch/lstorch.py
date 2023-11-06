import torch


def line_search(
    f,
    x_k,
    g_k,
    p_k,
    ls_method="back_tracking",
    ls_params={"alf": 1, "rho": 0.3, "mu": 1e-4, "iter_lim": 1000},
):
    """
    This function performs line search for an objective function "f" using pytorch

    ie. at x_k, find an alf for which x_k + alf*p_k decreases the objective function

    INPUTS:
        f < function > : objective function f(x) -> f
        x_k < tensor > : current best guess for f(x) minimum
        g_k < tensor > : gradient evaluated at x_k
        p_k < tensor > : search direction
        alf < float > : initial step length
        ls_method < str > : indicates which method to use with line search
        ls_params < dict{
            'alf' : initial guess for step length
            'rho' : decay constant for each iteration
            'mu' : constant for "Armijo sufficient decrease condition" (larger mu makes the condition more easily met)
            'iter_lim' : iteration limit
            'alf_lower_coeff' : coefficient for determining point one in quad_search
            'alf_upper_coeff' : coefficient for determining point three in quad_search
        } > : dictionary with parameters to use for line search

    RETURNS:
        alf_new < float > : computed search length
        converge < bool > : bool indicating whether line search converged
        message < string > : string with output from line_search method
    """
    if ls_method == "back_tracking":
        alf_new, converge, message = back_tracking(f, x_k, g_k, p_k, ls_params)

    elif ls_method == "quad_search":
        alf_new, converge, message = quad_search(f, x_k, g_k, p_k, ls_params)

    elif ls_method == "constant":
        alf_new = ls_params["alf"]
        converge = True
        message = "Constant steps size chosen, no backtracking required."

    else:
        raise ValueError("ls_method %s not recognized." % ls_method)
    return alf_new, converge, message


def armijo_suff_decrease(f, x_k, g_k, p_k, alf, mu):
    """
    This function returns whether or not the "Armijo sufficient decrease condition" is satisfied
    for a given function, gradient, search direction, step length, and positive constant

    INPUTS:
        f < function > : objective function f(x) -> f
        x_k < tensor > : current best guess for f(x) minimum
        g_k < tensor > : gradient evaluated at x_k
        p_k < tensor > : search direction
        alf < float > : step length
        mu < float > : small positive constant (included as is popular in practice)

    RETURNS:
        suff_decrease < bool > : bool indicating whether suffiecient decrease was observed
    """
    RHS = f(x_k) + mu * alf * torch.matmul(g_k.t(), p_k)
    LHS = f(x_k + alf * p_k)
    return torch.gt(RHS, LHS)


def back_tracking(f, x_k, g_k, p_k, ls_params):
    """
    This function performs back tracking line search

    INPUTS:
        f < function > : objective function f(x) -> f
        x_k < tensor > : current best guess for f(x) minimum
        g_k < tensor > : gradient evaluated at x_k
        p_k < tensor > : search direction
        alf < float > : initial step length
        ls_params < dict{
            'alf' < float > : initial guess for step-length
            'mu' < float > : small positive constant used in "Armijo suff. decrease condition"
            'rho' < float > : step-size dicount coefficient
            'iter_lim < int > : iteration limit for solver
        } > : dictionary with parameters to use for line search

    RETURNS:
        alf_new < float > : computed search length
        converge < bool > : bool indicating whether line search converged
        message < string > : string with output from back tracking method
    """
    alf = ls_params["alf"]
    rho = ls_params["rho"]
    mu = ls_params["mu"]
    iter_lim = ls_params["iter_lim"]
    alf_new = ls_params["alf"]
    iter = 0
    while not armijo_suff_decrease(f, x_k, g_k, p_k, alf_new, mu) and iter < iter_lim:
        alf_new = rho * alf_new
        iter += 1

    if iter == iter_lim:
        converge = False
        message = "Back-tracking line search iteration limit reached."
    else:
        converge = True
        message = "Back-tracking line search converged."

    return alf_new, converge, message


def quad_search(f, x_k, g_k, p_k, ls_params):
    """
    This function performs approximate quadratic line search

    INPUTS:
        f < function > : objective function f(x) -> f
        x_k < tensor > : current best guess for f(x) minimum
        g_k < tensor > : gradient evaluated at x_k
        p_k < tensor > : search direction
        alf < float > : initial step length
        ls_params < dict{
            'alf' < float > : initial guess for step-length
            'mu' < float > : small positive constant used in "Armijo suff. decrease condition"
            'rho' < float > : step-size dicount coefficient
            'iter_lim < int > : iteration limit for solver
            'alf_lower_coeff' : coefficient for determining point one in quad_search
            'alf_upper_coeff' : coefficient for determining point three in quad_search
        } > : dictionary with parameters to use for line search

    RETURNS:
        alf_new < float > : computed search length
        converge < bool > : bool indicating whether line search converged
        message < string > : string with output from back tracking method
    """
    mu = ls_params["mu"]
    iter_lim = ls_params["iter_lim"]
    alf_new = ls_params["alf"]
    alf_coeff1 = ls_params["alf_lower_coeff"]
    alf_coeff2 = ls_params["alf_upper_coeff"]
    iter = 0
    while not armijo_suff_decrease(f, x_k, g_k, p_k, alf_new, mu) and iter < iter_lim:
        a1 = alf_new
        a2 = alf_new * 0.1
        a3 = alf_new * 2.0
        f1 = f(x_k + a1 * p_k)
        f2 = f(x_k + a2 * p_k)
        f3 = f(x_k + a3 * p_k)

        A = torch.tensor(
            [
                [1 / 2 * (a1**2), a1, 1],
                [1 / 2 * (a2**2), a2, 1],
                [1 / 2 * (a3**2), a3, 1],
            ]
        )
        b = torch.tensor([[f1], [f2], [f3]])

        A_LU, pivots = torch.linalg.lu_factor(A)

        coeff = torch.linalg.lu_solve(A_LU, pivots, b.view(-1, 1)).view(-1)
        alf_new = -coeff[1] / coeff[0]
        iter += 1

    if iter == iter_lim:
        converge = False
        message = "Quadratic approx. line search iteration limit reached."
    else:
        converge = True
        message = "Quadratic approx. line search converged."

    return alf_new, converge, message
