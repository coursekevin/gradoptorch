import torch
from .lstorch import line_search


def gradoptorch(f, g, x_guess, opt_method='grad_exact', opt_params={'ep_g': 1e-8, 'ep_a': 1e-6, 'ep_r': 1e-2, 'iter_lim': 1000}, ls_method='back_tracking', ls_params={'rho': 0.3, 'mu': 1e-4,  'iter_lim': 1000}):
    """
    This function performs gradient based optimization for an objective function "f" using pytorch.
    Noteably this function allows the user to input a gradient and Hessian function for use in optimization

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_method < string > : optimization method
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'iter_lim' < int > : iteration limit
        } > : dictionary of optimization settings
        ls_method < string > : line-search method
        ls_params < dict{
            'alf' : initial guess for step length
            'rho' : decay constant for each iteration
            'mu' : constant for "Armijo sufficient decrease condition" (larger mu makes the condition more easily met)
            'iter_lim' : iteration limit
            'alf_lower_coeff' : coefficient for determining point one in quad_search (only required for quad_search)
            'alf_upper_coeff' : coefficient for determining point three in quad_search (only required for quad_search)
        } > : dictionary with parameters to use for line search

    OUTPUTS:
        x_opt < tensor > : optimal x
        f_opt < tensor > : f evaluated at the optimal x
        x_hist < list > : value of x over the optimization history
        f_hist < list > : value of f over the optimization history
        converge < bool > : bool indicating whether optimizer converged or not
        message < string > : message indicating why the optimizer ended


    OPTIMIZERS:
        'grad_exact' : exact gradient optimization
        'conj_grad_fr' : conjugate gradient descent using Fletcher-Reeves search direction
        'conj_grad_pr' : conjugate gradient descent using Polak-Ribiere search direction
        'newton_exact' : exact newton optimization
        'bfgs' : approximate newton optimization using bfgs
    """
    if opt_method == "grad_exact":
        # optimizes f using an explicit expression for the gradient
        x_opt, f_opt, x_hist, f_hist, converge, message = grad_exact(
            f, g, x_guess, opt_params, ls_method, ls_params)
    elif opt_method == "conj_grad_fr":
        # optimizes f using Fletcher-Reeves conj. gradient
        x_opt, f_opt, x_hist, f_hist, converge, message = conj_grad_fr(
            f, g, x_guess, opt_params, ls_method, ls_params)
    elif opt_method == "conj_grad_pr":
        # optimizes f using Polak-Ribiere conj. gradient
        x_opt, f_opt, x_hist, f_hist, converge, message = conj_grad_pr(
            f, g, x_guess, opt_params, ls_method, ls_params)
    elif opt_method == "newton_exact":
        # optimizes f using Newton's Method (requires Hess be defined)
        x_opt, f_opt, x_hist, f_hist, converge, message = newton_exact(
            f, g, x_guess, opt_params, ls_method, ls_params)
    elif opt_method == "bfgs":
        # optimizes f using the BFGS method
        x_opt, f_opt, x_hist, f_hist, converge, message = bfgs(
            f, g, x_guess, opt_params, ls_method, ls_params)

    else:
        x_opt = x_guess
        f_opt = None
        x_hist = None
        f_hist = None
        converge = False
        message = "Valid optimization method not chosen"

    return x_opt, f_opt, x_hist, f_hist, converge, message


def grad_exact(f, g, x_guess, opt_params, ls_method, ls_params):
    """
    This function performs gradient descent using the exact gradient as the search direction

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'iter_lim' < int > : iteration limit
        } > : dictionary of optimization settings
        ls_method < str > : indicates which method to use with line search
        ls_params < dict > : dictionary with parameters to use for line search
    """
    ep_g = opt_params['ep_g']
    ep_a = opt_params['ep_a']
    ep_r = opt_params['ep_r']
    iter_lim = opt_params['iter_lim']

    # initializations
    x_k = x_guess
    x_hist = [x_k]
    f_k = f(x_guess)
    f_hist = [f_k]
    k = 0
    conv_count = 0

    # how many iterations for rel. abs. tolerance met before stopping
    conv_count_max = 2

    while k < iter_lim:
        k += 1

        # compute gradient
        g_k = g(x_k)

        # check for gradient convergence
        if torch.norm(g_k) <= ep_g:
            converge = True
            message = "Exact gradient converged due to grad. tolerance."
            break

        # set search direction to gradient
        p_k = -g_k

        # perform line search
        alf, ls_converge, ls_message = line_search(f, x_k, g_k, p_k,
                                                   ls_method=ls_method, ls_params=ls_params)
        if not ls_converge:
            converge = ls_converge
            message = ls_message
            break

        # compute x_(k+1)
        x_k1, f_k1 = search_step(f, x_k, alf, p_k)

        # check relative and absolute convergence criteria
        if rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
            conv_count += 1

        x_k = x_k1
        f_k = f_k1

        x_hist.append(x_k)
        f_hist.append(f_k)

        if conv_count >= conv_count_max:
            converge = True
            message = "Exact gradient converged due to abs. rel. tolerance."
            break

    if k == iter_lim:
        converge = False
        message = "Exact gradient iteration limit reached."

    return x_k, f_k, x_hist, f_hist, converge, message


def conj_grad_fr(f, g, x_guess, opt_params, ls_method, ls_params):
    """
    This function performs conjugate descent using the Fletcher-Reeves gradient as the search direction

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'iter_lim' < int > : iteration limit
            'restart_iter' < int > : how often to "restart" the search direction with g(x_k) for numerical stability
        } > : dictionary of optimization settings
        ls_method < str > : indicates which method to use with line search
        ls_params < dict > : dictionary with parameters to use for line search
    """
    ep_g = opt_params['ep_g']
    ep_a = opt_params['ep_a']
    ep_r = opt_params['ep_r']
    iter_lim = opt_params['iter_lim']

    # initializations
    x_k = x_guess
    x_hist = [x_k]
    f_k = f(x_guess)
    f_hist = [f_k]
    k = 0
    conv_count = 0

    beta_k = 0
    g_k = g(x_k)
    g_k_norm = torch.norm(g_k)
    p_k = -g_k / g_k_norm

    # set restart count
    restart_count = 0
    restart_iter = opt_params['restart_iter']

    # how many iterations for rel. abs. tolerance met before stopping
    conv_count_max = 2

    while k < iter_lim:
        k += 1

        # check for gradient convergence
        if g_k_norm <= ep_g:
            converge = True
            message = "FR conjugate gradient converged due to grad. tolerance."
            break

        # perform line search
        alf, ls_converge, ls_message = line_search(f, x_k, g_k, p_k,
                                                   ls_method=ls_method, ls_params=ls_params)
        if not ls_converge:
            converge = ls_converge
            message = ls_message
            break

        # compute x_(k+1)
        x_k1, f_k1 = search_step(f, x_k, alf, p_k)

        # check relative and absolute convergence criteria
        if rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
            conv_count += 1

        x_k = x_k1
        f_k = f_k1

        f_hist.append(f_k)
        x_hist.append(x_k)

        if conv_count >= conv_count_max:
            converge = True
            message = "FR conjugate gradient converged due to abs. rel. tolerance."
            break

        # Compute next search direction (Note updates are here as for k == 0 grad is unique)
        g_k1 = g(x_k)

        # compute beta_k
        beta_k = fletcher_reeves(g_k1, g_k)

        # update g_k
        g_k = g_k1
        g_k_norm = torch.norm(g_k)

        # set search direction, every restart_iter iterations the gradients are reset for numerical stability
        if restart_iter == restart_count:
            restart_count = 0
            beta_k = 0
            p_k = -g_k/g_k_norm
        else:
            restart_count += 1
            p_k = -g_k + beta_k*p_k

    if k == iter_lim:
        converge = False
        message = "FR conjugate gradient iteration limit reached."

    return x_k, f_k, x_hist, f_hist, converge, message


def conj_grad_pr(f, g, x_guess, opt_params, ls_method, ls_params):
    """
    This function performs conjugate descent using the Polak-Ribiere gradient as the search direction

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'iter_lim' < int > : iteration limit
            'restart_iter' < int > : how often to "restart" the search direction with g(x_k) for numerical stability
        } > : dictionary of optimization settings
        ls_method < str > : indicates which method to use with line search
        ls_params < dict > : dictionary with parameters to use for line search
    """
    ep_g = opt_params['ep_g']
    ep_a = opt_params['ep_a']
    ep_r = opt_params['ep_r']
    iter_lim = opt_params['iter_lim']

    # initializations
    x_k = x_guess
    x_hist = [x_k]
    f_k = f(x_guess)
    f_hist = [f_k]
    k = 0
    conv_count = 0

    beta_k = 0
    g_k = g(x_k)
    g_k_norm = torch.norm(g_k)
    p_k = -g_k / g_k_norm

    # set restart count
    restart_count = 0
    restart_iter = opt_params['restart_iter']

    # how many iterations for rel. abs. tolerance met before stopping
    conv_count_max = 2

    while k < iter_lim:
        k += 1

        # check for gradient convergence
        if g_k_norm <= ep_g:
            converge = True
            message = "PR conjugate gradient converged due to grad. tolerance."
            break

        # perform line search
        alf, ls_converge, ls_message = line_search(f, x_k, g_k, p_k,
                                                   ls_method=ls_method, ls_params=ls_params)
        if not ls_converge:
            converge = ls_converge
            message = ls_message
            break

        # compute x_(k+1)
        x_k1, f_k1 = search_step(f, x_k, alf, p_k)

        # check relative and absolute convergence criteria
        if rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
            conv_count += 1

        x_k = x_k1
        f_k = f_k1

        f_hist.append(f_k)
        x_hist.append(x_k)

        if conv_count >= conv_count_max:
            converge = True
            message = "PR conjugate gradient converged due to abs. rel. tolerance."
            break

        # Compute next search direction (Note updates are here as for k == 0 grad is unique)
        g_k1 = g(x_k)

        # compute beta_k
        beta_k = polak_ribiere(g_k1, g_k)

        # update g_k
        g_k = g_k1
        g_k_norm = torch.norm(g_k)

        # set search direction, every restart_iter iterations the gradients are reset for numerical stability
        if restart_iter == restart_count:
            restart_count = 0
            beta_k = 0
            p_k = -g_k/g_k_norm
        else:
            restart_count += 1
            p_k = -g_k + beta_k*p_k

    if k == iter_lim:
        converge = False
        message = "PR conjugate gradient iteration limit reached."

    return x_k, f_k, x_hist, f_hist, converge, message


def newton_exact(f, g, x_guess, opt_params, ls_method, ls_params):
    """
    This function performs gradient descent using newton's method as the search direction

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'Hessian' < function > : function that returns the Hessian
            'iter_lim' < int > : iteration limit
        } > : dictionary of optimization settings
        ls_method < str > : indicates which method to use with line search
        ls_params < dict > : dictionary with parameters to use for line search
    """
    ep_g = opt_params['ep_g']
    ep_a = opt_params['ep_a']
    ep_r = opt_params['ep_r']
    H = opt_params['Hessian']
    iter_lim = opt_params['iter_lim']

    # initializations
    x_k = x_guess
    x_hist = [x_k]
    f_k = f(x_guess)
    f_hist = [f_k]
    k = 0
    conv_count = 0

    # how many iterations for rel. abs. tolerance met before stopping
    conv_count_max = 2

    while k < iter_lim:
        k += 1

        # compute gradient
        g_k = g(x_k)

        # check for gradient convergence
        if torch.norm(g_k) <= ep_g:
            converge = True
            message = "Exact Newton converged due to grad. tolerance."
            break

        # invert Hessian and find search direction
        H_k = H(x_k)
        H_LU, pivots, infos = torch.lu(H_k.reshape(
            [1, H_k.shape[0], -1]), get_infos=True)

        if infos.nonzero().size(0) != 0:
            # check if LU factorization failed
            converge = False
            message = "Hessian LU factorization failed."
            break

        # LU solve is designed for batch operations, hence the [0]
        delta_k = torch.lu_solve(-g_k.unsqueeze(0), H_LU, pivots)[0]

        if torch.matmul(delta_k.t(), g_k) < 0:
            p_k = delta_k
        else:
            p_k = -delta_k

        # perform line search
        alf, ls_converge, ls_message = line_search(f, x_k, g_k, p_k,
                                                   ls_method=ls_method, ls_params=ls_params)
        if not ls_converge:
            converge = ls_converge
            message = ls_message
            break

        # compute x_(k+1)
        x_k1, f_k1 = search_step(f, x_k, alf, p_k)

        # check relative and absolute convergence criteria
        if rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
            conv_count += 1

        x_k = x_k1
        f_k = f_k1

        x_hist.append(x_k)
        f_hist.append(f_k)

        if conv_count >= conv_count_max:
            converge = True
            message = "Exact Newton converged due to abs. rel. tolerance."
            break

    if k == iter_lim:
        converge = False
        message = "Exact Newton iteration limit reached."

    return x_k, f_k, x_hist, f_hist, converge, message


def bfgs(f, g, x_guess, opt_params, ls_method, ls_params):
    """
    This function performs Quasi-Newton gradient descent using the Broyden-Fletcher-Goldfarb-Shanno 
    approximation for the Hessian

    INPUTS:
        f < function > : objective function f(x) -> f
        g < function > : gradient function g(x) -> g
        x_guess < tensor > : initial x
        opt_params < dict{
            'ep_g' < float > : conv. tolerance on gradient
            'ep_a' < float > : absolute tolerance
            'ep_r' < float > : relative tolerance
            'iter_lim' < int > : iteration limit
        } > : dictionary of optimization settings
        ls_method < str > : indicates which method to use with line search
        ls_params < dict > : dictionary with parameters to use for line search
    """
    ep_g = opt_params['ep_g']
    ep_a = opt_params['ep_a']
    ep_r = opt_params['ep_r']
    iter_lim = opt_params['iter_lim']

    # initializations
    x_k = x_guess
    x_hist = [x_k]
    f_k = f(x_guess)
    f_hist = [f_k]
    g_k = g(x_k)

    I = torch.eye(x_k.shape[0])
    B_k_inv = I
    k = 0

    conv_count = 0

    # how many iterations for rel. abs. tolerance met before stopping
    conv_count_max = 2

    while k < iter_lim:
        k += 1

        # check for gradient convergence
        if torch.norm(g_k) <= ep_g:
            converge = True
            message = "BFGS converged due to grad. tolerance."
            break

        # set search direction to gradient
        p_k = torch.matmul(B_k_inv, -g_k)

        # perform line search
        alf, ls_converge, ls_message = line_search(f, x_k, g_k, p_k,
                                                   ls_method=ls_method, ls_params=ls_params)
        if not ls_converge:
            converge = ls_converge
            message = ls_message
            break

        # compute x_(k+1)
        x_k1, f_k1 = search_step(f, x_k, alf, p_k)

        # check relative and absolute convergence criteria
        if rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
            conv_count += 1

        x_k = x_k1
        f_k = f_k1

        x_hist.append(x_k)
        f_hist.append(f_k)

        if conv_count >= conv_count_max:
            converge = True
            message = "BFGS converged due to abs. rel. tolerance."
            break

        # update inv. hessian approx.
        # compute gradient
        g_k1 = g(x_k)

        s_k = alf*p_k
        y_k = g_k1 - g_k

        tmp_mat = I - torch.matmul(s_k, y_k.t())/torch.matmul(s_k.t(), y_k)
        B_k_inv = torch.matmul(torch.matmul(tmp_mat, B_k_inv), tmp_mat) + \
            torch.matmul(s_k, s_k.t())/torch.matmul(s_k.t(), y_k)

        g_k = g_k1

    if k == iter_lim:
        converge = False
        message = "BFGS iteration limit reached."

    return x_k, f_k, x_hist, f_hist, converge, message


def search_step(f, x_k, alf, p_k):
    """
    This function performs an optimization step given a step length and step direction

    INPUTS:
        f < function > : objective function f(x) -> f
        x_k < tensor > : current best guess for f(x) minimum
        alf < float > : step length
        p_k < tensor > : step direction

    OUTPUTS:
        x_(k+1) < tensor > : new best guess for f(x) minimum
        f_(k+1) < tensor > : function evaluated at new best guess
    """
    x_k1 = x_k + alf*p_k
    f_k1 = f(x_k1)
    return x_k1, f_k1


def rel_abs_convergence(f_k, f_k1, ep_a, ep_r):
    """
    This function checks for relative/absolute convergence
    ie. if |f(x_(k+1)) - f(x_k)| <= ep_a + ep_r * |f(x_k)|

    INPUTS:
        f_k < tensor > : function evaluated at x_k
        f_k1 < tensor > : function evaluated at x_(k+1)
        ep_a < float > : absolute tolerance
        ep_r < float > : relative tolerance

    OUTPUTS:
        rel/abs convergence < bool > : bool indicating whether rel/abs criterion was met
    """
    LHS = torch.abs(f_k1 - f_k)
    RHS = ep_a + ep_r*torch.abs(f_k)

    return torch.gt(RHS, LHS)


def fletcher_reeves(g_k1, g_k):
    """
    This function computes the Fletcher-Reeves conjugate gradient search direction constant

    INPUTS:
        g_k1 < tensor > : gradient at current time step
        g_k < tensor > : gradient at previous time step

    OUTPUTS:
        b_k1 < tensor > : conj. gradient update constant at current time step
    """
    return torch.matmul(g_k1.t(), g_k1) / torch.matmul(g_k.t(), g_k)


def polak_ribiere(g_k1, g_k):
    """
    This function computes the Polak-Ribier conjugate gradient search direction constant

    INPUTS:
        g_k1 < tensor > : gradient at current time step
        g_k < tensor > : gradient at previous time step

    OUTPUTS:
        b_k1 < tensor > : conj. gradient update constant at current time step
    """
    return torch.matmul(g_k1.t(), (g_k1 - g_k)) / torch.matmul(g_k.t(), g_k)
