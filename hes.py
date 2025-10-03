import numpy as np

def fit_heston_variance_moments(ctx, dt=1/252, enforce_feller=True):
    """
    Fit CIR/Heston variance dynamics from one variance series (ctx).
    Model: dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW_t
           => Exact 1-step mean:   E[v_{t+1}|v_t] = theta + (v_t - theta)*phi,  phi=e^{-kappa dt}
              Exact 1-step var:    Var[v_{t+1}|v_t] = c0 + c1*v_t,
                                   c1 = (xi^2 * phi / kappa) * (1 - phi)
                                   c0 = (theta * xi^2 / (2*kappa)) * (1 - phi)**2
    Estimation:
      1) OLS: v_{t+1} = alpha + beta*v_t + u_t  =>  beta = phi, alpha = theta*(1-phi)
         =>  kappa = -ln(beta)/dt,  theta = alpha/(1-beta)
      2) OLS: u_t^2 = gamma0 + gamma1*v_t  (estimates c0, c1)
         =>  xi^2 = gamma1 * kappa / (beta*(1-beta))
             (optionally cross-check via gamma0)
    Returns:
      params: dict(kappa, theta, xi)
      y_hat : one-step mean forecasts E[v_{t+1}|v_t] (length L-1)
    """
    v = np.asarray(ctx, dtype=float).ravel()
    if v.size < 3:
        raise ValueError("Need at least 3 variance observations.")
    vt, vnext = v[:-1], v[1:]
    n = vt.size

    # ---------- Step 1: OLS for conditional mean ----------
    X = np.column_stack([np.ones(n), vt])      # [1, v_t]
    beta_ols = np.linalg.lstsq(X, vnext, rcond=None)[0]
    alpha_hat, beta_hat = float(beta_ols[0]), float(beta_ols[1])

    # Guardrails (beta in (0,1) for mean-reverting CIR with dt>0)
    eps = 1e-10
    beta_hat = np.clip(beta_hat, eps, 1 - 1e-6)

    phi_hat   = beta_hat
    kappa_hat = -np.log(phi_hat) / dt
    theta_hat = alpha_hat / (1.0 - phi_hat)

    # ---------- Step 2: OLS on squared residuals for variance ----------
    u = vnext - (alpha_hat + beta_hat * vt)
    Yu = u**2
    Zu = np.column_stack([np.ones(n), vt])     # [1, v_t]
    gamma = np.linalg.lstsq(Zu, Yu, rcond=None)[0]
    gamma0_hat, gamma1_hat = float(gamma[0]), float(gamma[1])

    # xi^2 from c1 (=gamma1_hat)
    denom = (phi_hat * (1.0 - phi_hat))
    if denom <= 0:
        raise RuntimeError("Degenerate mean estimate produced nonpositive denominator.")
    xi2_from_c1 = gamma1_hat * kappa_hat / denom

    # Optional cross-check using c0
    d = (1.0 - phi_hat)
    xi2_from_c0 = (2.0 * kappa_hat * gamma0_hat) / (theta_hat * d * d) if theta_hat > 0 else np.nan

    # Combine (use c1 as primary; if c0 gives positive also, average them)
    if np.isfinite(xi2_from_c0) and xi2_from_c0 > 0:
        xi2_hat = 0.7 * xi2_from_c1 + 0.3 * xi2_from_c0
    else:
        xi2_hat = xi2_from_c1

    xi2_hat = max(xi2_hat, eps)
    xi_hat  = np.sqrt(xi2_hat)

    # Optional Feller enforcement: 2*kappa*theta >= xi^2
    if enforce_feller and (xi2_hat > 2.0 * kappa_hat * theta_hat):
        xi_hat = np.sqrt(max(2.0 * kappa_hat * theta_hat, eps))

    # ---------- One-step forecasts ----------
    y_hat = theta_hat + (vt - theta_hat) * phi_hat   # = alpha_hat + beta_hat*vt

    params = {"kappa": kappa_hat, "theta": theta_hat, "xi": xi_hat}
    return params, y_hat
