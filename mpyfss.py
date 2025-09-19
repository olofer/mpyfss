"""
Simplified subspace-like multibatch MIMO LTI system identification.
Method based on vector least-squares auto-regression, with model reduction.

QUICK TEST: python3 mpyfss.py
USAGE: just import this file & call mpyfss.estimate(.)

"""

import numpy as np


def dvarxdata_(
    y: np.ndarray,
    u: np.ndarray,
    p: int,
    dterm: bool = False,
    kmin: int = None,
    kmax: int = None,
):
    """
    Create regressor for one contiguous batch of time-series I/O data.
    It is assumed that the time-dimension is across columns (rows are signal channels).
    """
    ny, N = y.shape
    nu = u.shape[0]
    assert u.shape[1] == N
    k1 = p if kmin is None else kmin
    k2 = N if kmax is None else kmax
    Neff = k2 - k1
    assert Neff >= 1, "Lag too large for batch time-series length"
    nz = ny + nu
    Z = np.vstack((u, y))
    assert Z.shape[0] == nz and Z.shape[1] == N
    Y = np.tile(np.nan, (ny, Neff))
    nzp = nz * p

    if dterm:
        Zp = np.tile(np.nan, (nzp + nu, Neff))
        for k in range(k1, k2):
            kk = k - k1
            lags = np.arange(k - 1, k - p - 1, -1)
            Y[:, kk] = y[:, k]
            past = Z[:, lags].T.flatten()
            direct = u[:, k]
            Zp[:, kk] = np.concatenate([past, direct])
    else:
        Zp = np.tile(np.nan, (nzp, Neff))
        for k in range(k1, k2):
            kk = k - k1
            lags = np.arange(k - 1, k - p - 1, -1)
            Y[:, kk] = y[:, k]
            Zp[:, kk] = Z[:, lags].T.flatten()

    assert np.all(np.isfinite(Y))
    assert np.all(np.isfinite(Zp))

    return Y, Zp


def dvarxdata_transposed_(
    y: np.ndarray,
    u: np.ndarray,
    p: int,
    dterm: bool = False,
    kmin: int = None,
    kmax: int = None,
):
    """
    Create regressor for one contiguous batch of time-series I/O data.
    It is assumed that the time-dimension is across rows (columns are signal channels).
    """
    N, ny = y.shape
    nu = u.shape[1]
    assert u.shape[0] == N
    k1 = p if kmin is None else kmin
    k2 = N if kmax is None else kmax
    Neff = k2 - k1
    assert Neff >= 1, "Lag too large for batch time-series length"
    nz = ny + nu
    Z = np.hstack((u, y))
    assert Z.shape[0] == N and Z.shape[1] == nz
    Y = np.tile(np.nan, (Neff, ny))
    nzp = nz * p

    if dterm:
        Zp = np.tile(np.nan, (Neff, nzp + nu))
        for k in range(k1, k2):
            kk = k - k1
            lags = np.arange(k - 1, k - p - 1, -1)
            Y[kk, :] = y[k, :]
            Zp[kk, :] = np.concatenate([Z[lags, :].flatten(), u[k, :]])
    else:
        Zp = np.tile(np.nan, (Neff, nzp))
        for k in range(k1, k2):
            kk = k - k1
            lags = np.arange(k - 1, k - p - 1, -1)
            Y[kk, :] = y[k, :]
            Zp[kk, :] = Z[lags, :].flatten()

    assert np.all(np.isfinite(Y))
    assert np.all(np.isfinite(Zp))

    return Y, Zp


def get_stats(batches: int, get_batch: callable) -> dict:
    """
    Loop across the batches and calculate the per-channel RMSs both for the individual batches
    and the across-batches per-channel RMSs. Additionally return the time-series lengths for each batch.
    This is a sequential-form calculation.
    """

    rmsu = list()
    rmsy = list()
    size = list()

    for b in range(batches):
        u, y = get_batch(b)
        assert u.shape[1] == y.shape[1]
        nb = u.shape[1]
        ssqu = np.sum(u * u, axis=1)
        ssqy = np.sum(y * y, axis=1)
        rmsu.append(np.sqrt(ssqu / nb))
        rmsy.append(np.sqrt(ssqy / nb))
        size.append(nb)

    size = np.array(size)
    rmsu = np.column_stack(rmsu)
    rmsy = np.column_stack(rmsy)

    nu = rmsu.shape[0]
    ny = rmsy.shape[0]

    total_rmsu = np.sqrt(
        np.sum(rmsu * rmsu * np.tile(size.reshape((1, batches)), (nu, 1)), axis=1)
        / np.sum(size)
    )
    total_rmsy = np.sqrt(
        np.sum(rmsy * rmsy * np.tile(size.reshape((1, batches)), (ny, 1)), axis=1)
        / np.sum(size)
    )

    return {
        "size": size,
        "total_size": np.sum(size),
        "rmsu": rmsu,
        "rmsy": rmsy,
        "total_rmsu": total_rmsu,
        "total_rmsy": total_rmsy,
    }


def merge_covariance_(
    acc: dict, ZZ: np.ndarray, YZ: np.ndarray, YY: np.ndarray, N: int
):
    """
    Modifies the elements of "acc" in-place, unless it is the initial assignment.
    """
    if acc["N"] == 0:
        acc["ZZ"] = np.copy(ZZ)
        acc["YZ"] = np.copy(YZ)
        acc["YY"] = np.copy(YY)
        acc["N"] = N
    else:
        n1, n2 = acc["N"], N
        a_, b_ = n1 / (n1 + n2), n2 / (n1 + n2)
        acc["ZZ"] *= a_
        acc["ZZ"] += b_ * ZZ
        acc["YZ"] *= a_
        acc["YZ"] += b_ * YZ
        acc["YY"] *= a_
        acc["YY"] += b_ * YY
        acc["N"] += n2


def default_sequential_accumulator_(
    batches: int,
    get_batch: callable,
    p: int,
    dterm: bool = False,
    transposed_batch: bool = False,
    verbose: bool = False,
):
    STATS = {"ZZ": None, "YZ": None, "YY": None, "N": int(0)}

    for b in range(batches):
        u, y = get_batch(b)

        if transposed_batch:
            Yb, Zb = dvarxdata_transposed_(
                y,
                u,
                p,
                dterm=dterm,
            )
            Nb = Yb.shape[0]
            Yb *= 1.0 / np.sqrt(Nb)
            Zb *= 1.0 / np.sqrt(Nb)
            merge_covariance_(STATS, Zb.T @ Zb, Yb.T @ Zb, Yb.T @ Yb, Nb)
        else:
            Yb, Zb = dvarxdata_(
                y,
                u,
                p,
                dterm=dterm,
            )
            Nb = Yb.shape[1]
            Yb *= 1.0 / np.sqrt(Nb)
            Zb *= 1.0 / np.sqrt(Nb)
            merge_covariance_(STATS, Zb @ Zb.T, Yb @ Zb.T, Yb @ Yb.T, Nb)

        if verbose:
            print("shapes:", b, Yb.shape, Zb.shape)

    return STATS["ZZ"], STATS["YZ"], STATS["YY"], STATS["N"]


def mvarx_(
    batches: int,
    get_batch: callable,
    p: int,
    dterm: bool = False,
    transposed_batch: bool = False,
    verbose: bool = False,
    beta: float = 0.0,
    return_ee: bool = True,
    return_yy: bool = True,
    return_yz: bool = True,
    return_zz: bool = True,
    custom_accumulator: callable = None,
) -> dict:
    """
    Estimate the VARX block coefficients up to lag-order p.
    Direct term is optional (dterm=True, default is False).
    """
    assert batches >= 1, "Must provide at least 1 batch of data"

    if custom_accumulator is None:
        ZZ, YZ, YY, Ntot = default_sequential_accumulator_(
            batches,
            get_batch,
            p,
            dterm=dterm,
            transposed_batch=transposed_batch,
            verbose=verbose,
        )
    else:
        if not get_batch is None:
            print(
                "WARNING: custom_accumulator is given but get_batch is not None -- it will be ignored"
            )
        custom_args = {
            "p": p,
            "dterm": dterm,
            "transposed": transposed_batch,
            "verbose": verbose,
        }
        ZZ, YZ, YY, Ntot = custom_accumulator(batches, custom_args)

    if verbose:
        print("Total regressors:", Ntot)
        print("YY:", YY.shape, "YZ:", YZ.shape, "ZZ:", ZZ.shape)

    beta_scale = np.trace(ZZ) / ZZ.shape[0]
    if beta_scale == 0.0:
        beta_scale = 1.0
        print("WARNING: beta-scale = 0")

    # Estimate the Markov parameters as H = YZ * inv(ZZ).
    # Solve H * ZZ = YZ --> ZZ.T * H.T = YZ.T
    if beta > 0.0:
        # Optional basic Tikhonov regularization
        Ht = np.linalg.solve(ZZ.T + beta * beta_scale * np.eye(ZZ.shape[0]), YZ.T)
    else:
        Ht = np.linalg.solve(ZZ.T, YZ.T)

    # Evaluate the error EE:
    EE = YY - Ht.T @ YZ.T - YZ @ Ht + (Ht.T @ ZZ) @ Ht

    # Extract root-mean-square error and signal levels
    rmse = np.sqrt(np.diag(EE))
    rmsy = np.sqrt(np.diag(YY))

    return {
        "H": Ht.T,
        "dterm": dterm,
        "p": p,
        "Ntot": Ntot,
        "beta": beta,
        "rmse": rmse,  # ratio rmse / rmsy is useful to look at
        "rmsy": rmsy,
        "EE": EE if return_ee else None,
        "YY": YY if return_yy else None,
        "YZ": YZ if return_yz else None,
        "ZZ": ZZ if return_zz else None,
    }


def mfir_(
    H: np.ndarray,
    p: int,
    H0: np.ndarray = None,
    return_a: bool = True,
    return_p: bool = False,
) -> dict:
    """
    Input: markov block parameters [H(1), ..., H(p)] and the optional "direct" block H(0)
    Output: state-space matrices (A,B,C,D) for the implied FIR system, and optionally its controllability Gramian P.
    """
    nout = H.shape[0]
    ninp = H.shape[1] // p
    assert H.shape[1] == ninp * p, "lag p must divide H.cols() exactly"

    nd = p * nout
    A = None if not return_a else np.zeros((nd, nd))
    B = np.zeros((nd, ninp))
    C = np.zeros((nout, nd))
    D = np.zeros((nout, ninp))

    rr = 0
    cc = 0
    for ii in range(p):
        idxr = np.arange(rr, rr + nout)
        idxc = np.arange(cc, cc + ninp)
        B[idxr, :] = H[:, idxc]
        if return_a and ii != p - 1:
            A[np.ix_(idxr, idxr + nout)] = np.eye(nout)
        rr += nout
        cc += ninp

    C[:, :nout] = np.eye(nout)

    if not H0 is None:
        assert H0.shape[0] == nout
        assert H0.shape[1] <= ninp
        D[:, : H0.shape[1]] = H0  # zero-padding of H(0) implied

    P = None
    if return_p:
        # Optional calculation of the Gramian P (Gramian Q is equal to identity)
        P = np.zeros((nd, nd))
        V = np.copy(B)
        for ii in range(p):
            P += V @ V.T
            V = np.vstack([V[np.arange(nout, nd), :], np.zeros((nout, ninp))])

    return {"A": A, "B": B, "C": C, "D": D, "P": P}


def impulse_response_(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, nk: int
) -> tuple:
    """
    General extraction of MIMO impulse response up to nk timesteps.
    Returns H (3D array with dims nk,ny,nu) & H0 (2D array with dims ny,nu).
    """
    ny, nu, nx = C.shape[0], B.shape[1], A.shape[0]
    assert (
        A.shape[1] == nx
        and D.shape[0] == ny
        and D.shape[1] == nu
        and B.shape[0] == nx
        and C.shape[1] == nx
    )

    H = np.tile(np.nan, (nk, ny, nu))
    H0 = np.tile(np.nan, (ny, nu))
    for uu in range(nu):
        u = np.zeros((nu,))
        u[uu] = 1.0
        x = np.zeros((nx,))
        H0[:, uu] = C @ x + D @ u
        x = A @ x + B @ u
        for kk in range(nk):
            H[kk, :, uu] = C @ x
            x = A @ x

    assert np.all(np.isfinite(H))
    assert np.all(np.isfinite(H0))

    return H, H0


def input_to_state_map_(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """
    Create a rectangular matrix that maps an input sequence to the future state.
    """
    ny = B.shape[0] // p
    assert B.shape[0] == p * ny
    nz = B.shape[1]

    Mpf = np.zeros((p * ny, p * nz))
    Mpf[:, :nz] = B
    cc = nz

    for ii in range(1, p):
        if not A is None:
            Mpf[:, cc : (cc + nz)] = A @ Mpf[:, (cc - nz) : cc]
        else:
            Mpf[: ((p - 1) * ny), cc : (cc + nz)] = Mpf[ny:, (cc - nz) : cc]

        cc = cc + nz

    assert cc == p * nz
    return Mpf


def extract_system_(
    T: np.ndarray, Ti: np.ndarray, package: dict, ny: int, nu: int
) -> dict:
    """
    Helper routine that takes the SVD-derived panels T, Ti and the FIR-package and
    carves out/extracts final state-space matrices (A,B,K,C,D) for the order n implied by the panels.
    This might be called several times after a single SVD; to get many candidate systems.
    """
    n = T.shape[1]
    assert Ti.shape[0] == n
    assert package["C"].shape[0] == ny

    # Transform & truncate predictor to n states
    Ak = Ti @ np.vstack([T[ny:, :], np.zeros((ny, n))])
    # Ak = Ti @ package["A"] @ T
    Bk = Ti @ package["B"]
    Ck = package["C"] @ T
    Dk = package["D"]

    K = Bk[:, nu:]
    D = Dk[:, :nu]
    C = Ck
    B = Bk[:, :nu] + K @ Dk[:, :nu]
    A = Ak + K @ Ck

    assert K.shape == (n, ny)
    assert D.shape == (ny, nu)
    assert B.shape == (n, nu)
    assert C.shape == (ny, n)
    assert A.shape == (n, n)

    """
    rep.K = B(:, (nu+1):end);
    rep.D = D(:, 1:nu) * (rmsy / rmsu);
    rep.C = C;
    rep.B = (B(:, 1:nu) + rep.K * D(:, 1:nu)) * (rmsy / rmsu);
    rep.A = A + rep.K * rep.C;
    """

    return {"A": A, "B": B, "K": K, "C": C, "D": D}


def estimate(
    batches: int,
    get_batch: callable,
    p: int,
    n: any,  # single integer, or a list of integers
    dterm: bool = False,
    transposed_batch: bool = False,
    reduction_method: str = "weighted",
    alpha: float = 1.0e-8,
    beta: float = 0.0,
    custom_accumulator: callable = None,
) -> dict:
    """
    System identification of LTI state space models {A,B,K,C,D}. K is the steady-state Kalman gain.
    The dimension of the final state-space can either be provided as a single integer n, or a list of integers.
    This method integrates VARX estimation (lag-order p) with subsequent model reduction.
    The "reduction_method" parameter can be "weighted" (default) or "unweighted".
    Regularization parameter alpha applies to the "weighted" model reduction step.
    Regularization parameter beta applies to the VARX regression step.
    Advanced usage allows specifying a custom covariance accumulator function.
    """
    assert batches >= 1, "Must provide at least 1 batch of data"
    assert alpha >= 0.0, "alpha >= 0 required"
    assert isinstance(n, list) or isinstance(n, int)

    need_zz = reduction_method.lower() == "weighted"

    markov_coefs = mvarx_(
        batches,
        get_batch,
        p,
        dterm=dterm,
        transposed_batch=transposed_batch,
        beta=beta,
        return_zz=need_zz,
        return_yz=False,
        custom_accumulator=custom_accumulator,
    )

    ny = markov_coefs["H"].shape[0]

    if dterm:
        nu = (markov_coefs["H"].shape[1] - ny * p) // (p + 1)
        assert markov_coefs["H"].shape[1] == (ny + nu) * p + nu
        system_package = mfir_(
            markov_coefs["H"][:, : (p * (ny + nu))],
            p,
            H0=markov_coefs["H"][:, (p * (ny + nu)) :],
            return_a=False,
            return_p=False,
        )
    else:
        nu = (markov_coefs["H"].shape[1] // p) - ny
        assert markov_coefs["H"].shape[1] == (ny + nu) * p
        system_package = mfir_(
            markov_coefs["H"], p, H0=None, return_a=False, return_p=False
        )

    if reduction_method.lower() == "unweighted":
        # Should be equivalent to balanced truncation of the mfir_ system
        Mpy = input_to_state_map_(None, system_package["B"], p)
        U_, S_, _ = np.linalg.svd(Mpy, full_matrices=False, compute_uv=True)

    elif reduction_method.lower() == "weighted":
        dim = p * (ny + nu)
        Rzz = markov_coefs["ZZ"] if not dterm else markov_coefs["ZZ"][:dim, :dim]
        assert Rzz.shape == (dim, dim)
        alpha_scale = np.trace(Rzz) / dim
        if alpha_scale == 0.0:
            alpha_scale = 1.0
            print("WARNING: alpha-scale = 0")
        La = np.linalg.cholesky(Rzz + alpha * alpha_scale * np.eye(dim))
        Mpy = input_to_state_map_(None, system_package["B"], p)
        U_, S_, _ = np.linalg.svd(Mpy @ La, full_matrices=False, compute_uv=True)

    else:
        assert False, "Unrecognized reduction_method"

    assert S_.shape == (ny * p,)

    if isinstance(n, list):
        system = list()
        for order in n:
            assert isinstance(order, int)
            assert order <= p * ny
            T = U_[:, :order] @ np.diag(S_[:order])
            Ti = np.diag(1.0 / S_[:order]) @ U_[:, :order].T
            system.append(extract_system_(T, Ti, system_package, ny, nu))

    else:
        assert n <= p * ny
        T = U_[:, :n] @ np.diag(S_[:n])
        Ti = np.diag(1.0 / S_[:n]) @ U_[:, :n].T
        system = extract_system_(T, Ti, system_package, ny, nu)

    def single_or_list_of(sys_, item_):
        return sys_[item_] if isinstance(sys_, dict) else [s_[item_] for s_ in sys_]

    return {
        "mvarx": markov_coefs,
        "mfir": system_package,
        "sv": S_,
        "A": single_or_list_of(system, "A"),
        "B": single_or_list_of(system, "B"),
        "K": single_or_list_of(system, "K"),
        "C": single_or_list_of(system, "C"),
        "D": single_or_list_of(system, "D"),
    }


def unscale_system_(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Su: any,
    Sy: any,
):
    """
    Assuming that the system matrices (A,B,K,C,D) were estimated for the scaled I/O data (Su*u, Sy*y),
    undo the scaling so that the returned system matrices are associated with the original I/O data (u,y).
    Su, Sy can be scalars or vectors (in which case they correspond to diagonal scalings).
    """
    nu, ny = B.shape[1], C.shape[0]

    if isinstance(Su, float):
        Su = np.ones((nu,)) * Su

    if isinstance(Sy, float):
        Sy = np.ones((ny,)) * Sy

    assert len(Su) == nu
    assert len(Sy) == ny

    SU = np.diag(Su)
    SY = np.diag(Sy)
    iSY = np.diag(1.0 / Sy)

    return np.copy(A), B @ SU, K @ SY, iSY @ C, iSY @ D @ SU


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=7)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nu", type=int, default=4)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--nmin", type=int, default=250)
    parser.add_argument("--nmax", type=int, default=500)
    args = parser.parse_args()

    print("*** SANITY CHECK / SMOKE TEST ***")
    print(args)

    assert args.nmax >= args.nmin
    assert args.n <= args.p * args.ny

    # Create a bogus dataset with random numbers -- B batches

    DATA = list()
    for b in range(args.B):
        nb = np.random.randint(args.nmin, args.nmax + 1)
        DATA.append((np.random.randn(args.nu, nb), np.random.randn(args.ny, nb)))

    def GET_BATCH(index: int):
        return DATA[index][0], DATA[index][1]

    # Evaluate the RMSs
    rms = get_stats(args.B, GET_BATCH)

    print("batch sizes:", rms["size"])
    print("overall RMS(u):", rms["total_rmsu"])
    print("overall RMS(y):", rms["total_rmsy"])

    # Verify featurizer variants are exactly consistent
    for b in range(args.B):
        ub_, yb_ = GET_BATCH(b)
        for dterm_ in [False, True]:
            Zb1, Yb1 = dvarxdata_(yb_, ub_, args.p, dterm_)
            Zb2, Yb2 = dvarxdata_transposed_(yb_.T, ub_.T, args.p, dterm_)
            assert np.all(Zb1 == Zb2.T) and np.all(Yb1 == Yb2.T)

    # Estimate VARX coefficients .. expect all coeffs to be ~ zero
    varx_0 = mvarx_(args.B, GET_BATCH, args.p, dterm=False, verbose=True)
    varx_1 = mvarx_(args.B, GET_BATCH, args.p, dterm=True, verbose=True)

    print("H0:", varx_0["H"].shape, np.mean(varx_0["H"]), np.mean(np.abs(varx_0["H"])))
    print("H1:", varx_1["H"].shape, np.mean(varx_1["H"]), np.mean(np.abs(varx_1["H"])))

    # Quick check of mfir_ & its associated input-2-state function:
    mfir_0A = mfir_(varx_0["H"], args.p, H0=None, return_a=True, return_p=True)
    assert np.sum(mfir_0A["D"] ** 2) == 0
    map_A = input_to_state_map_(mfir_0A["A"], mfir_0A["B"], args.p)
    mfir_0B = mfir_(varx_0["H"], args.p, H0=None, return_a=False)
    assert mfir_0B["A"] is None
    map_B = input_to_state_map_(mfir_0B["A"], mfir_0B["B"], args.p)
    assert np.allclose(map_A, map_B, rtol=1.0e-14, atol=1.0e-14)
    assert np.allclose(map_A @ map_A.T, mfir_0A["P"])

    # Explicitly verify the correctness of the mfir_ state space representation
    D_pert = np.random.randn(mfir_0A["C"].shape[0], mfir_0A["B"].shape[1])
    H, H0 = impulse_response_(
        mfir_0A["A"], mfir_0A["B"], mfir_0A["C"], mfir_0A["D"] + D_pert, args.p
    )
    assert np.allclose(D_pert, H0)
    assert np.allclose(
        np.column_stack([H[k, :, :] for k in range(args.p)]), varx_0["H"]
    )

    # Integrated estimation of state-space matrices -- internally uses mvarx_(.) & mfir_(.) & co
    sys_0 = estimate(args.B, GET_BATCH, args.p, args.n, dterm=False)
    sys_1 = estimate(args.B, GET_BATCH, args.p, args.n, dterm=True)

    assert np.sum(sys_0["D"] ** 2) == 0

    sys_0_multiple = estimate(args.B, GET_BATCH, args.p, [args.n, args.n], dterm=False)
    assert len(sys_0_multiple["A"]) == 2
    for k in range(len(sys_0_multiple["A"])):
        for m in ["A", "B", "K", "C", "D"]:
            assert np.all(sys_0[m] == sys_0_multiple[m][k])

    # Check the transposed batch option
    def GET_TRANSPOSED_BATCH(index: int):
        return DATA[index][0].T, DATA[index][1].T

    sys_0_t = estimate(
        args.B, GET_TRANSPOSED_BATCH, args.p, args.n, dterm=False, transposed_batch=True
    )
    sys_1_t = estimate(
        args.B, GET_TRANSPOSED_BATCH, args.p, args.n, dterm=True, transposed_batch=True
    )
    sys_0_u = unscale_system_(
        sys_0["A"], sys_0["B"], sys_0["K"], sys_0["C"], sys_0["D"], 1.0, 1.0
    )

    for i, m in enumerate(["A", "B", "K", "C", "D"]):
        assert np.all(sys_0[m] == sys_0_t[m])
        assert np.all(sys_1[m] == sys_1_t[m])
        assert np.all(sys_0[m] == sys_0_u[i])

    print("*** DONE ***")

# TODO: utilities to run post-processing passes for residual statistics ~ or at least evaluate specific batches
# TODO: QR based VARX solver option --> using a QR merging operation
# TODO: Can I make this work with CUPY or NUMPY equally?
# TODO: option to split up the regressor assembly within-batch with blocking (might be needed for scale)
# TODO: RSVD variant for very large systems
