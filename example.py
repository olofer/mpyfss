"""
Collection of examples using the MPYFSS.ESTIMATE(.) function.
"""

import numpy as np
from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import argparse

import mpyfss


def basic_siso_example():
    # Continuous-time system: H(s) = 1 / (s^2 + 2s + 2)
    num = [1]
    den = [1, 2, 2]
    dt = 0.05  # Sampling interval
    N = 500  # batch signal length
    system_dt = signal.cont2discrete((num, den), dt, method="bilinear")
    system_d = signal.dlti(system_dt[0], system_dt[1], dt=system_dt[2])
    t1, y1 = signal.dstep(system_d, n=N)
    t2, y2 = signal.dimpulse(system_d, n=N)
    u3 = np.random.randn(N)
    t3, y3 = signal.dlsim(system_d, u3)
    y1, y2, y3 = np.squeeze(y1), np.squeeze(y2), np.squeeze(y3)
    print(y1.shape, y2.shape, y3.shape)
    assert np.all(t1 == t2) and np.all(t1 == t3)
    t = t1 * dt  # Convert sample indices to time

    y1_noisy = y1 + np.random.randn(*y1.shape) * 0.01
    y2_noisy = y2 + np.random.randn(*y2.shape) * 0.01
    y3_noisy = y3 + np.random.randn(*y3.shape) * 0.01

    plt.figure()
    plt.plot(t, y1_noisy, linewidth=2, label="step: noisy", alpha=0.50)
    plt.plot(t, y2_noisy, linewidth=2, label="impulse: noisy", alpha=0.50)
    plt.plot(t, y3_noisy, linewidth=2, label="dither: noisy", alpha=0.50)
    plt.title("Data batches (outputs)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create 3 batches: step-, impulse-, and dither-responses.
    # Note that if the input signals are not sufficiently "exciting"
    # then the VARX coefficients might not be solvable without regularization.
    def get_batch_(j):
        if j == 0:
            return np.ones((1, N)), y1_noisy.reshape((1, N))
        elif j == 1:
            u = np.zeros((1, N))
            u[0, 0] = 1.0
            return u, y2_noisy.reshape((1, N))
        elif j == 2:
            return u3.reshape((1, N)), y3_noisy.reshape((1, N))
        else:
            assert False, "invalid batch index"

    # Call MPYFSS.ESTIMATE(.)
    sys = mpyfss.estimate(3, get_batch_, 24, 2)
    print(sys.keys())
    print(sys["sv"])  # NOTE: the first two singular values dominate

    dtsys = signal.dlti(sys["A"], sys["B"], sys["C"], sys["D"], dt=dt)

    # Produce a step response with the estimated system!
    t4, y4 = signal.dstep(dtsys, n=N)
    y4 = np.squeeze(y4)
    assert np.all(t1 == t4)

    plt.figure()
    plt.plot(t, y1, linewidth=2, label="step: true system", alpha=0.50)
    plt.plot(
        t,
        y4,
        linewidth=2,
        label="step: estimated system (%i states)" % (sys["A"].shape[0]),
        alpha=0.50,
    )
    plt.title("System step response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def basic_mimo_example(
    M: int = 100, ny: int = 8, nu: int = 2, N: int = 5000, n: int = 20
):
    # Reference: https://github.com/olofer/mpfss/blob/master/test_mpfss_openloop.m
    L = 5 / 11
    c = 400
    tau = L / c
    Ts = (1 / 50) * tau
    r1 = 0.1 * tau
    r2 = 50 / (c * L)

    # quadrature integration accuracy
    qtol = 1e-12

    # Allocate system matrices (continuous time)
    A = np.zeros((2 * M, 2 * M))
    B = np.zeros((2 * M, nu))
    C = np.zeros((ny, 2 * M))

    fu = list()
    for ii in range(nu):
        xplonk = (L / 16) + (L / (nu + 1)) * ii
        assert xplonk < L and xplonk > 0
        fu.append(lambda x: np.exp(-1.0 * ((x - xplonk) ** 2) / ((0.01 * L) ** 2)))

    rho = 1
    xpick = np.arange(1, ny + 1) * rho * L / (ny + 1)
    assert np.all(xpick > 0) and np.all(xpick < L)

    # Populate continuous-time matrices (A, B, C)
    for mm in range(M):
        kk = 2 * mm
        A[kk : (kk + 2), kk : (kk + 2)] = [
            [0, 1],
            [
                -((c * (mm + 1) * np.pi / L) ** 2),
                -(r1 + r2 * ((mm + 1) * np.pi / L) ** 2),
            ],
        ]
        Xm = lambda x: np.sin(x * np.pi * (mm + 1) / L)
        for ii in range(nu):
            integral_value, _ = quad(
                lambda x: fu[ii](x) * Xm(x), 0.0, L, epsabs=qtol, epsrel=qtol
            )
            B[kk : (kk + 2), ii] = [0.0, (2 / L) * integral_value]

        for ii in range(ny):
            C[ii, kk : (kk + 2)] = [Xm(xpick[ii]), 0.0]

    print(A.shape, B.shape, C.shape)

    # Discretize with zero-order-hold, then simulate a batch of length N
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete(
        (A, B, C, np.zeros((ny, nu))), Ts, method="zoh"
    )

    print(Ad.shape, Bd.shape, Cd.shape, Dd.shape)
    assert np.sum(Dd * Dd) == 0

    U = np.random.randn(*(N, nu))
    t, Y, _ = signal.dlsim(
        signal.dlti(Ad, Bd, Cd, Dd, dt=Ts), U, x0=np.zeros(Ad.shape[0])
    )
    print(t.shape, Y.shape)

    stdu = np.sqrt(np.trace(U.T @ U) / (nu * N))
    stdy = np.sqrt(np.trace(Y.T @ Y) / (ny * N))
    print("std(U)=", stdu, "std(Y)=", stdy)
    Y_noisy = Y + 0.05 * stdy * np.random.randn(*(N, ny))

    plt.plot(t / Ts, Y_noisy, linewidth=2, alpha=0.5)
    plt.grid(True)
    plt.xlabel("time-step")
    plt.ylabel("system output (with noise)")
    plt.show()

    def get_batch_(i: int):  # embed scaling of data in the batch fetching function
        assert i == 0
        return (1 / stdu) * U.T, (1 / stdy) * Y_noisy.T

    # Call MPYFSS.ESTIMATE(.)
    sys = mpyfss.estimate(
        1, get_batch_, 30, n, dterm=False, reduction_method="weighted", alpha=1.0e-6
    )
    print(sys.keys())

    # print(sys["sv"])
    # plt.plot(sys["sv"])
    # plt.show()

    # First make a plot of the eigenvalues of the A-matrix
    eigvals_true = np.linalg.eigvals(Ad)
    eigvals_esti = np.linalg.eigvals(sys["A"])

    theta_ = np.linspace(-np.pi, np.pi, 500)

    plt.plot(
        np.real(eigvals_true),
        np.imag(eigvals_true),
        linestyle="none",
        marker="+",
        markersize=8,
        color="black",
        label="True",
    )
    plt.plot(
        np.real(eigvals_esti),
        np.imag(eigvals_esti),
        linestyle="none",
        marker="o",
        markersize=8,
        color="blue",
        fillstyle="none",
        label="Estimated",
        alpha=0.80,
    )
    plt.plot(
        np.cos(theta_),
        np.sin(theta_),
        linestyle="-",
        linewidth=1.0,
        color="red",
        alpha=0.50,
    )
    plt.grid(True)
    plt.xlabel("$Re(\lambda)$")
    plt.ylabel("$Im(\lambda)$")
    plt.title("System matrix (discrete-time) eigenvalues")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    dtsys = signal.dlti(sys["A"], sys["B"], sys["C"], sys["D"], dt=Ts)

    # TODO: make a sigma-plot to visualize the system frequency response match
    # TODO: figure out why the Python results are a little bit different than the Matlab results ?!

    """
    figure;
    sigma(sysdt, 'b-', syses, 'r-', syses2, 'g-');
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="siso-open-loop")
    args = parser.parse_args()

    print(args)

    if args.which.lower() == "siso-open-loop":
        basic_siso_example()
    elif args.which.lower() == "mimo-open-loop":
        basic_mimo_example()
    else:
        raise NotImplementedError

    print("done.")
