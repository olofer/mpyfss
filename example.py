"""
Collection of examples using the MPYFSS.ESTIMATE(.) function.
"""

import numpy as np
from scipy import signal
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="siso-open-loop")
    args = parser.parse_args()

    print(args)

    if args.which.lower() == "siso-open-loop":
        basic_siso_example()
    else:
        raise NotImplementedError

    print("done.")
