"""
WIP WIP WIP
This first example can be a simple 2nd order prototype with output noise

Are these still OK in newest scipy?

dlsim
cont2discrete

"""

import numpy as np
import scipy.signal as scsp
import matplotlib.pyplot as plt
import argparse

import mpyfss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type="str", default="siso-open-loop")
    args = parser.parse_args()

    # Generate data from a true system (single batch), then estimate the system with mpyfss.estimate()
    # Look at the true and estimated system in the frequency domain
    # Finally simulate the estimated system & compare outputs

    # Only do the simulation if the example is open-loop (for now)

    print("done.")
