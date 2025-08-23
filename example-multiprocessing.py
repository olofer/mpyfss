"""
Demonstration of defining and using a custom covariance accumulator.
This example also illustrates how to loop over a nearly "limitless" set of batches since 
they are generated at fetch-time using the batch index as a seed for the RNG used 
to produce instances of noise.

The data-generating process is a basic closed-loop simulation.
"""

import argparse
import numpy as np
import multiprocessing
import mpyfss

NT: int = 5000


def produce_batch(idx: int):
    # Use idx as a seed value for a local RNG, then generate input data from the RNG and also for output noise..
    U = np.zeros((NT, 2))
    Y = np.zeros((NT, 2))
    return U, Y


def get_batch_covariance(idx: int):
    """
    Runs on a worker process in parallel with other jobs.
    """
    U, Y = produce_batch(idx)
    p = 10
    dterm = False
    Yi, Zi = mpyfss.dvarxdata_transposed_(Y, U, p, dterm)
    Ni = Yi.shape[0]
    return {"ZZ": (Zi.T @ Zi) / Ni, "YZ": (Yi.T @ Zi) / Ni, "N": Ni}


def merge_result(summary: dict, item: dict):
    """
    Runs on the host process; merges results dicts into a single summary.
    Does in-place modification to be more memory efficient.
    """
    if summary["N"] > 0:
        a_: float = summary["N"] / (summary["N"] + item["N"])
        b_: float = item["N"] / (summary["N"] + item["N"])
        summary["ZZ"] *= a_
        summary["ZZ"] += b_ * item["ZZ"]
        summary["YZ"] *= a_
        summary["YZ"] += b_ * item["YZ"]
        summary["N"] += item["N"]
    else:
        print("1st!")
        summary["ZZ"] = item["ZZ"]
        summary["YZ"] = item["YZ"]
        summary["N"] = item["N"]


def custom_accumulator_function(num_batches: int, workers: int, chunksize: int):
    """
    Custum accumulator that can be passed along to MPYFSS.ESTIMATE(.)
    """
    summary = {"ZZ": None, "YZ": None, "N": int(0)}

    job_list = [k for k in range(num_batches)]

    with multiprocessing.Pool(processes=workers) as pool:
        for k, item in enumerate(
            pool.imap_unordered(get_batch_covariance, job_list, chunksize=chunksize)
        ):
            merge_result(summary, item)
            print(k, item["N"])

    return summary["ZZ"], summary["YZ"], summary["N"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=10)
    args = parser.parse_args()

    #
    # TODO: need to figure out how to pass along a custom argument dict to the custom function
    #       since it does not have to be structured the same way as the default accumulator
    #

    ZZ, YZ, N = custom_accumulator_function(args.batches, args.workers, args.chunksize)

    print(ZZ.shape, YZ.shape)
    print("total N:", N)

    assert np.all(np.isfinite(ZZ))
    assert np.all(np.isfinite(YZ))

    print("done.")
