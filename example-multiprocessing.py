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
import functools
import mpyfss

NT: int = 5000


def produce_batch(idx: int):
    """
    This is the "data loader" with the responsibility to return I/O batch data.
    It can generate a simulation like in this example, or it could load a file from disk, or arrays from memory.
    """
    rng = np.random.default_rng(seed=idx)
    U = rng.standard_normal((NT, 2))
    Y = rng.standard_normal((NT, 2))
    return U, Y


def get_batch_covariance(idx: int, params: dict):
    """
    Runs on a worker process in parallel with other jobs.
    """
    U, Y = produce_batch(idx)
    assert U.shape[0] == Y.shape[0], "Batch producer gave inconsistent array lengths"
    assert params["transposed"], "standard argument inconsistent with batch producer"
    Yi, Zi = mpyfss.dvarxdata_transposed_(Y, U, params["p"], params["dterm"])
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
        # print("1st!")
        summary["ZZ"] = item["ZZ"]
        summary["YZ"] = item["YZ"]
        summary["N"] = item["N"]


def custom_accumulator_function(
    num_batches: int, standard_args: dict, workers: int, chunksize: int
):
    """
    Custum accumulator that can be made acceptable for MPYFSS.ESTIMATE(.)
    """
    summary = {"ZZ": None, "YZ": None, "N": int(0)}

    job_list = [
        k for k in range(num_batches)
    ]  # note that it is not necessary to realize the range first

    with multiprocessing.Pool(processes=workers) as pool:
        worker_function = functools.partial(get_batch_covariance, params=standard_args)
        for k, item in enumerate(
            pool.imap_unordered(worker_function, job_list, chunksize=chunksize)
        ):
            merge_result(summary, item)
            print(k, item["N"])

    return summary["ZZ"], summary["YZ"], summary["N"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--p", type=int, default=12)
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=10)
    args = parser.parse_args()

    # standard_params = {"p": 10, "dterm": False}

    shippable_custom_accumulator = functools.partial(
        custom_accumulator_function, workers=args.workers, chunksize=args.chunksize
    )

    # ZZ, YZ, N = custom_accumulator_function(args.batches, standard_params, args.workers, args.chunksize)
    # ZZ, YZ, N = shippable_custom_accumulator(args.batches, standard_params)
    # print(ZZ.shape, YZ.shape)
    # print("total N:", N)
    # assert np.all(np.isfinite(ZZ))
    # assert np.all(np.isfinite(YZ))

    SYS = mpyfss.estimate(
        args.batches,
        None,
        args.p,
        args.nx,
        False,
        transposed_batch=True,
        custom_accumulator=shippable_custom_accumulator,
    )

    print("done.")
