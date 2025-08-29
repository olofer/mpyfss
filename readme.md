# mpyfss

Linear time-invariant (LTI) system identification of multiple-input multiple-output (MIMO) systems. Uses vector auto-regression (VARX) combined with weighted model reduction. The signal data can be provided in multiple batches. 

This `mpyfss` repository contains reworked ***selected pieces*** from the `mpfss` repository (https://github.com/olofer/mpfss) which is strictly `matlab`/`octave` and therefore less accessible in many use-cases.  

## Usage

Execute a basic smoke-test like this:
```
python3 mpyfss.py
```
The only external dependence is `numpy`.

For general usage, just `import mpyfss` and define a `get_batch(.)` function to get going (it takes a single index argument and is responsible to return a tuple of I/O signal data `numpy` arrays). This *might* look as follows.

```
import mpyfss

...

def get_my_batch(i:int):
  return data[i]["U"], data[i]["Y"]

lag = 20       # length of VARX window (timesteps)
order = 10     # order of the final state-space model
dterm = False  # should the model have a direct feed-through?

model = mpyfss.estimate(nbatch, get_my_batch, lag, order, dterm)

...
```

See `example.py` for complete examples. The below demonstration also requires `matplotlib` and `scipy` in addition to `numpy`.

```
python3 example.py # default is SISO open-loop
python3 example.py --which mimo-open-loop
python3 example.py --which siso-closed-loop
```

## Custom accumulator usage

For large datasets it can be helpful to use `multiprocessing` to load (or generate) and calculate batch covariances in parallel. 

```
python3 example-multiprocessing.py
```

See source code for details.

## References

See: https://github.com/olofer/mpfss
