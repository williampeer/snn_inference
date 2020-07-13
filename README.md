# snn_inference

Currently in what I would describe as a development/alpha-version.

## Usage
`python main.py -h`

See Models/ for models that are already implemented and their implementations.

Test/ for tests that you may want to run just to figure out how the Models can be used outside of the experiments context (or what happens under the hood).

exp_suite.py should give you a gist of what running an experiment does

### Sample usage
```
python main.py -s Synthetic -lr 0.001 -ti 40 -N 1 -bs 400 -tvr 10.0 -rpti 4000 -optim Adam -ipr 0.5 -mt LIF -lfn van_rossum_dist -es 1
```

```
python main.py -s DataDriven -lr 0.001 -ti 40 -N 1 -bs 400 -tvr 10.0 -rpti 4000 -optim Adam -ipr 0.5 -mt LIF -lfn van_rossum_dist -ds exp404 -es 1
```

Output of python main.py -h:
```
main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> 
                  -bs <batch-size> -tvr <van-rossum-time-constant> -ic <input-coefficient> 
                  -rpti <rows-per-training-iteration> -optim <optimiser> -ipr <initial-poisson-rate> 
                  -mt <model-type> -lfn <loss-fn> -ds <data-set> -es <evaluate-step> -fmp <fitted-model-path>
```
