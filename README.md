# Matching the Prior and Level Spacing of Inverse Heat Dissipation Models to Brain MRI

This fork of the IHDM code asks whether the settings that Rissanen et al. chose for natural images
suit brain MRI. Two settings of IHDM depend on the data: the terminal blur of the prior, which
decides how much of a training image each sample inherits, and the spacing of the blur levels,
which decides how the training effort is spread over frequency. We set both from the statistics
of registered T1 slices (IXI) and measure what the change gains in fidelity and what it costs in
diversity and copying, on IXI and, without refitting, on OASIS-1. The `ihdm/` package holds our
code; the released model code is unchanged in `model_code/`.

Four configurations cross the two changes:

| Configuration | Terminal blur | Level spacing |
|---|---|---|
| A0, *default* (the paper's) | $W/2$ = 96 px | logarithmic |
| A1, *+prior* | $W/8$ = 24 px | logarithmic |
| A2, *+spacing* | $W/2$ = 96 px | matched to IXI |
| A3, *matched* | $W/8$ = 24 px | matched to IXI |

- **Tutorial:** [`ihdm_mri_tutorial.ipynb`](ihdm_mri_tutorial.ipynb) walks through both changes
  with figures computed from our code, then re-plots our results; a rendered copy is
  [`docs/tutorial/ihdm_mri_tutorial.html`](docs/tutorial/ihdm_mri_tutorial.html). Its source is
  [`ihdm_mri_tutorial.py`](ihdm_mri_tutorial.py) (jupytext), and its inputs are in
  [`assets/`](assets/README.md).
- **Results:** [`docs/RESULTS/results_discussion.md`](docs/RESULTS/results_discussion.md) and the
  paper figures with their numbers in [`docs/RESULTS/paper/`](docs/RESULTS/paper/).
- **Reproduction:** the commands are in
  [Section 13 of `results_discussion.md`](docs/RESULTS/results_discussion.md#13-reproduction) and
  at the end of each figure's notes in `docs/RESULTS/paper/` (for example `F2.md`).

Set up the environment and run the tutorial on a CPU (it reads only tracked files):

```bash
conda env create -f environment.yml && conda activate ihdm
# or, for an existing env: conda env update -n ihdm -f environment.yml --prune
pip install -e ".[dev]"
jupyter nbconvert --to notebook --execute --inplace ihdm_mri_tutorial.ipynb
```

The datasets (IXI, OASIS-1, LSUN), the training checkpoints and the evaluation archives are not in
the repository. The code finds them through `IHDM_DATA_ROOT` and the other variables of
`ihdm/paths.py`; only small derived files (spectra, frozen schedules, result tables) are tracked.

## Upstream README (Rissanen et al.)

# Generative Modelling With Inverse Heat Dissipation
 
This repository is the official implementation of the methods in the publication:

* Severi Rissanen, Markus Heinonen, and Arno Solin (2023). **Generative Modelling With Inverse Heat Dissipation**. In *International Conference on Learning Representations (ICLR)*. [[arXiv]](https://arxiv.org/abs/2206.13397) [[project page]](https://aaltoml.github.io/generative-inverse-heat-dissipation)

## Arrangement of code

The "`configs`" folder contains the configuration details on different experiments and the "`data`" folder contains the data. MNIST and CIFAR-10 should run as-is with automatic torchvision data loading, but the other experiments require downloading the data to the corresponding `data/` folders. The "`model_code`" contains the U-Net definition and utilities for working with the proposed inverse heat dissipation model. "`scripts`" contains additional code, for i/o, data loading, loss calculation and sampling. "`runs`" is where the results get saved at.

## Used Python packages

The file "requirements.txt" contains the Python packages necessary to run the code, and they can be installed by running

```pip install -r requirements.txt```

If you have issues with installing the `mpi4py` through pip, you can also install it using conda with `conda install -c conda-forge mpi4py`. 

## Training

You can get started by running an MNIST training script with

```python train.py --config configs/mnist/default_mnist_configs.py --workdir runs/mnist/default```

This creates a folder "`runs/mnist/default`", which contains the folder "`checkpoint-meta`", where the newest checkpoint is saved periodically. "`samples`" folder contains samples saved during training. You can change the frequency of checkpointing and sampling with the command line flags "`training.snapshot_freq_for_preemption=?`" and "`config.training.sampling_freq=?`". 

## Sampling
Once you have at least one checkpoint, you can do sampling with "`sample.py`", with different configurations:

### Random samples
Random samples: 

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
```

### Share the initial state
Samples where the prior state u_K is fixed, but the sampling noise is different:

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --same_init
```

### Share the noise
Samples where the prior state u_K changes, but the sampling noises are shared (results in similar overall image characteristics, but different average colours if the maximum blur is large enough):

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --share_noise
 ```

### Interpolation
Produces an interpolation between two random points generated by the model. 

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --interpolate --num_points=20
```

## Evaluation

The script "`evaluation.py`" contains code for evaluating the model with FID-scores and NLL (ELBO) values. For example, if you have a trained cifar-10 model trained with `configs/cifar10/default_cifar10_configs.py` and the result is in the folder `runs/cifar10/default/checkpoints-meta`, you can run the following (checkpoint=0 refers to the last checkpoint, other checkpoints in `runs/cifar10/default/checkpoints` are numbered as 1,2,3,...):

### FID scores
This assumes that you have `clean-fid` installed. 

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py
            --workdir runs/cifar10/default --checkpoint 0
            --dataset_name=cifar10
            --experiment_name=experiment1 --param_name=default --mode=fid
            --delta=0.013 --dataset_name_cleanfid=cifar10
            --dataset_split=train --batch_size=128 --num_gen=50000
```

### NLL values
The result contains a breakdown of the different terms in the NLL.

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py
            --workdir runs/cifar10/default --checkpoint 0
            --dataset_name=cifar10
            --experiment_name=experiment1 --param_name=default --mode=elbo
            --delta=0.013
```

### Result folder
The results will be saved in the folder `runs/cifar10/evaluation_results/experiment1/` in log files, where you can read them out. The idea in general is that `experiment_name` is an upper-level name for a suite of experiments that you might want to have (e.g., FID w.r.t. different delta), and `param_name` is the name of the calculated value within that experiment (e.g., "delta0.013" or "delta0.012"). 

## Citation

If you use the code in this repository for your research, please cite the paper as follows:

```bibtex
@inproceedings{rissanen2023generative,
  title={Generative modelling with inverse heat dissipation},
  author={Severi Rissanen and Markus Heinonen and Arno Solin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

This software is provided under the [MIT license](LICENSE).

## Spectral-allocation experiment (UAM GenAI course, 2026)

This fork also hosts a course-project experiment on whether the terminal blur and the blur
schedule should depend on the spectrum of the data (brain MRI vs. photographs). The
specification, milestones and contracts live under `docs/README.md`; start there.

Environment: `conda env create -f environment.yml` (env name `ihdm`), then `pip install -e .`.
Run the test suite with `pytest -q` (see `pyproject.toml` for markers and paths).

