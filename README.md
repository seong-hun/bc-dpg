# Exploration Guided Continuous-Time DPG

## Documents

**MED 2020**:
Continuous-Time Deterministic Policy Gradient-Based Controller
for Morphing Aircraft without Exploration

[Github (private)](https://github.com/seong-hun/med-2020/tree/submitted)

**Lab Seminar**:
February 10, 2020:

[PDF (priviate)](https://drive.google.com/file/d/1f1zddPFNfISP1pHkUacKMTVpD3YiormL/view?usp=sharing)


## Usage

### Data Generation

1. Generate 50 sample trajectories, and save the results into `data/samples`.
```bash
$ python main.py sample
```

### Train GAN

1. Train a GAN for the samples `data/samples/*.h5`.
```bash
$ python main.py train --gan data/samples
```

2. (Optional) Test the learnt generator stored in `data/gan/trained-00100.pth`.
```bash
$ python main.py test --gan data/gan/trained-00100.pth
```
The test results for each generator will be save in `data/tmp.h5`.
Then see the result of the GAN.
```bash
$ python main.py plot --gan data/tmp.h5
```

### COPDAC

COPDAC Training can be done with two options, with and without GAN.

**With GAN**

1. Train COPDAC with a GAN target selector.
```bash
$ python main.py train --copdac --with-gan=data/gan/trained-00100.pth data/samples
```

**Without GAN**

1. Train COPDAC without GAN for the sample `data/sample.h5`.
```bash
$ python main.py train --copdac data/samples
```

To plot using `data/run.h5`:
```bash
$ python main.py plot data/run.h5
```

## TODO

- Add click options at the top of the main.
- Add a (CPU/GPU) multiprocessing function
- Auto subplot
