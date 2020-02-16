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

**Vanila**

1. Train COPDAC without GAN for the sample `data/sample.h5`.
```bash
$ python main.py train --copdac -n 5 data/samples
```

**With GAN and Regulator**

1. Train COPDAC with a GAN target selector and a regulator.
```bash
$ python main.py train --copdac -n 2 --with-gan=data/gan/trained-00100.pth --with-reg data/samples
```

2. Plot a training history
```bash
$ python main.py plot --copdac data/copdac/BaseEnv-COPDAC-gan-reg.h5
```

3. Test and plot the learnt control input.
```bash
$ python main.py run -p data/copdac/BaseEnv-COPDAC-gan-reg.h5
```

## TODO

- Add a (CPU/GPU) multiprocessing function
