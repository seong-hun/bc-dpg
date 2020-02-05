# Exploration Guided Continuous-Time DPG

## Documents

**MED 2020**:
Continuous-Time Deterministic Policy Gradient-Based Controller
for Morphing Aircraft without Exploration

[Github (private)](https://github.com/seong-hun/med-2020/tree/submitted)


## Usage

To generate 50 sample trajectories, and save to `sample.h5`:
```bash
$ python main.py sample -n 50 -o sample.h5
```

To train 1000 epochs on the sample `data/sample.h5`:
```bash
$ python main.py train -i data/sample.h5 -n 1000
```

To plot using `data/run.h5`:
```bash
$ python main.py plot data/run.h5
```
