To generate 50 sample trajectories, and save to `sample.h5`:
```bash
$ python main.py sample -n 50 -o sample.h5
```

To train 1000 epochs on the sample `data/sample.ht`:
```bash
$ python main.py train -i data/sample.h5 -n 1000
```
