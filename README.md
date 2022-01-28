# GaiaCWoLa

### Repository structure 
```sh
python
├── functions.py # misc. functions, including plotting
├── models.py # NN architecture 
├── train.py # submit a training via the command line
└── replicate_via_machinae.py # recreate full Via Machinae-style plot
notebooks
├── train.ipynb # generic training notebook, for testing/prototyping
├── Example_Finding_GD1.ipynb # shows how to find GD1
├── Example_Finding_GD1_Tail.ipynb # shows how to find the GD1 tail
├── Example_Finding_Mock_Stream.ipynb # shows how to find a mock stream
├── mislabeled_stream_stars.ipynb # shows how to find additional GD1 stream stars that should have been labeled
└── all_gd1.ipynb # Work-in-progress, for iterating towards Via Machine-style plot
```

### Quickstart 
Make sure you have `conda` installed on your system. 
```sh
conda env create -n gaia -f requirements.yml # if this gives you trouble, try using requirements_no_builds.yml
conda activate gaia
python -m ipykernel install --user --name gaia --display-name "gaia"
jupyter lab
```
Then, navigate to one of the example notebooks in the `notebooks` folder (making sure to specify `gaia` as your kernel).

Submit a training script: 
```sh
python python/train.py gd1 # other options: gd1_tail, mock
```

### Further reading: 
- [CWoLa Paper](https://arxiv.org/abs/1708.02949)
- [Via Machinae Paper](https://arxiv.org/abs/2104.12789)
- [Matt's anomaly detection workshop talk](https://indico.desy.de/indico/event/25341/session/0/contribution/15/material/slides/0.pdf)
- [Gaia Dataset Info](https://gea.esac.esa.int/archive/)
