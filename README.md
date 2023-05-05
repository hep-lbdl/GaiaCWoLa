# GaiaCWoLa

### Repository structure 
```sh
python
├── functions.py # misc. functions, including plotting
├── models.py # define NN architecture 
├── run_mock_streams.py # apply CWoLa on 100 simulated streams
└── full_gd1_scan.py # apply CWoLa on all 21 patches covering the GD-1 stream
notebooks
├── example.ipynb # shows how to run CWoLa on a simulated stream and a real patch of GD-1
└── make_plots.ipynb # shows how to replicate each of the figures in the paper
```

### Quickstart 
Make sure you have `conda` installed on your system. 
```sh
conda env create -n gaia -f requirements.yml # if this gives you trouble, try using requirements_no_builds.yml
conda activate gaia
python -m ipykernel install --user --name gaia --display-name "gaia"
jupyter lab
```
Then, navigate to one of the notebooks in the `notebooks` folder (making sure to specify `gaia` as your kernel).

### Further reading: 
- [CWoLa Paper](https://arxiv.org/abs/1708.02949)
- [Via Machinae 1.0](https://arxiv.org/abs/2104.12789)
- [Via Machinae 2.0](https://arxiv.org/abs/2303.01529)
- [Matt's anomaly detection workshop talk](https://indico.desy.de/indico/event/25341/session/0/contribution/15/material/slides/0.pdf)
- [Gaia Dataset Info](https://gea.esac.esa.int/archive/)
