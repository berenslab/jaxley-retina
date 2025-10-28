# jaxley-retina
This repository contains a biophysical model of the outer retina using Jaxley as described in our [paper](https://www.biorxiv.org/content/10.1101/2025.10.20.683356v1). Updates in progress 🛠️

## Installation
After cloning the repository, it can be installed via
```sh
pip install jaxley-retina
```

## Data
Phototransduction cascade stimuli and recordings can be downloaded [here](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg). For further information see the related paper [Chen et al. 2024](https://elifesciences.org/articles/93795).

Original glutamate release data can be downloaded [here](https://zenodo.org/records/3760607). This data was further processed and traces were selected for training the ribbon synapse model in the code in this repository.

## License
[MIT License](https://github.com/berenslab/jaxley-retina/blob/main/LICENSE)

## Citation
```
@article{kadhim2025data,
  title={A data and task-constrained mechanistic model of the mouse outer retina shows robustness to contrast variations},
  author={Kadhim, Kyra L and Beck, Jonas and Huang, Ziwei and Macke, Jakob H and Rieke, Fred and Euler, Thomas and Deistler, Michael and Berens, Philipp},
  journal={bioRxiv},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```