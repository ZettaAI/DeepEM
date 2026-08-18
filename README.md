# DeepEM
Deep Learning for EM Connectomics

## Isotropic models
Use `deepem/models/v2/` for new isotropic training. Those models build their
core from `deepem/models/core/rsunet.py`, which upsamples trilinearly on every
axis. The older isotropic models (`deepem/models/*_iso*.py`) get their core
from emvision, whose `BilinearUp` builds its kernel from the x/y axes only and
therefore does not interpolate along z. They are kept unchanged so existing
checkpoints keep reproducing.

Existing isotropic checkpoints can be carried over with `--pretrain`: the v2
state dict is the old one minus the `up.up.0.weight` buffers, so every learned
weight transfers.

ONNX export of these models needs `--opset_version 11` or higher (the default);
ONNX `Resize` only gained `coordinate_transformation_mode` in opset 11, and
exporting below that silently produces a graph that does not match PyTorch.

## Citation
[Lee et al. 2017](https://arxiv.org/abs/1706.00120)
```
@article{lee2017superhuman,
  author    = {Kisuk Lee and
               Jonathan Zung and
               Peter Li and
               Viren Jain and
               H. Sebastian Seung},
  title     = {Superhuman Accuracy on the {SNEMI3D} Connectomics Challenge},
  journal   = {arXiv preprint arXiv:1706.00120},
  year      = {2017},
}
```
[Dorkenwald et al. 2019](https://www.biorxiv.org/content/10.1101/2019.12.29.890319v1)
```
@article {Dorkenwald2019.12.29.890319,
	author = {Dorkenwald, Sven and Turner, Nicholas L. and Macrina, Thomas and Lee, Kisuk and Lu, Ran and Wu, Jingpeng and Bodor, Agnes L. and Bleckert, Adam A. and Brittain, Derrick and Kemnitz, Nico and Silversmith, William M. and Ih, Dodam and Zung, Jonathan and Zlateski, Aleksandar and Tartavull, Ignacio and Yu, Szi-Chieh and Popovych, Sergiy and Wong, William and Castro, Manuel and Jordan, Chris S. and Wilson, Alyssa M. and Froudarakis, Emmanouil and Buchanan, JoAnn and Takeno, Marc and Torres, Russel and Mahalingam, Gayathri and Collman, Forrest and Schneider-Mizell, Casey and Bumbarger, Daniel J. and Li, Yang and Becker, Lynne and Suckow, Shelby and Reimer, Jacob and Tolias, Andreas S. and da Costa, Nuno Ma{\c c}arico and Reid, R. Clay and Seung, H. Sebastian},
	title = {Binary and analog variation of synapses between cortical pyramidal neurons},
	elocation-id = {2019.12.29.890319},
	year = {2019},
	doi = {10.1101/2019.12.29.890319},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/12/31/2019.12.29.890319},
	eprint = {https://www.biorxiv.org/content/early/2019/12/31/2019.12.29.890319.full.pdf},
	journal = {bioRxiv}
}
```
