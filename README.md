# UKANWeed

Code for the paper "UKANWeed: An application of Kolmogorov-Arnold Networks to Weed Mapping"

## Installation

You need [UV](https://docs.astral.sh/uv/getting-started/) to run this code.

You can install the venv with:

```bash
    uv sync
    source .venv/bin/activate
```

## Usage
You can run the training/evaluation with:

```bash
    python Seg_UKAN/train.py --cfg parameters/UKAN.yaml
```

All scripts needed to reproduce the results of the paper are in the `scripts.sh` file.

## Citation
If you use this code, please cite the paper:
```bibtex
@inproceedings{UKANWeed,
  title={UKANWeed: An application of Kolmogorov-Arnold Networks to Weed Mapping},
  author={Pasquale De Marinis, Elena Tavoletti, Gennaro Vessio, Giovanna Castellano},
  booktitle = {Image Analysis and Processing. ICIAP 2025 Workshops},
  year={2025},
  note={in press}
}
```
