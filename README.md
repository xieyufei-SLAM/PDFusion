# PDFusion: A Domain-Adaptive Incremental Learning Model for Lithium-Ion Battery State Estimation

## Overview

PDFusion is a cutting-edge domain-adaptive incremental learning model designed for the joint estimation of the State of Charge (SOC) and State of Energy (SOE) in lithium-ion battery management systems. This model leverages a combination of physical models and data-driven approaches to improve prediction accuracy under varying conditions.

**Key Features:**

- **Domain Adaptation**: Enhances model robustness across different battery conditions.
- **Incremental Learning**: Allows the model to continuously learn from new data without retraining from scratch.
- **Physical-Data Driven Approach**: Combines physical models with advanced data-driven methods for more accurate state estimation.

## Architecture

PDFusion incorporates several innovative components:
- **Weighted fusion denoising Dual Kalman Filtering (WDKF)**: This method filters and fuses the physical model's output, enhancing prediction accuracy.
- **Transformer Model with Time-frequency Interactive Attention (TIA)**: Captures long-term dependencies and high-frequency details to address nonlinear temporal dependencies and non-stationary characteristics of batteries.

![](figs\frame.jpg)

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Matplotlib
- ...

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

**Set Dataset**: Ensure your dataset is organized in the following structure:

```bash
data/
├── Li_ion/
│   ├── battery1.csv
│   ├── battery2.csv
│   └── ...
└── IL/
```

**Analyse** :the DataRun the following script to preprocess your data or Calce Data:

```bash
python SOC_OCV_Fit.py
python SOC_SOE_calc_Calce.py
python SHAPanalysis.py
python concat_data.py
...
```

### Model Training

You can alse train the model using our data examples, run and config your args to change model or using error correction strategy:

```bash
python run.py
```

## Code Reference

PDFusion is inspired by and partially based on the [PatchTST](https://github.com/yourusername/PatchTST) codebase. We appreciate the contributions from the authors of PatchTST.

## Citation

If you use PDFusion in your research, please cite:

```
@article{2024PDFusion,
  title={PDFusion: A Domain-Adaptive Incremental Learning Model for Lithium-Ion Battery State Estimation},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  publisher={Publisher Name}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

We welcome contributions! Please read our [CONTRIBUTING](CONTRIBUTING.md) guide to get started.

## Acknowledgments

We acknowledge the support from [PatchTST](https://github.com/PatchTST/PatchTST)for providing foundational code that inspired parts of this project.

