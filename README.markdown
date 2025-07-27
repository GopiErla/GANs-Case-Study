# Generative Adversarial Networks (GANs): Case Study

This repository contains the implementation of Generative Adversarial Networks (GANs) developed for the *Advanced Research Topics in Data Science* module at the University of Hertfordshire. The project, completed by Gopi Erla on July 27, 2025, explores GANs for synthetic 2D data and real-world sketch generation using PyTorch.

## Introduction

This project investigates the design and evaluation of GANs applied to synthetic 2D datasets and real-world sketches. The objectives are to:
- Strengthen understanding of generative modeling.
- Implement GANs from scratch for 2D synthetic data (sine wave, spiral, Swiss Roll).
- Apply a Deep Convolutional GAN (DCGAN) to generate smiley face sketches from the QuickDraw dataset.

The project is divided into:
- **Part 1**: GANs on synthetic 2D data.
- **Part 2**: DCGAN on QuickDraw smiley face sketches.

## Repository Structure

```
├── sine_wave_gan.py         # Sine Wave GAN implementation
├── spiral_gan.py            # Spiral GAN implementation
├── swiss_roll_gan.py        # Swiss Roll GAN implementation
├── quickdraw_smiley_gan.py  # QuickDraw Smiley Face DCGAN implementation
├── data/                    # Directory for QuickDraw dataset (download separately)
├── results/                 # Output directory for plots and samples
└── README.md                # This file
```

## Requirements

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

- **Python**: 3.8+
- **PyTorch**: For GAN implementation.
- **NumPy**: For data generation.
- **Matplotlib**: For visualization.
- **Scikit-learn**: For Swiss Roll dataset.

For the QuickDraw experiment, download `smiley_face.npy` from [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and place it in `data/`.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Prepare QuickDraw dataset:
   - Place `smiley_face.npy` in `data/`.

3. Run experiments:
   - Sine Wave GAN: `python sine_wave_gan.py`
   - Spiral GAN: `python spiral_gan.py`
   - Swiss Roll GAN: `python swiss_roll_gan.py`
   - QuickDraw DCGAN: `python quickdraw_smiley_gan.py`

4. Results:
   - Plots and generated samples are saved in `results/`.

## Part 1: GANs on 2D Synthetic Data

### 1.1 Sine Wave GAN
- **Objective**: Generate 2D points \((x, \sin(x))\) for \(x \in [-\pi, \pi]\).
- **Data**: Real samples from \((x, \sin(x))\).
- **Architecture**:
  - Generator: 2-layer MLP, ReLU, outputs 2D points.
  - Discriminator: 2-layer MLP, LeakyReLU, sigmoid output.
- **Training**:
  - Loss: Binary Cross Entropy (BCE).
  - Optimizer: Adam (lr=0.001, β₁=0.5, β₂=0.999).
  - Epochs: 5000, Batch size: 64.
- **Results**: Generator approximated sine wave with minor noise, demonstrating GANs’ ability to learn non-linear distributions.

### 1.2 Spiral GAN
- **Objective**: Replicate a 2D spiral distribution.
- **Data**: 1000 points from parametric equations \(x = r \cdot \cos(\theta)\), \(y = r \cdot \sin(\theta)\), with noise.
- **Architecture**:
  - Generator: 3-layer MLP, LeakyReLU, BatchNorm, Tanh output.
  - Discriminator: 3-layer MLP, LeakyReLU.
- **Training**:
  - Loss: BCE.
  - Optimizer: Adam (lr=0.0001).
  - Epochs: 50,000, Batch size: 64.
- **Results**: Accurate spiral shape by epoch 49,000+, with tight curvature.

### 1.3 Swiss Roll GAN
- **Objective**: Model a 2D Swiss Roll manifold.
- **Data**: 10,000 points from `sklearn.datasets.make_swiss_roll`, 2D slice with Gaussian noise (std=0.2).
- **Architectures Compared**:
  - **Original**: 2-layer MLPs (Generator: ReLU; Discriminator: LeakyReLU).
  - **Modified**: 3-layer MLPs (Generator: LeakyReLU, Tanh; Discriminator: ReLU).
- **Training**:
  - Loss: BCE.
  - Optimizer: Adam (lr=0.0002).
  - Epochs: 2000, Batch size: 256.
- **Results**:
  - Original: Partial spiral, minor mode collapse.
  - Modified: Clear spiral, better spread, no mode collapse.

## Part 2: QuickDraw Smiley Face DCGAN

- **Dataset**: QuickDraw "smiley face" (28x28 grayscale, normalized to [-1, 1]).
- **Objective**: Generate realistic smiley face sketches.
- **Architecture**:
  - Generator: 100D noise → ConvTranspose2d, BatchNorm, ReLU, Tanh output.
  - Discriminator: Conv2d, LeakyReLU (slope=0.2), optional Dropout, Sigmoid output.
- **Training**:
  - Loss: BCE with label smoothing (real=0.9).
  - Optimizer: Adam (lr=0.0001, β₁=0.5, β₂=0.999).
  - Epochs: 100, Batch size: 128.
- **Results**:
  - Epoch 10: Basic outlines.
  - Epoch 100: Clear faces with varied expressions.
  - Discriminator loss ~0.7, generator loss decreased steadily.

## Reflections and Insights

- **Architectural Improvements**:
  - LeakyReLU prevented vanishing gradients.
  - BatchNorm and Tanh improved realism and convergence.
- **Training Stability**:
  - Label smoothing and dropout enhanced robustness.
  - Visual inspection was critical due to limited objective metrics.
- **Challenges**:
  - Balancing generator and discriminator learning rates.
  - Avoiding mode collapse in synthetic data experiments.
- **Future Work**:
  - Explore Wasserstein GANs for smoother training.
  - Investigate conditional GANs for controlled sketch generation.

## Conclusion

This project demonstrated GANs’ capability to model complex distributions, from synthetic 2D data to real-world sketches. Key findings include the importance of deeper architectures, non-linear activations, and stable training techniques. Future enhancements could include advanced GAN variants or conditional generation.

## References

- PyTorch: [https://pytorch.org](https://pytorch.org)
- QuickDraw Dataset: [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

## Author

- **Gopi Erla**
- **Module**: Advanced Research Topics in Data Science
- **Institution**: University of Hertfordshire
- **Date**: July 27, 2025