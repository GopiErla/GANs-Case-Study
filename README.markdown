# Generative Adversarial Networks (GANs): Case Study

This repository contains a Jupyter notebook implementing Generative Adversarial Networks (GANs) for the *Advanced Research Topics in Data Science* module at the University of Hertfordshire. The project, completed by Gopi Erla, explores GANs for generating synthetic 2D data and real-world sketches using PyTorch.

## Project Overview

This case study investigates the design, implementation, and evaluation of GANs applied to:
- **Synthetic 2D Data**: Sine wave, spiral, and Swiss Roll distributions.
- **Real-World Sketches**: Smiley face sketches from Google’s QuickDraw dataset.

The objectives are to:
- Deepen understanding of generative modeling techniques.
- Implement GANs from scratch for 2D synthetic datasets.
- Apply a Deep Convolutional GAN (DCGAN) to generate smiley face sketches.

All experiments are implemented in a single Jupyter notebook: `Generative Adversarial Networks (GANs) Case Study.ipynb`.

## Repository Structure

```
├── Generative Adversarial Networks (GANs) Case Study.ipynb  # Notebook with all GAN experiments
├── data/                                           # Directory for QuickDraw dataset (download separately)
├── results/                                        # Directory for output plots and samples
└── README.md                                       # This file
```

## Requirements

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

- **Python**: 3.8+
- **PyTorch**: For GAN implementation.
- **NumPy**: For data generation.
- **Matplotlib**: For visualization.
- **Scikit-learn**: For Swiss Roll dataset.
- **Jupyter**: To run the notebook.

For the QuickDraw experiment, download the `smiley_face.npy` file from the [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and place it in the `data/` directory.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Prepare the QuickDraw Dataset**:
   - Download `smiley_face.npy` from [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset).
   - Place it in the `data/` directory.

3. **Run the Notebook**:
   - Start Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open `Generative Adversarial Networks (GANs) Case Study.ipynb` and run all cells.
   - The notebook includes sections for:
     - Sine Wave GAN
     - Spiral GAN
     - Swiss Roll GAN
     - QuickDraw Smiley Face DCGAN

4. **View Results**:
   - Plots and generated samples (e.g., sine wave points, spiral patterns, Swiss Roll distributions, and smiley face sketches) are saved in the `results/` directory.

## Experiments

### Part 1: GANs on Synthetic 2D Data

#### 1.1 Sine Wave GAN
- **Objective**: Generate 2D points \((x, \sin(x))\) for \(x \in [-\pi, \pi]\).
- **Data**: Real samples from \((x, \sin(x))\).
- **Architecture**:
  - Generator: 2-layer MLP with ReLU, outputs 2D points.
  - Discriminator: 2-layer MLP with LeakyReLU, sigmoid output.
- **Training**:
  - Loss: Binary Cross Entropy (BCE).
  - Optimizer: Adam (lr=0.001, β₁=0.5, β₂=0.999).
  - Epochs: 5000, Batch size: 64.
- **Results**: Generator approximated the sine wave with minor noise.

#### 1.2 Spiral GAN
- **Objective**: Replicate a 2D spiral distribution.
- **Data**: 1000 points from parametric equations \(x = r \cdot \cos(\theta)\), \(y = r \cdot \sin(\theta)\), with noise.
- **Architecture**:
  - Generator: 3-layer MLP with LeakyReLU, BatchNorm, Tanh output.
  - Discriminator: 3-layer MLP with LeakyReLU.
- **Training**:
  - Loss: BCE.
  - Optimizer: Adam (lr=0.0001).
  - Epochs: 50,000, Batch size: 64.
- **Results**: Accurate spiral shape with tight curvature by epoch 49,000+.

#### 1.3 Swiss Roll GAN
- **Objective**: Model a 2D Swiss Roll manifold.
- **Data**: 10,000 points from `sklearn.datasets.make_swiss_roll`, 2D slice with Gaussian noise (std=0.2).
- **Architectures**:
  - Original: 2-layer MLPs (Generator: ReLU; Discriminator: LeakyReLU).
  - Modified: 3-layer MLPs (Generator: LeakyReLU, Tanh; Discriminator: ReLU).
- **Training**:
  - Loss: BCE.
  - Optimizer: Adam (lr=0.0002).
  - Epochs: 2000, Batch size: 256.
- **Results**:
  - Original: Partial spiral, minor mode collapse.
  - Modified: Clear spiral, better spread, no mode collapse.

### Part 2: QuickDraw Smiley Face DCGAN
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
  - Epoch 100: Realistic faces with varied expressions.

## Reflections

- **Architectural Improvements**:
  - LeakyReLU and BatchNorm improved convergence.
  - Tanh ensured realistic outputs for complex distributions.
- **Training Stability**:
  - Label smoothing and dropout enhanced robustness.
  - Visual inspection was critical due to limited objective metrics.
- **Challenges**:
  - Balancing generator and discriminator learning rates.
  - Avoiding mode collapse in synthetic data experiments.

## Conclusion

This project demonstrated GANs’ ability to model synthetic 2D distributions and real-world sketches. Deeper architectures, non-linear activations, and stable training techniques were critical for success. Future work could explore Wasserstein GANs or conditional GANs for enhanced performance.

## References

- PyTorch: [https://pytorch.org](https://pytorch.org)
- QuickDraw Dataset: [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

## Author

- **Gopi Erla**
- **Module**: Advanced Research Topics in Data Science
- **Institution**: University of Hertfordshire
