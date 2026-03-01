# Neural Network Quantum State Tomography

This repository implements a machine learning approach to quantum state tomography. Instead of using traditional iterative algorithms like maximum likelihood estimation, we train a deep neural network to map measurement probabilities directly to a valid quantum state description (Cholesky representation). The method follows the procedure described in the research paper, generating synthetic datasets for Hilbert space dimensions \(d = 3, 5, 7, 9\) and training a feed‑forward network with approximately \(2\times10^5\) parameters.

## Overview

Quantum state tomography reconstructs the density matrix \(\rho\) of an unknown quantum system from measurements. We frame this as a supervised learning problem:

1. **Generate random density matrices** from the Ginibre ensemble (\(\rho = XX^\dagger / \text{Tr}(XX^\dagger)\)).
2. **Define a fixed measurement scheme** – a rank‑1 POVM (positive operator‑valued measure) constructed from Haar‑distributed pure states via the square‑root measurement.
3. **Compute ideal measurement probabilities** using Born’s rule: \(p_\ell = \text{Tr}(\rho \Pi_\ell)\).
4. **Mix ideal probabilities with simulated experimental noise** – 75% of training samples keep the ideal probabilities, while 25% are obtained by multinomial sampling with a random number of trials (between \(d^2\) and \(10^5\)).
5. **Represent the target state** by its Cholesky decomposition \(\rho = T^\dagger T\), where \(T\) is a lower‑triangular matrix. The output vector consists of the real diagonal entries and the real/imaginary parts of the off‑diagonal entries, yielding a \(d^2\)-dimensional vector.
6. **Train a deep neural network** to predict the Cholesky vector from the probability vector.

The trained network can then reconstruct a quantum state from measurement data orders of magnitude faster than iterative methods.

## Dataset Generation

The datasets are created separately for each dimension \(d \in \{3,5,7,9\}\). For each dimension:

- **POVM:** A fixed set of \(N = d^2\) rank‑1 operators \(\{\Pi_\ell\}\) is constructed using the square‑root (pretty good) measurement from \(N\) Haar‑random pure states. This guarantees \(\sum_\ell \Pi_\ell = I\).
- **States:** \(10^6\) random density matrices are drawn from the Ginibre ensemble (using QuTiP’s `rand_dm` with `distribution='ginibre'`).
- **Input features:** For each state, the ideal probability vector \(p_\ell = \text{Tr}(\rho \Pi_\ell)\) is computed. For 25% of the training samples, this vector is replaced by sampled frequencies: draw \(T\) trials from a multinomial distribution with probabilities \(p_\ell\) and divide by \(T\).
- **Output labels:** The Cholesky vector of the state is computed and stored.
- **Splits:**
  - Training: 800,000 samples (with the 75%/25% ideal/sampled mixture)
  - Validation: 200,000 samples (same mixture)
  - Test: 1,000 samples (ideal probabilities only, for final benchmarking)

All data are saved as `.npy` files (e.g., `X_train.npy`, `Y_train.npy`, …).

## Neural Network Architecture

We use a deep feed‑forward network implemented in **TensorFlow/Keras**:

- **Input layer:** size \(d^2\) (the probability vector)
- **8 hidden layers** with neuron counts: 200, 180, 180, 160, 160, 160, 160, 100
- **Activation:** ReLU for all hidden layers
- **Output layer:** size \(d^2\) (Cholesky vector) with **tanh** activation – the Cholesky elements naturally lie in \([-1,1]\)
- **Total trainable parameters:** approximately \(2\times10^5\)

## Training

- **Loss function:** Mean squared error (MSE)
- **Optimizer:** Nadam (Nesterov‑accelerated Adam) with learning rate \(0.001\), \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-7}\)
- **Batch size:** 100
- **Maximum epochs:** 2000
- **Early stopping:** training halts if validation loss does not improve for 200 consecutive epochs; the best model (lowest validation loss) is restored.
- **Callbacks:** `EarlyStopping` and `ModelCheckpoint` (to save the best model)

Training history (loss and MAE) can be plotted to monitor convergence.

## Requirements

- Python 3.7+
- QuTiP (for quantum state generation)
- NumPy
- TensorFlow 2.x
- Matplotlib (for plotting)
- SciPy (for multinomial sampling)

Install dependencies with:

```bash
pip install numpy scipy matplotlib tensorflow qutip
