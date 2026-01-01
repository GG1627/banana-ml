# 🍌 Banana Ripeness Prediction

Predict remaining shelf life of bananas using Vision Transformer (ViT) fine-tuning for regression.

## What It Does

Takes banana images as input and predicts how many days are left before the banana becomes overripe (0-12 days scale).

## Approach

- **Architecture**: Vision Transformer (ViT-Base) fine-tuned for regression
- **Method**: Transfer learning with pre-trained ImageNet weights
- **Task**: Continuous value prediction (regression, not classification)

## Results

- **Mean Absolute Error**: **0.58 days**
- **Root Mean Squared Error**: **0.79 days**

![Evaluation Results](evaluation_results.png)

## Quick Start

# Install dependencies
pip install torch torchvision timm scikit-learn pandas numpy matplotlib pillow

# Run training in notebooks/setup.ipynb## Model Performance

The model achieves production-ready accuracy, predicting banana ripeness within less than 1 day on average.