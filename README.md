# VisionLensAI - Vision Quality Prediction with Deep Learning (PyTorch)

VisionLensAI is a PyTorch-based machine learning project that predicts a **vision quality score** from synthetic eye and lens condition data. It demonstrates an end-to-end ML workflow: data generation, preprocessing, neural network training, evaluation, and saving model artifacts.

## What it does
- Generates synthetic samples representing vision-related factors (blur, focus error, contrast, lighting sensitivity)
- Trains a PyTorch neural network to predict a vision quality score (regression)
- Evaluates performance and saves a trained model checkpoint

## Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib

## Setup
```bash
pip install -r requirements.txt