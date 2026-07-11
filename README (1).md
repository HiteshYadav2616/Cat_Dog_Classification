# 🐶🐱 Dog vs Cat Image Classifier (PyTorch CNN)

A convolutional neural network built from scratch in PyTorch to classify images as **Cat** or **Dog**, trained on the [Kaggle Cats vs Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

## 📊 Results

| Metric | Score |
|---|---|
| Train Accuracy (final epoch) | 89.22% |
| Validation Accuracy (best) | 86.32% |
| **Test Accuracy** | **86.27%** |

Trained for 15 epochs; best checkpoint saved automatically whenever validation accuracy improved.

## 🧠 Model Architecture

A custom CNN (`MyModel`) with 3 convolutional blocks followed by a fully connected classifier:

```
Conv2d(3 → 16, 3x3) → ReLU → MaxPool2d
Conv2d(16 → 32, 3x3) → ReLU → MaxPool2d
Conv2d(32 → 64, 3x3) → ReLU → MaxPool2d
Flatten
Linear(64*16*16 → 256) → ReLU → Dropout(0.5)
Linear(256 → 64) → ReLU
Linear(64 → 2)
```

- **Input size:** 128×128 RGB images
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam (lr = 0.001)
- **Epochs:** 15
- **Batch size:** 100

## 🗂️ Dataset & Preprocessing

- Images loaded via `torchvision.datasets.ImageFolder`
- Split into **70% train / 15% validation / 15% test**, using a fixed random seed (42) for reproducibility
- **Training transforms:** resize (128×128), random horizontal flip, random rotation (±10°), normalization (ImageNet mean/std)
- **Validation/test transforms:** resize + normalization only (no augmentation)
- Dataset sizes used: 15,015 train / 3,217 validation / 3,219 test images

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision matplotlib
```

### Dataset setup

Download the Cats vs Dogs dataset and organize it in `ImageFolder` format:

```
PetImages/
├── Cat/
│   ├── 0.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    └── ...
```

Update the `root` path in the notebook to point to your local dataset location.

### Training

Run the notebook cells in order:
1. Load and transform the dataset
2. Split into train/val/test sets and create `DataLoader`s
3. Train the model — the best checkpoint is saved to `Cat_Dog.pth` whenever validation accuracy improves
4. Evaluate on the held-out test set

### Inference

Load the saved weights and run predictions on a single image:

```python
model.load_state_dict(torch.load("Cat_Dog.pth"))
model.eval()
```

The notebook includes an example that displays a test image alongside its predicted vs. actual label.

## 📁 Project Structure

```
.
├── Dog_Cat.ipynb      # Main notebook (data loading, training, evaluation, inference)
├── Cat_Dog.pth         # Saved model weights (best validation checkpoint)
└── README.md
```

## 🔧 Possible Improvements

- Add batch normalization to stabilize and speed up training
- Experiment with transfer learning (e.g., ResNet18, EfficientNet) for higher accuracy
- Add a learning rate scheduler
- Track experiments with TensorBoard or Weights & Biases
- Expand augmentation (color jitter, random crop) to improve generalization

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
