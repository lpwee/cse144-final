# cse144-final

## Leaderboard Link
https://www.kaggle.com/competitions/ucsc-cse-144-winter-2026-final-project/

## Download the data
```
pip install kaggle
```
```
kaggle competitions download -c ucsc-cse-144-winter-2026-final-project
unzip ucsc-cse-144-winter-2026-final-project.zip
```

## Dependency Management

uv sync
uv run jupyter notebook

## Performance Improvements

### Baseline

The baseline model fine-tunes a pretrained ResNet-50 on the training set (1,079 images, 100 classes) using Adam with a learning rate of 1e-4. Images are resized to 224x224 and normalized with ImageNet statistics. No data augmentation or regularization is applied.

### Data Augmentation

With only ~10 images per class, overfitting is a major concern. To address this, we added random augmentations to the training pipeline so the model sees a different variation of each image every epoch:

- **RandomResizedCrop(224, scale=(0.8, 1.0))** — Randomly crops and resizes the image, forcing the model to learn from different spatial regions rather than memorizing exact positions.
- **RandomHorizontalFlip** — Flips images horizontally with 50% probability, effectively doubling the data diversity.
- **RandomRotation(15)** — Rotates images up to 15 degrees, making the model invariant to slight orientation changes.
- **ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)** — Randomly perturbs color properties, preventing the model from relying on specific lighting conditions.

A separate `test_transform` without augmentation is used at inference time so predictions are deterministic.

### SGD with Momentum

We replaced the Adam optimizer with SGD (lr=0.01, momentum=0.9, weight_decay=1e-4):

- **SGD with momentum** tends to generalize better than Adam on image classification tasks, as it finds flatter minima in the loss landscape.
- **Weight decay (1e-4)** adds L2 regularization, penalizing large weights to further reduce overfitting.
- **Higher learning rate (0.01 vs 1e-4)** is standard for SGD and allows the model to make larger updates early in training, while momentum smooths out noisy gradients.

## Preliminary attempts
EfficientNet with no fine-tuning:
0.00909

