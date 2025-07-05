# CNN-Deep-Learning-Model-for-255-Bird-Species-Classification
# 🐦 Bird Species Classification using CNNs & Transfer Learning

**Team Members**: George Kanazi, Ammar Mnaa  
**Dataset**: 89,885 bird images across 255 species  
**Goal**: Classify bird species using CNNs and evaluate performance using various deep learning strategies.

---

## 📁 Dataset Overview

- Total images: **89,885**
- Classes: **255 unique bird species**
- Image size: `[3, 224, 224]`
- Dataset splits:
  - **Train**: 84,635
  - **Validation**: 2,625
  - **Test**: 2,625 (5 samples per class)

📝 Most images are high quality, with a single bird covering ~50% of the frame.  
No augmentation was initially applied due to dataset size and compute constraints.

---

## 🔍 Part 1: Exploratory Data Analysis (EDA)

- Dataset is **relatively balanced**, with ~161 images per class.
- Min: 130 | Max: 263
- Sample images reveal both visually similar birds (e.g., Fan Tailed Widow vs. Fairy Bluebird) and highly distinct ones (e.g., Wild Turkey, Barn Owl).
- See plots and sample visualizations in the notebook.

---

## 🧠 Part 2: CNN Modeling

### 🏗 Initial Model

- CNN trained with **5-fold cross-validation** (20 epochs per fold)
- **Average Accuracy** on test set: **66.4%**
- Based on early results, we proposed the following enhancements:
  1. **Batch Normalization** – stabilize and speed up training
  2. **Learning Rate Scheduler** – improve convergence
  3. **Dropout** – reduce overfitting *(not applied initially)*

### ⚙️ Improved Model

- Implemented **batch norm** and **learning rate scheduler**
- Further suggestions:
  1. Deepen the architecture
  2. Use pretrained weights (transfer learning)
  3. Add dropout when needed
- **New Average Accuracy**: **80.28%**  
- **Peak Accuracy (25 epochs)**: **~90%**

---

## 🧪 Part 3: Transfer Learning

Used the following pretrained architectures (epochs = 10, batch size = 64):

| Model       | # Params     | Val Acc | Test Acc | Test Loss | Train Time |
|-------------|--------------|---------|----------|-----------|------------|
| ResNet18    | 11.7M        | 0.9006  | 0.9209   | 0.282     | 59 mins    |
| AlexNet     | 59.1M        | 0.7573  | 0.7832   | 1.012     | 49 mins    |
| GoogleNet   | 6.6M         | 0.8907  | 0.9078   | 0.3533    | –          |
| ResNet34    | 21.8M        | 0.907   | 0.9362   | 0.2344    | 64 mins    |

---

## 🛠️ Experiment: Feature Extraction

- Used **GoogleNet** as a feature extractor
- Extracted 1024-dimensional feature vectors (before final layer)
- Trained:
  - **RandomForest** (`max_depth=10`)
  - **KNN** (`n_neighbors=30`)
- Performance dropped significantly vs. end-to-end CNNs
- KNN performed better than RF due to feature similarity matching
- Confusion matrices were not useful due to:
  - Large number of classes (525)
  - Small test set (2,625 samples → only 5 per class)

---

## 🔄 Experiment: Test Data Augmentation

- Applied rotations, flips, and zoom to **test data**
- Accuracy **dropped by 6%**  
  ➤ Reason: Model never saw augmented birds during training

---
## 🐤 Experiment: Adding a New Bird Species (256 Classes)
We initially trained our improved CNN model on 255 bird species. To evaluate the model's flexibility and adaptability, we introduced an additional class, bringing the total number of species to 256.

📥 Added 200 training images of a new species: Common Kingfisher

🧪 Added 5 test samples for the new class

🔁 Retrained only the improved CNN model

📈 Results:
Surprisingly, the overall test accuracy improved

The model successfully learned the new species without misclassifying other birds as the new one

Only 1 out of the 5 new samples was misclassified

The improved generalization suggests the model benefited from exposure to more data and diversity

---

## 📌 Key Techniques
 
- 🔧 CNN training
- 🔁 K-Fold Cross Validation
- 🧮 Batch Normalization
- 🔥 ReLU Activation
- 🧠 Transfer Learning (ResNet, GoogleNet, AlexNet)
- 📊 Softmax + CrossEntropy Loss
- 🎯 Feature Extraction for ML models (KNN, RandomForest)
- 🧪 Augmentation Testing

---

## 📷 Sample Results

- Classification examples: correct & incorrect predictions
- Learning curves for each fold
- Performance plots for baseline vs. improved vs. transfer learning models

---

## 💡 Conclusion

We showed that:
- Deeper models and transfer learning drastically improve accuracy
- Augmentation on test data without training augmentation reduces performance
- Feature extraction with classical ML models is less effective on complex datasets
- Our final model reached up to **90% accuracy** on the 255-class dataset
