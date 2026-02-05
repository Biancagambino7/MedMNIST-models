# 🧠 Model Comparison on MedMNIST (DermaMNIST)

This project explores **image classification on the MedMNIST dataset**, with a specific focus on **DermaMNIST**.  
The goal is to compare multiple machine learning and deep learning models to determine which performs best when considering **both accuracy and computational efficiency**.

All experiments are implemented and run in **Google Colab**, but you can find the main dataset [here](https://medmnist.com/)

---

## 🚀 Open in Google Colab

Click the badge below to run the notebook instantly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aZdJwG76srdLOoGHXBFtxFF0J0AYa9uI?usp=sharing)

---

## 📊 Dataset: MedMNIST – DermaMNIST

**MedMNIST** is a collection of lightweight medical image datasets designed for benchmarking machine learning models.

This project uses **DermaMNIST**, which contains:
- 2D RGB images of skin lesions
- Multiple diagnostic classes
- Small image size, making it ideal for fast experimentation and fair model comparison

---

## 📘 Notebook Walkthrough

The notebook is organized as a **step-by-step experimental pipeline**.  
Running the notebook **from top to bottom** will reproduce all results.
If you want to just look at the models and results, refer to the **table of contents** to the left of the notebook.

### 1️⃣ Library Imports & Setup
- Imports required Python and PyTorch libraries
- Sets device configuration (CPU/GPU)
- Ensures reproducibility and consistent environment setup

---

### 2️⃣ Data Loading & Basic Visualization
- Loads the DermaMNIST dataset
- Displays example images and labels
- Provides an initial understanding of class distribution and image structure

---

### 3️⃣ Data Transformations
- Demonstrates different preprocessing and transformation options
- Includes resizing, normalization, and tensor conversion
- Ensures consistent input formatting across all models

---

### 4️⃣ Experimental Baseline Configuration
A shared baseline is established so all models are compared fairly:
- Same train/test split
- Same number of epochs
- Same batch size
- Same optimization strategy where applicable

This ensures performance differences are due to **model architecture**, not training conditions.

---

### 4️⃣ Running the Models
This is where you will find the types of models evaluated which you can then compare:
 - Linear
 - MLP
 - CNN
 - ResNet
 - ViT

You can see accuracy after each step, and then compare at the very end to find the best one.

---

## 🧪 Models Evaluated

The following models were implemented and tested on the same dataset:

1. **Linear Model**
   - Baseline approach
   - Helps establish a lower bound for performance

2. **Multi-Layer Perceptron (MLP)**
   - Fully connected neural network
   - Captures non-linear relationships but ignores spatial structure

3. **Convolutional Neural Network (CNN)**
   - Leverages spatial features in images
   - Designed specifically for image data

4. **ResNet**
   - Deep convolutional network with residual connections
   - Improves gradient flow and deeper feature learning

5. **Vision Transformer (ViT)**
   - Transformer-based architecture adapted for images
   - Powerful but computationally expensive

---

## 📈 Key Findings

- **CNN consistently performed the best overall**
  - High accuracy
  - Relatively fast training time
  - Strong balance between performance and efficiency

- **Linear and MLP models**
  - Faster to train
  - Significantly lower accuracy due to lack of spatial feature learning

- **ResNet**
  - Accuracy changed based on epoch size, but was always lower than CNN
  - Higher computational cost compared to standard CNN

- **Vision Transformer (ViT)**
  - Longest training time
  - Did not consistently outperform CNN on this dataset
  - Less efficient for small medical image datasets like DermaMNIST

---

## ✅ Conclusion

- For epoch of 10 these were the results: Linear Accuracy: 0.4596100278551532, MLP Accuracy: 0.577158774373259,
CNN Accuracy: 0.7884401114206128, ResNet Accuracy: 0.3279944289693593, 

- For **DermaMNIST**, a well-designed **CNN** provides the best tradeoff between:
  - Accuracy, Training time, Computational efficiency

I noticed that weather it was small amounts of epochs and large amounts of epochs, CNN was still the highest accuracy just the difference in number varied with small amounts it was closer to linear and mlp accuracies and larger was closer to resnet and vit accuracy, but still always larger. Overall, the more complex models such as **ResNet** and **ViT** did not offer sufficient performance gains to justify their added cost in this setting. CNN was the greatest between accuracy and time. </b> </b>

The CNN significantly outperformed the other models because MedMNIST consists of low-resolution (28×28) medical images and a relatively small dataset, where preserving local spatial structure is critical. Convolutional layers efficiently capture edges and textures while sharing weights, reducing overfitting and enabling faster convergence with small batch sizes and limited training epochs. In contrast, MLPs lose spatial information when images are flattened, and transformer-based models such as ViT require larger datasets, higher image resolutions, and larger batch sizes to learn effective spatial representations.

---

## 🛠 Technologies Used

- Python
- Google Colab
- PyTorch
- MedMNIST
- NumPy
- Matplotlib

---

## ✨ Author

Created by **Bianca Gambino**  

