# Brain Tumor Diagnosis with NumPy

## 🚀 **Project Overview**
This project implements a machine learning model to diagnose brain tumors from medical images, using NumPy as the primary data processing library.

## 📚 **Table of Contents**
- [Introduction](#-introduction)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contact](#-contact)

---

## 🧑‍⚕️ **Introduction**
Early diagnosis of brain tumors is crucial to improving treatment and survival rates. The goal of this project is to facilitate automated diagnosis by analyzing medical images (e.g., MRI scans) using **NumPy-based** data processing techniques. This is a **from-scratch** model implemented purely with NumPy, intended for **educational purposes** only, and **not** for real-world use.

---

## ⚡ **Features**
- **Image Processing:** Uses NumPy for handling and analyzing medical images.
- **Custom Models:** Implements machine learning models without relying on high-level frameworks.
- **Performance Evaluation:** Includes metrics to assess model accuracy and efficiency.

---

## 🛠️ **Prerequisites**
Before running the project, ensure you have the following Python packages installed:

- `NumPy`
- `Matplotlib`
- `cv2` (OpenCV)

---

## 📥 **Installation**

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/OT1devl/Brain-tumor-diagnosis-with-numpy.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Brain-tumor-diagnosis-with-numpy
   ```

3. (Optional) Create a virtual environment to isolate dependencies:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ **Usage**

1. Ensure that medical images are located in the appropriate directory within the `datasets` folder.

2. Run the training notebook to train the model:

   ```bash
   jupyter notebook train.ipynb
   ```

3. Once trained, evaluate the model's performance using the evaluation code.

---

## 🗂️ **Project Structure**

- `datasets/`: Contains medical images used for training and testing.
- `accuracies.py`: Implements functions to calculate model accuracy.
- `activations.py`: Contains activation functions used in the neural network.
- `layers.py`: Defines the layers of the neural network.
- `losses.py`: Implements loss functions to evaluate the model.
- `models.py`: Implements the machine learning model.
- `optimizers.py`: Includes optimization algorithms for training.
- `utils.py`: Provides utility functions for various project tasks.
- `train.ipynb`: Notebook for training the model.

---

## 📱 **Contact**

For any questions or suggestions, please contact me at:  
📧 [otidevv1@gmail.com](mailto:otidevv1@gmail.com)  
🌐 Visit my [GitHub profile](https://github.com/OT1devl).

