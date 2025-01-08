# Fire and Smoke Detection using CNN and Computer Vision

## Overview
This project aims to create a fire and smoke detection model using a Convolutional Neural Network (CNN) and computer vision techniques. The model identifies fire and smoke from images and sends alerts to ensure safety in monitored areas. The system is designed for real-time fire detection, useful in environments such as campus surveillance systems.

## Table of Contents
Project Description
Technologies Used
Setup and Installation
Training the Model
Usage
Project Structure
Contributing
License
Acknowledgments

## Project Description
This system leverages Convolutional Neural Networks (CNNs) for detecting fire and smoke in images. The model is trained using a dataset of fire and smoke images, which are pre-processed for object classification tasks. This system has been developed to detect fire and smoke in real-time to improve safety protocols.

## Key Features:
CNN-based Fire and Smoke Detection: Utilizing deep learning to detect fire and smoke.
Real-time Image Classification: The model can classify fire and smoke in real-time images.
Alert System: Integrated with a web-based alert system to notify authorities upon detection.

## Technologies Used
Python: The primary programming language used in the project.
TensorFlow/Keras: For training and deploying the CNN model.
OpenCV: For image processing and visualization.
Flask: For creating a web-based alert system (optional).
Google Colab: Used for training the CNN model in a cloud environment.
Git: Version control for tracking changes and managing the codebase.

Setup and Installation
Follow these steps to set up the project on your local machine.

Prerequisites
Python 3.x installed on your system.
pip for managing Python packages.
Git for version control.
An IDE or text editor (e.g., VSCode, PyCharm).
Clone the Repository
First, clone the repository:

bash
Copy code
git clone https://github.com/your-username/fire-and-smoke-detection.git
cd fire-and-smoke-detection
Create a Virtual Environment
It is recommended to create a virtual environment to isolate the dependencies of the project:

bash
Copy code
python -m venv myenv
Activate the virtual environment:

On Windows:
bash
Copy code
myenv\Scripts\activate
On macOS/Linux:
bash
Copy code
source myenv/bin/activate
Install Dependencies
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
This will install all the necessary libraries such as TensorFlow, OpenCV, etc.

## Training the Model
Data Collection
Dataset: You need to gather or use a pre-existing dataset containing images of fire and smoke. This dataset should include images in various environments and conditions to train a robust model.

Labeling: If the dataset is not labeled, you can use tools like LabelImg to annotate the images for classification.

## Training the Model
Training Script: Open the train_model.py script to begin training the CNN model on the labeled dataset.
Data Preprocessing: The images will be preprocessed (resizing, normalization, etc.) before feeding them into the CNN.
Model Architecture: The model is built using Keras with TensorFlow backend. It is a custom CNN architecture designed for binary classification (fire vs. no fire, smoke vs. no smoke).
After training, the model will be saved as fire_and_smoke_model.h5.

For a detailed step-by-step training process, refer to the train_model.py script.

## Usage
Fire and Smoke Detection
Once the model is trained, you can use it for fire and smoke detection. Here's how to run the detection:

Load the Model:
python
Copy code
from tensorflow.keras.models import load_model

model = load_model('fire_and_smoke_model.h5')
Prepare the Image:
python
Copy code
import cv2

## Load the image
img = cv2.imread('image.jpg')

## Resize the image to match the model input size
img_resized = cv2.resize(img, (224, 224))  # Resize to 224x224 (input size for the model)
Preprocess the Image:
python
Copy code
import numpy as np

## Normalize the image
img_array = np.array(img_resized) / 255.0
img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for batch dimension
Make Predictions:
python
Copy code

## Get prediction
prediction = model.predict(img_array)
print(prediction)
If the model predicts fire or smoke, you can trigger the alert system.

## Web Application (Alert System)
To start the web application, you can use Flask to set up an alert system that notifies when fire or smoke is detected.

bash
Copy code
python app.py
This will launch a local server to interact with the system, which can be used to display alerts and notifications.

## Project Structure
Here is the structure of the project:

bash
Copy code
fire-and-smoke-detection/
│
├── app.py                 # Flask application for the alert system
├── train_model.py         # Script for training the CNN model
├── fire_and_smoke_model.h5 # Trained CNN model
├── requirements.txt       # List of required Python packages
├── .gitignore             # Files to be ignored by Git
└── README.md              # Project documentation

## Contributing
We welcome contributions! To contribute to the project:

## Fork the repository.
Create a new branch (git checkout -b feature-name).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push to your branch (git push origin feature-name).
Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for more details.

## Acknowledgments
TensorFlow for deep learning and model training.
Keras for building the CNN model.
OpenCV for image processing and computer vision tasks.
Google Colab for cloud-based model training.
Git for version control and collaboration.
