# Fire and Smoke Detection using CNN and Computer Vision also the YOLO model. 

## Overview
This project provides a comprehensive, real-time fire and smoke detection system leveraging deep learning (YOLOv8) and computer vision. The system is designed to identify fire and smoke in images or video streams, trigger alerts, and provide a web-based interface for monitoring. It is suitable for campus surveillance, homes, industrial settings, and other safety-critical environments.
 
## Features
- **Real-time Detection:** Fast and accurate fire and smoke detection using a YOLOv8-based model.
- **Custom Dataset:** Trained on a dataset of over 1,390 images, including both collected and sourced images, with extensive data augmentation for robustness.
- **Alert System:** Plays an audible alert and can be extended to send notifications when fire or smoke is detected.
- **Web Interface:** User-friendly web interface built with Flask for uploading images, viewing detection results, and monitoring video streams.
- **Extensible:** Easily adaptable to new datasets, additional alert mechanisms, or deployment environments.
- **Sample Data:** Includes sample images and model weights for quick testing and demonstration.

## Directory Structure
```
fire detection model/
├── Advance-fire-and-smoke-detection-system-using-CNN.ipynb  # Jupyter notebook for model development and experiments
├── app.py                                                  # Main Flask web server application
├── fire-in-a-house.webp                                    # Sample image for testing
├── fire.pt                                                 # Trained YOLOv8 model weights
├── fire.py                                                 # Fire and smoke detection logic and utilities
├── main.py                                                 # Alternate script for running detection
├── smoke-background.webp                                   # Additional sample image
├── templates/                                              # HTML templates for the web interface
│   ├── about.html
│   ├── contact.html
│   └── index.html
├── static/                                                 # Static files (CSS, JS, images, audio)
│   └── alert.mp3                                           # Alert sound played on detection
├── README.md                                               # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd "fire detection model"
   ```
2. **Install dependencies**
   - If `requirements.txt` is available:
     ```powershell
     pip install -r requirements.txt
     ```
   - If not, install the main dependencies manually:
     ```powershell
     pip install flask opencv-python torch ultralytics
     ```

### Running the Application
1. **Start the Flask web server:**
   ```powershell
   python app.py
   ```
2. **Access the web interface:**
   Open your browser and go to [http://localhost:5000](http://localhost:5000)

## Usage
- **Web Interface:**
  - Upload an image or use a webcam feed to detect fire or smoke.
  - The system displays detection results and plays an alert sound if fire or smoke is found.
  - Navigate to About and Contact pages for more information.
- **Jupyter Notebook:**
  - Use `Advance-fire-and-smoke-detection-system-using-CNN.ipynb` for model training, evaluation, and experimentation.
- **Scripts:**
  - `main.py` and `fire.py` can be used for running detection from the command line or integrating into other systems.

## Model Details
- **Architecture:** YOLOv8 (You Only Look Once, version 8) for object detection.
- **Training:**
  - Dataset: 1,390+ images with fire and smoke, including data augmentation.
  - The model is trained to distinguish between fire, smoke, and background.
- **Weights:**
  - The trained model weights are stored in `fire.pt`.
  - You can retrain the model using the provided notebook and your own dataset if desired.

## File Descriptions
- `app.py`: Flask application providing the web interface and API endpoints for detection.
- `fire.py`: Contains the core logic for loading the model and running detection on images or video frames.
- `main.py`: Alternate entry point for running detection scripts.
- `Advance-fire-and-smoke-detection-system-using-CNN.ipynb`: Jupyter notebook for model development, training, and evaluation.
- `fire.pt`: Pre-trained YOLOv8 model weights for fire and smoke detection.
- `fire-in-a-house.webp`, `smoke-background.webp`: Sample images for testing the detection pipeline.
- `static/alert.mp3`: Audio file played when fire or smoke is detected.
- `templates/`: HTML templates for the web interface (index, about, contact pages).

## Extending the Project
- **Add new alert mechanisms:** Integrate email, SMS, or IoT device notifications in `app.py` or `fire.py`.
- **Deploy to cloud or edge devices:** Adapt the Flask app for deployment on servers, Raspberry Pi, or cloud platforms.
- **Retrain the model:** Use the notebook to retrain with new data for improved accuracy or additional classes.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License
This project is for educational and research purposes. Check individual files for license details.

## Acknowledgements
- YOLOv8 by Ultralytics
- OpenCV for computer vision utilities
- Flask for the web interface
- Dataset contributors and open-source community

