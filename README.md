#  Attendance System using Face Recognition

*By Vansh Dahiya*

## Introduction

The Attendance System using Face Recognition is a Python-based project that utilizes computer vision techniques to automatically recognize and record attendance of individuals. The system uses a webcam to capture facial images, extracts facial features using deep learning models, and matches them against a pre-trained model to identify individuals. The project aims to provide an efficient and accurate alternative to traditional attendance systems.

![image](https://github.com/VD0023/AttendanceSystemFaceRecognition/assets/99820386/d6d21cd7-dd67-4efe-b5eb-9a584e9c88ef)

## Features

1. **Face Detection**: The system utilizes a pre-trained deep learning model to detect faces in real-time video streams. This ensures accurate detection of faces even in varying lighting conditions and different orientations.

2. **Facial Feature Extraction**: Facial features are extracted from the detected faces using a deep learning model. This process generates high-dimensional embeddings that represent unique characteristics of each individual's face.

3. **Model Training**: The system employs machine learning techniques to train a Support Vector Machine (SVM) classifier on the extracted facial embeddings. The SVM model learns to recognize individuals based on their facial features.

4. **Real-time Recognition**: The trained model is used to recognize individuals in real-time video streams. The system matches the extracted facial embeddings with the trained SVM classifier and assigns names and roll numbers to recognized faces.

5. **CSV Database Integration**: The system integrates a CSV database to store student information such as names and roll numbers. The system reads from the database to display the corresponding student information when a face is recognized.

## Requirements

To run the Attendance System using Face Recognition project, the following requirements should be met:

1. **Python**: The project is developed using the Python programming language.

2. **OpenCV**: The OpenCV library is required for video capture, image processing, and face detection.

3. **imutils**: The imutils library is used for image resizing and other utility functions.

4. **scikit-learn**: The scikit-learn library is used for training the SVM classifier and label encoding.

5. **Deep Learning Models**: The project uses pre-trained deep learning models for face detection (deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel) and facial feature extraction (openface_nn4.small2.v1.t7).

## Project Workflow

The Attendance System using Face Recognition project consists of several stages that are executed sequentially. Here is an overview of the project workflow:

1. **Dataset Creation**: The dataset creation code captures facial images using a webcam and saves them in a directory structure based on individual names. It also stores corresponding student information in a CSV file.

2. **Preprocessing and Embeddings**: The preprocessing and embeddings code reads the captured facial images, detects faces using the face detection model, and extracts facial embeddings using the deep learning embedding model. The extracted embeddings and corresponding labels are saved in a pickle file for further processing.

3. **Training the Model**: The training code loads the extracted embeddings and labels, encodes the labels using label encoding, and trains an SVM classifier using the scikit-learn library. The trained model is saved for future use.

4. **Recognizing Persons**: The recognizing persons code loads the pre-trained face detection and embedding models, as well as the trained SVM classifier. It continuously captures video frames, detects faces, extracts embeddings, and matches them against the trained model to recognize individuals. The recognized faces are displayed along with their corresponding names and roll numbers from the CSV database.

5. **CSV Database Integration**: The recognizing persons code integrates a CSV database that contains student information. When a face is recognized, the code searches the CSV database for a matching name and retrieves the corresponding roll number. The name and roll number are displayed alongside the recognized face.

![image](https://github.com/VD0023/AttendanceSystemFaceRecognition/assets/99820386/538adcf9-4d0d-4845-b687-a2b5e8f97d99)

## Usage

To use the Attendance System using Face Recognition project, follow these steps

:

1. **Dataset Creation**: Run the dataset creation code to capture facial images of individuals and store them in the appropriate directory structure. Provide the necessary student information during the capture process to populate the CSV database.

2. **Preprocessing and Embeddings**: Execute the preprocessing and embeddings code to preprocess the captured images, detect faces, and extract facial embeddings. This step generates the necessary data for training the model.

3. **Training the Model**: Run the training code to train the SVM classifier on the extracted embeddings and labels. This step generates the trained model that will be used for recognition.

4. **Recognizing Persons**: Execute the recognizing persons code to initiate the real-time recognition process. The code will utilize the trained model to recognize faces in the video stream and display the recognized individuals along with their names and roll numbers from the CSV database.

## Conclusion

The Attendance System using Face Recognition project provides an efficient and accurate method for recording attendance. By leveraging deep learning models and machine learning algorithms, the system achieves real-time face detection, facial feature extraction, and person recognition. The integration of a CSV database further enhances the system by providing additional information about recognized individuals. This project showcases the capabilities of computer vision and machine learning in automating attendance management systems.
