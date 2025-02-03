# **Project: Image Recognition and Processing**

## **Topic:**  
### **Detection of Edible Plant Diseases Using Image Recognition Based on the YOLOv8 Model**

---

## **Dataset Description**
The dataset chosen for model training is the *"Plants Diseases Detection and Classification"* dataset from Roboflow, licensed under Creative Commons. This dataset is designed for detecting plant diseases from leaf images. It consists of **2,516 images** with **8,732 labeled objects** classified into **29 categories**:

| Plant | Condition | Class (No. of Annotations) |
|--------|-------------|-------------------------------|
| Apple | Diseased | Apple Scab Leaf (152) |
| Apple | Diseased | Apple Rust Leaf (206) |
| Apple | Healthy | Apple Leaf (240) |
| Bell Pepper | Diseased | Bell Pepper Leaf Spot (267) |
| Bell Pepper | Healthy | Bell Pepper Leaf (315) |
| Blueberry | Healthy | Blueberry Leaf (766) |
| Cherry | Healthy | Cherry Leaf (247) |
| Corn | Diseased | Corn Gray Leaf Spot (74) |
| Corn | Diseased | Corn Leaf Blight (361) |
| Corn | Diseased | Corn Rust Leaf (123) |
| Grape | Diseased | Grape Leaf Black Rot (124) |
| Grape | Healthy | Grape Leaf (231) |
| Peach | Healthy | Peach Leaf (659) |
| Potato | Diseased | Potato Leaf Early Blight (325) |
| Potato | Diseased | Potato Leaf Late Blight (245) |
| Potato | Healthy | Potato Leaf (21) |
| Raspberry | Healthy | Raspberry Leaf (554) |
| Soybean | Healthy | Soybean Leaf (259) |
| Squash | Diseased | Squash Powdery Mildew Leaf (245) |
| Strawberry | Healthy | Strawberry Leaf (483) |
| Tomato | Diseased | Tomato Early Blight Leaf (205) |
| Tomato | Diseased | Tomato Mold Leaf (279) |
| Tomato | Diseased | Tomato Two-Spotted Spider Mites Leaf (2) |
| Tomato | Diseased | Tomato Septoria Leaf Spot (414) |
| Tomato | Diseased | Tomato Leaf Bacterial Spot (259) |
| Tomato | Diseased | Tomato Leaf Mosaic Virus (261) |
| Tomato | Diseased | Tomato Leaf Late Blight (216) |
| Tomato | Diseased | Tomato Leaf Yellow Virus (796) |
| Tomato | Healthy | Tomato Leaf (403) |

The dataset is split as follows:
- **Training:** 2,041 images  
- **Validation:** 250 images  
- **Testing:** 249 images  

---

## **System Requirements Description**

### **Objective:**
The project's goal is to **detect and localize plant diseases in images using the pre-trained YOLOv8 model.** The system aims to:
- **Classify objects (leaves) present in images.**  
- **Enable interaction with the model through a simple interface.**  

### **Scope:**
The project implements the following functionalities:
- **Uploading an image for analysis**  
- **Processing and analyzing the image for disease detection**  

### **Libraries Used:**
- **Ultralytics** – For loading and training the YOLOv8 model  
- **Roboflow** – To access the selected dataset  
- **IPyWidgets** – For building a user interface in Jupyter Notebook  
- **Pillow (PIL)** – Image processing  
- **os, shutil** – File manipulation  
- **time** – Adding timestamps for clear file naming  
- **matplotlib** – Visualizing model performance metrics  

---

## **Development Environment**
The project is implemented in **Python** and runs in **Google Colab**.  
The **interface is available in Jupyter Notebook** and operates locally in a Python environment.  
The system supports common image formats like **JPEG and PNG**.  

### **Output:**
- Images with **bounding boxes** for detected diseases  
- **Log with predictions**, including confidence scores and bounding box coordinates  
- **Model weights** for future use  

---

## **Limitations:**
- The script **can only detect classes it was trained on**.  
- Supports only **common image formats** (JPEG, PNG).  
- The model was trained in **Google Colab with limited free resources**, meaning its **accuracy could improve with longer training sessions**, which would require **purchasing additional computing units**.  

---

## **YOLOv8 Model Description**

### **Model Choice:**
The **YOLOv8 model** was selected due to its **real-time object detection capabilities**.  
Different YOLOv8 variants (**n, s, m, l, x**) have been used in similar projects like **strawberry ripeness detection** and **pedestrian recognition in urban areas**.

### **Model Training:**
```python
from ultralytics import YOLO
import os
import shutil

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=os.path.join(dataset.location, "data.yaml"),  # Dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    name="plant_disease_training"  # Project name
)
```

### **Validation Results:**

| Model | Precision | Recall | mAP@50 | F1-Score |
|--------|------------|------------|-----------|------------|
| **YOLOv8 (strawberry ripeness detection)** | 0.802 | 0.773 | 0.809 | 0.786 |
| **YOLOv8 (pedestrian detection)** | 0.832 | 0.87 | 0.874 | 0.850 |
| **Our Model** | **0.595** | **0.484** | **0.55** | **0.53** |

### **Performance Insights:**
- **Larger and more diverse datasets improve model accuracy.**  
- **Extended training sessions could enhance model performance.**  
- **Using a different YOLOv8 variant might yield better results.**  

---

## **User Instructions**
To interact with the model:
1. **Open** `Interface.ipynb` and run the script.  
2. **Provide the model weight file** located at:  
   ```
   Project_Image_Processing/model_detection/Pretrained_Model/weights
   ```
3. **Upload an image for detection.**  
4. **The program will perform detection using the model.**  

---

## **References**
1. **Gamani, A.-R. A., Arhin, I., & Asamoah, A. K. (2024)** – *Performance Evaluation of YOLOv8 Model Configurations for Instance Segmentation of Strawberry Fruit Development Stages.* [arXiv](https://arxiv.org/html/2408.05661v1).  
2. **Björklund, T. and Jonsson, F. (2024)** – *Analysis of Deep Learning in Autonomous Systems.* [KTH](https://kth.diva-portal.org/smash/get/diva2:1778368/FULLTEXT01.pdf).  
3. **Ultralytics Documentation** – [YOLOv8 Framework](https://docs.ultralytics.com/).  
4. **Roboflow Dataset** – [Plants Diseases Detection](https://universe.roboflow.com/vit-ll00j/plants-diseases-detection-and-classification-nf9in).  

