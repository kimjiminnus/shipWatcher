## The shipWatcher
This is my first Machine Learning(Computer Vision) project using **PyTorch**, and aims to detect unidentified personnel / vehicle approaching a warship at night.
This project is based on my personal experiences as a conscript in the Korean Navy, and aims to solve a problem that fellow crew and I suffered from.
Having started studying ML during my service, I find it meaningful to end it by helping those who protect our country to this day.

## Description of project
In the Navy, Gangway Watch is one of the most dreaded tasks, where 2 people guard the Gangway 24/7 in rotational shifts.
During the fully manual 8-hour shifts, my crew and I would suffer from Alert Fatigue due to constant false alarms and psychological strains.
This inefficient system was prone to human error, causing the inability to detect actual threats when it actually mattered most.
Thus, this project has the reduction of the False Positive rate as its top priority, and will be evaluated using ROC curves and AUC scores.

## Current Status: Training & Data Collection**
This project is currently in the **active development phase**. 
I am focusing on improving model accuracy by 
* Addressing domain gaps between web-scraped images and real-world video frames
* Training models with images of various weather conditions that a Naval Base is susceptible to due to its proximity to the sea.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/kimjiminnus/shipWatcher.git
cd shipWatcher
pip install -r requirements.txt
```

### 2. Local file organisation structure
```bash
shipWatcher/
├── data/               # DISCLAIMER: Image Datasets will only be uploaded once enough images that can appropriately capture the high-noise port environment are found
│   ├── train/          # Training Images
│   │   ├── Empty/
│   │   ├── Person/
│   │   └── Vehicle/
│   └── val/            # Validation Images
│       ├── Empty/
│       ├── Person/
│       └── Vehicle/
├──models/
│   └── shipWatcher.pth        # Contains state_dict of optimal model
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Logic for sorting and creating training & validation datasets       
│   ├── inference.py           # Script for testing model on user-selected images
│   ├── model_def.py           # get_shipWatcher(): ResNet-18 architecture definition 
│   ├── train.py               # Script for training, validating model from scratch using prepared data
│   ├── tune_hyperparams.py    # Script for training, validating model using various types of hyperparameters and values
│   ├── utils.py               # Contains device_configuration, image_transform, class_list
│   └── video_processing.py    # Script for using model to analyse videos / live feeds
└── requirements.txt    # List of necessary libraries (PyTorch, Pillow, etc.) 
```

### 3. Training shipwatcher from scratch & saving state_dict
```bash
python src/train.py
```

### 4. Testing your own image files on the shipWatcher 
```bash
python src/inference.py
```


## The Tech Stack
* **Language:** Python 3.x
* **Framework:** PyTorch
* **Computer Vision:** Torchvision, OpenCV
* **Model:** ResNet-18 with a customised 3-neuron output layer (Transfer Learning)

## Challenges I'm Solving
* **Data Scarcity:** Finding suitable datasets of images in a Naval Base proves to be challenging due to Operational Security(OPSEC)
* **Class Imbalance:** My "Empty" class is smaller than "Vehicle" or "Person." I am implementing **Weighted Random Sampling** to prevent model bias.
* **Accuracy:** Current accuracy is limited by background noise. I am working on **Hard Negative Mining** to reduce false positives on poles and shadows.

## 🚀 Future Roadmap
1. Once the model is trained with appropriate data and validated & tested on saved videos, adjust cv2.VideoCapture parameter to process live video feed.
2. Implement a Confusion Matrix for better error analysis.
3. Export to CoreML and Streamlit to share & deploy model.
