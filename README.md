## The shipWatcher
This is my first Machine Learning(Computer Vision) project using **PyTorch**, and aims to detect unidentified personnel / vehicle approaching a warship at night.
This project is based on my personal experiences as a conscript in the Korean Navy, and aims to solve a problem that fellow crew and I suffered from.

## Description of project
In the Navy, Gangway Watch was one of the most dreaded task, where the Gangway was to be guarded 24/7 by 2 people in rotational shifts.
During the fully manual 8-hour duty, my crew and I would suffer from Alert Fatigue due to regular false alarms and impromptu patrols by officers on-duty.
This inefficient system would often cause those on-duty to be unable to detect actual threats when required.
I first started studying ML during my service, and I found it meaningful to end it with a project that helps those who sacrifice so much to protect our country to this day.

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
├── data/
│   ├── train/          # Training Images
│   │   ├── Empty/
│   │   ├── Person/
│   │   └── Vehicle/
│   └── val/            # Validation Images
│       ├── Empty/
│       ├── Person/
│       └── Vehicle/
├── models/             # Where 'shipWatcher.pth' will be saved
├── src/                # Source code (train.py, inference.py, data_preprocessing.py, tune_hyperparams.py, video.processing.py)
└── requirements.txt
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
