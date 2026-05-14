## The shipWatcher
Deep Learning(Computer Vision) project using **PyTorch**, and is an End-to-End Modular Pipeline that aims to detect and identify unidentified personnel / vehicle approaching a warship at night.
It is based on my personal experiences as a conscript in the Korean Navy, and aims to solve a problem that fellow crew and I suffered from.

## Description of project
In the Navy, Gangway Watch is one of the most dreaded tasks, where 2 people guard the Gangway 24/7 in rotational shifts.
During the fully manual 8-hour shifts, my crew and I would suffer from Alert Fatigue due to constant false alarms and psychological strains.
This inefficient system is prone to human error, causing the inability to detect actual threats when it actually mattered most.
Therefore, this project has the reduction of the False Positive Rate (FPR) as its top priority, and will be evaluated using ROC curves and AUC scores.

## Getting Started

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


## Tech Stack
* **Language:** Python 3.x
* **Framework:** PyTorch
* **Computer Vision:** Torchvision, OpenCV
* **Model:** ResNet-18 with a customised 3-neuron output layer (Transfer Learning)


