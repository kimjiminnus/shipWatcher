## The shipWatcher
The shipWatcher is a Computer Vision image classification project using **PyTorch**, with an end-to-end modular pipeline that detects and identifies unidentified personnel / vehicle approaching a warship at night.
It is based on my experiences as a conscript in the Korean Navy guarding the gangway during Watch Duty.

## Description of project
The port is a noisy environment where the surroundings aren't clear enough for human vision to identify objects accurately.
A pre-trained ResNet-18 Model is fine-tuned to receive frames as input and classify it into one of [Empty, Person, Vehicle].
This project aims to reduce the False Positive Rate (FPR) using ROC curves and AUC scores for evaluation 


### Project Structure
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

### Training shipwatcher from scratch & saving state_dict
```bash
python src/train.py
```

### Testing your own image files on the shipWatcher 
```bash
python src/inference.py
```


## Tech Stack
* **Language:** Python 3.x
* **Framework:** PyTorch
* **Computer Vision:** Torchvision, OpenCV
* **Model:** ResNet-18 with a customised 3-neuron output layer (Transfer Learning)


