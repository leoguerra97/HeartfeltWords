# Thesis
Thesis project: \
Code repository for the execution of an ECG captioning model.\
The model is composed of an Encoder and a Decoder. The encoder is a pytorch model and the decoder is a Language model, at the moment we are using a GPT-2 model. These are connected by a linear layer that converts the encoder output to the language model's embedding size. 

The encoder is a CNN model that takes as input a 12-lead ECG signal and outputs a vector of 768 features (after connecting linear layer). \
The decoder is a GPT-2 model that takes as input the encoder output and outputs a caption.

## Installation
Install the requirements using the following command: \
`pip install -r requirements.txt` \
There may be some requirements that are not in the requirements.txt file. If you encounter any errors, install the missing requirements using pip.

## Usage
- Demo: \
There is a demo folder which contains the demo. To run the demo launch the server in the demo.py file
The following variables must be set:
  - BASEPATH = Base path of the project
  - ECG_FOLDER = folder path containing the dataset ECGs from basepath
  - REPORTS_PATH = path containing the translated reports from basepath
  - JSON_FILE_TO_LOAD = path to load a json file containing the ECGs, their reports and their captions to display in the "homescreen" of the demo
  - IMAGE_FOLDER_PATH = path to the folder containing the images to display in the "homescreen" of the demo (images are generated and saved here from the JSON file automatically)
  - GPT_2_MODEL_PATH = path to the FULL (Encoder + Decoder) model
  - UPLOAD_DIR = folder containing the ECGs uploaded for captioning
    

- Training: \
  To train the model, you have to train the encoder and the decoder models.
  The encoder model is trained on the classification task as specified in the thesis.\
  The target of the encoder can be single target between 5 superclasses or multi target between 71 classes. Training is performed using the parameters specified in the config file and the encoder model weights are then saved into the results folder.  \
  After having saved encoder model weights, decoder training can be performed.
  The training is performed using the parameters specified in the config file and the FULL MODEL weights are then saved into the results folder. \
  
- Files:
    - config.py: contains the configuration parameters for the model
    - demo.py: contains the code relative to the demo. 
    - encoder_model.py: contains the encoder model.
    - decoder_model.py: contains the coder for decoder (caption model) and the Full model.
    - encoder_train.py: contains the script to train the encoder model.
    - helper_code.py: contains helper functions to load dataset and set filters.
    - ecg_dataset.py: contains Dataset Class which is used to load and prepare the dataset.
    - model_evaluate.py: contains the script to evaluate the model on the test set.
    - requirements.txt: contains the required packages.