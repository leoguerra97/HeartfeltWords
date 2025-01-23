import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from flask import Flask, render_template, request, jsonify, Response
from flask_api import status
from transformers import GPT2Tokenizer

from config import INPUT_SHAPE, NET_FILTER_SIZE, NET_SEQ_LEN, KERNEL_SIZE, DROPOUT_RATE
from decoder_model import FullModel, CaptionModel
from demo_generate import generate, upload_generate
from encoder_model import PTResNet1d
from encoder_train import load_ptbxl
import cv2

warnings.simplefilter(action='ignore', category=FutureWarning)

# Flask constructor takes the name of
# current module (__name__) as argument.

app = Flask(__name__)

BASEPATH = '/Users/Leonardo/Desktop/Thesis/Thesis_Project/'
ECG_FOLDER = 'ptb_xl/'
REPORTS_PATH = 'en_df_round4.csv'
# NOTE: If I change JSON_FILE_TO_LOAD run save_new_images() in main to add new images to image folder
JSON_FILE_TO_LOAD = 'results/resnet_79_multilabel/pred_results_json.json'
IMAGE_FOLDER_PATH = 'demo/static/images/'
GPT_2_MODEL_PATH = BASEPATH + 'results/full_model/full_model_1/model_last.pth'
UPLOAD_DIR = 'demo/static/uploads/'


cols = 4
rows = 6
loaded_models = []
supported_models = ['gpt2', 'T5']
active_model = 'gpt2'

upload_response = []
upload_response_element = None


def save_new_images(data):
    count = 0
    for element in data:
        count += 1
        image_name = BASEPATH + IMAGE_FOLDER_PATH + str(data[element]['filename'][0]).replace('/', '_') + '.png'
        if image_name not in os.listdir(BASEPATH + IMAGE_FOLDER_PATH):
            record = wfdb.rdrecord(BASEPATH + ECG_FOLDER + str(data[element]['filename'][0]))
            image = wfdb.plot_wfdb(record, figsize=(6, 9), return_fig=True)
            image.savefig(image_name)
            plt.close(image)
            print(f'img {count} saved\n')


def get_session_data(data):
    session_data = []
    range_of_data = range(0, len(list(data.keys())) - 1)  # len of df
    for el in range(0, 20):
        chosen = random.choice(range_of_data)
        sample = data.get(str(chosen))
        sample['ecg_name'] = sample['ecg_name'].replace('/', '_')
        session_data.append(sample)
    return session_data


def get_session_upload(data):
    session_upload = []
    return session_upload


def load_gpt2_model():
    print('Loading GPT2 Model...')
    input_shape = INPUT_SHAPE
    net_filter_size = NET_FILTER_SIZE  # filter size in resnet layers
    net_seq_len = NET_SEQ_LEN  # number of samples per resnet layer
    kernel_size = KERNEL_SIZE  # 'kernel size in convolutional layers
    dropout_rate = DROPOUT_RATE
    n_classes = 71
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.add_tokens("<SEP>")
    gpt_tokenizer.add_tokens("<END>")

    # Create Encoder
    encoder_model = PTResNet1d(input_dim=input_shape, blocks_dim=list(zip(net_filter_size, net_seq_len)), n_classes=n_classes, kernel_size=kernel_size, dropout_rate=dropout_rate)

    # Compute decoder shape
    test_input = torch.unsqueeze(torch.rand(*(np.array(input_shape))), dim=0)
    encoder_output_shape = encoder_model(test_input)[1].shape
    print('Encoder output size:')
    print(encoder_output_shape)
    encoded_ecg_length = encoder_output_shape[2]  # 2500 ---> 5
    encoded_ecg_size = encoder_output_shape[1]  # 12 ---> 1024

    # Create Decoder
    decoder_model = CaptionModel(tokenizer=gpt_tokenizer, encoded_ecg_length=encoded_ecg_length, encoded_ecg_size=encoded_ecg_size)

    # Create Full Model
    model = FullModel(caption_model=decoder_model, encoder_model=encoder_model, tokenizer=gpt_tokenizer)
    model.to(device=device)
    model_checkpoint = torch.load(GPT_2_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint['model'])
    model.eval()
    print('GPT2 Model Loaded!')
    return model, gpt_tokenizer


def load_t5_model():
    # TODO: implement
    change_model('gpt2')
    pass


def load_reports():
    reports_w_images = os.listdir(BASEPATH + IMAGE_FOLDER_PATH)
    reports_w_images = [x.replace('_', '/', 2).replace('.png', '') for x in reports_w_images]
    df = load_ptbxl(BASEPATH + ECG_FOLDER, BASEPATH, 'en')
    df = df[df['filename_hr'].isin(reports_w_images)]
    ecg_df = df[['filename_hr', 'report', 'diagnostic_superclass']]
    return ecg_df


def change_model(model: str):
    global active_model
    active_model = model
    if model not in loaded_models:
        if model == 'gpt2':
            loaded_models.append(model)
            return load_gpt2_model()
        elif model == 'T5':
            loaded_models.append(model)
            return load_t5_model()
        else:
            raise ValueError('Model not supported')


@app.route('/')
def index():
    print(os.getcwd())
    session_data = get_session_data(data)
    return render_template('ecg_images.html', page_data=session_data, name='LEO')


@app.route('/upload')
def upload():
    session_upload = get_session_upload(data)
    return render_template('ecg_upload.html', page_data=session_upload, name='LEO')


@app.route('/predict', methods=['POST'])
def api_predict():
    input_json = request.get_json(force=False)
    app.logger.debug(
        'Endpoint /predict received request {}'.format(input_json))
    R = dict()
    R['ecg_data'] = input_json['ecg_data']

    try:
        # get prediction
        if active_model == 'gpt2':
            model = full_gpt2_model
            tokenizer = gpt_tokenizer
        elif active_model == 't5':
            # model = full_t5_model
            # tokenizer = t5_tokenizer
            model = full_gpt2_model
            tokenizer = gpt_tokenizer
        else:
            raise ValueError('Model not supported')

        pred = generate(model, tokenizer, path=BASEPATH + ECG_FOLDER + R['ecg_data'].replace('_', '/', 2))
        R['prediction'] = 'Prediction: ' + pred

        # Get reference
        reference = ecg_df[ecg_df['filename_hr'] == R['ecg_data'].replace('_', '/', 2)]['report'].values[0]
        R['real'] = 'Real: ' + reference

        ### BUILD RESPONSE ###
        R['message'] = 'OK'
        R['status'] = status.HTTP_200_OK
        return jsonify(R), R['status']
    except Exception as e:
        print(e)
        R['message'] = 'Internal server error: {}'.format(e)
        R['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
        return jsonify(R), R['status']


@app.route('/upfile', methods=['GET', 'POST'])
def upfile():
    global upload_response_element
    #global upload_response
    files = request.files.getlist('file')
    print(files)
    name = files[0].filename.split('.')[0]
    if len(files) != 2:
        return 'Error: send record composed of .dat file and .hea file simultaneously'
    for file in files:
        #if !(file.filename.endswith('.hea') or file.filename.endswith('.dat')):
         #   return 'Error: send record composed of .dat file and .hea file simultaneously'
        if name not in file.filename:
            return 'Error: file names do not match'
    for file in files:
        if file not in os.listdir(BASEPATH + UPLOAD_DIR):
            file.save(BASEPATH + UPLOAD_DIR + file.filename)
    print('Files received, generating prediction')
    R = dict()
    try:
        # get prediction
        R['prediction'] = upload_generate(full_gpt2_model, gpt_tokenizer, path=BASEPATH + UPLOAD_DIR + name)
        # get image
        ecg_data = wfdb.rdrecord(BASEPATH + UPLOAD_DIR + name)
        image = wfdb.plot_wfdb(ecg_data, figsize=(6, 9), return_fig=True)
        img_name = name + '.png'
        img_path = BASEPATH + UPLOAD_DIR + img_name
        image.savefig(img_path)
        plt.close(image)
        print(f'img {name} saved')
        R['img_path'] = '/static/uploads/' + img_name
        ### BUILD RESPONSE ###
        R['message'] = 'OK'
        R['status'] = status.HTTP_200_OK
        #upload_response.append(R)
        upload_response_element = R
        return jsonify(R), R['status']

    except Exception as e:
        print(e)
        R['message'] = 'Internal server error: {}'.format(e)
        R['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
        return jsonify(R), R['status']


@app.route('/get_upload_prediction', methods=['GET'])
def get_upload_prediction():
    #R = upload_response[-1]
    R = upload_response_element
    #print(upload_response_element)
    return jsonify(R)

@app.route('/files', methods=['GET'])
def get_files():
    data_files = os.listdir(BASEPATH + IMAGE_FOLDER_PATH)
    labels = [ecg_df[ecg_df['filename_hr'] == x.replace('_', '/', 2).replace('.png', '')]['diagnostic_superclass'].values[0] for x in data_files]
    R = dict()
    R['files'] = data_files
    R['labels'] = labels
    # print(data_files)
    response = app.response_class(
        # response=json.dumps(data_files),
        response=json.dumps(R),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/change_model', methods=['POST'])
def change_model():
    model = request.get_json(force=False)
    return model
    # if model not in supported_models:
    #     return 'Model not supported', 400
    # else:
    #     change_model(model)


# main driver function
if __name__ == '__main__':
    file = open(BASEPATH + JSON_FILE_TO_LOAD)
    data = json.load(file)
    ecg_df = load_reports()
    active_model = 'gpt2'
    full_gpt2_model, gpt_tokenizer = load_gpt2_model()
    # full_t5_model, t5_tokenizer = load_t5_model()
    # save_new_images(data)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=8000)

'''
command to send predict request from terminal using a path
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"ecg_data": "records500/01000/01351_hr"}' \
http://localhost:6006/predict
'''
