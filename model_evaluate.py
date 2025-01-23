import torch
import argparse
import os

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn import preprocessing
from torch.nn import functional as nnf
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import INPUT_SHAPE, SEED, SEQ_LENGTH, BATCH_SIZE, SCALE_MULT, NET_FILTER_SIZE, NET_SEQ_LEN, KERNEL_SIZE, \
    DROPOUT_RATE
from decoder_model import FullModel, CaptionModel
from ecg_dataset import ECGDataset
from encoder_model import PTResNet1d
from helper_code import load_ptbxl_data_paths, correct_reports, open_reports, encode_target, \
    split_data, str2bool


def generate(model, tokenizer, ecg_element,
            entry_length = 67,  # maximum number of words CHANGE
            top_p = 0.8,
            temperature = 1.,
            stop_token : str = "<END>"):

    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    ecg_element = torch.tensor(ecg_element, dtype=torch.float32)
    torch.tensor(ecg_element).to(device, dtype=torch.float32)
    # encode ECG and pass through MLP in decoder
    _, encoded_ecg_data_element = model.encoder(torch.unsqueeze(ecg_element, 0))
    encoded_ecg_data_element = encoded_ecg_data_element.view(-1, model.decoder.encoded_ecg_length, model.decoder.encoded_ecg_size)
    encoded_ecg_data_element = model.decoder.ecg_project(encoded_ecg_data_element).view(-1, model.decoder.encoded_ecg_length, model.decoder.gpt_embedding_size)
    # Add <SEP> token
    sep = torch.tensor(tokenizer.encode('<SEP>'), dtype=torch.int64).to(device)
    sep_embedding = model.decoder.gpt.transformer.wte(sep).unsqueeze(0)

    encoded_ecg_data_element = torch.cat((encoded_ecg_data_element, sep_embedding), dim=1)
    encoded_ecg_data_element.to(device)
    tokens = None
    for i in range(entry_length):
        outputs = model.decoder.gpt(inputs_embeds=encoded_ecg_data_element)
        logits = outputs.logits
        # print('logits shape:')
        # print(logits.shape)
        logits = logits / (temperature if temperature > 0 else 1.0)
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            nnf.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        # print('next token and next_token_embed shape:')
        # print(next_token.shape)
        next_token_embed = model.decoder.gpt.transformer.wte(next_token)
        if tokens is None:  # first token
            tokens = next_token
        else:  # subsequent tokens
            tokens = torch.cat((tokens, next_token), dim=1)
        encoded_ecg_data_element = torch.cat((encoded_ecg_data_element, next_token_embed), dim=1)
        if stop_token_index == next_token.item():
            break

        # output_list = list(tokens.squeeze().cpu().numpy())
        # output_text = tokenizer.decode(output_list)
        # print(tokenizer.decode(next_token_embed.squeeze().cpu().numpy()))
    if tokens is None:
        output_text = '_'
    else:
        if len(tokens[0]) == 1:
            output_list = list(tokens.reshape((-1)).cpu().numpy())
        else:
            output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)

    return output_text


def eval_test(model, test_dataloader, tokenizer, encoded_ecg_length):
    model.eval()
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(test_dataloader),
                    desc=eval_desc.format(0, 0), position=0)
    caption_list = []
    report_list = []
    filenames = []
    bleu_1 = []
    bleu_4 = []
    for idx, (ecg_element, report, filename) in enumerate(test_dataloader):
        output = generate(model, tokenizer, ecg_element)
        caption_list.append(output)
        report_list.append(report)

        output = output[:-6]  # Remove end token
        out_list = output.split()
        caption_list = report.split()
        bleu_1_score = sentence_bleu([caption_list], out_list, weights=(1, 0, 0, 0))
        bleu_4_score = sentence_bleu([caption_list], out_list, smoothing_function=SmoothingFunction().method4,
                                     weights=(0.25, 0.25, 0.25, 0.25))
        filenames.append(filename)
        bleu_1.append(bleu_1_score)
        bleu_4.append(bleu_4_score)

        eval_bar.desc = eval_desc.format(idx, 0)
        eval_bar.update(1)
    eval_bar.close()
    return caption_list, report_list, filenames, bleu_1, bleu_4


def load_ptbxl(basepath, reports_path, report_language, remove_sinus=False, drop_no_reports=True):
    paths_df = load_ptbxl_data_paths(basepath)
    paths_df.rename(columns={'report': 'original_report'}, inplace=True)
    reports_df = open_reports(reports_path, report_language)
    paths_df.drop(["patient_id", "age", "sex", "height", "weight", "recording_date", "strat_fold"],
                  axis=1,
                  inplace=True)

    # Inner Join between dataframes
    df = pd.concat([paths_df, reports_df], axis=1, join="inner")

    # Correct "no reports"
    print('Removing following reports:')
    df = correct_reports(df)

    if drop_no_reports:
        df.drop(df.index[df['report'] == 'no_report'].tolist(), axis=0, inplace=True)

    # Remove unwanted text
    if remove_sinus:
        print("Removing sinus rhythm caption part")
        df['report'] = df['report'].str.replace('sinus ', '')
        df['report'] = df['report'].str.replace('rhythm', '')
        df['report'] = df['report'].str.replace('sinusrhythm', '')

    df['report'] = df['report'].str.replace('sinusrhythm', 'sinus rhythm')
    df['report'] = df['report'].str.replace('ekg', 'ecg')
    df['report'] = df['report'].str.lstrip()

    # Add labels column
    df['labels'] = df.scp_codes.apply(lambda x: list(dict.keys(x)))
    print(df['labels'])
    df['segment'] = 0

    return df


if __name__ == "__main__":

    # Set script behaviour
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, nargs='?',
                        const=True, default="./ptb_xl/",
                        help="ptbxl basepath")
    parser.add_argument("--reports_path", type=str, nargs='?',
                        const=True, default="./",
                        help="reports csv path")
    parser.add_argument("--reports_language", type=str, nargs='?',
                        const=True, default='en',
                        help="reports language")
    parser.add_argument("--output_path", type=str, nargs='?',
                        const=True, default="results/encoder/",
                        help="encoder output path")
    parser.add_argument("--output_name", type=str, nargs='?',
                        const=True, default="encoder_name",
                        help="encoder name")
    parser.add_argument("--remove_sinus", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Remove 'sinus rhythm' from captions")
    parser.add_argument("--drop_no_reports", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="drop None reports")
    parser.add_argument("--multilabel", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Multilabel or multiclass")
    parser.add_argument("--epochs", type=int, nargs='?',
                        const=True, default=10,
                        help="train epochs")
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use gpu")
    parser.add_argument("--model_path", type=str, nargs='?',
                        const=True, default='results/encoder/model_name/model_0.pth')

    args = parser.parse_args()

    basepath = args.basepath
    reports_path = args.reports_path
    reports_language = args.reports_language
    output_path = args.output_path
    output_name = args.output_name
    remove_sinus = args.remove_sinus
    drop_no_reports = args.drop_no_reports
    multilabel = args.multilabel
    epochs = args.epochs
    model_path = args.model_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # fixed parameters
    input_shape = INPUT_SHAPE
    seed = SEED
    seq_lenght = SEQ_LENGTH
    batch_size = BATCH_SIZE
    scale_mult = SCALE_MULT  # check
    net_filter_size = NET_FILTER_SIZE  # filter size in resnet layers
    net_seq_len = NET_SEQ_LEN  # number of samples per resnet layer
    kernel_size = KERNEL_SIZE  # 'kernel size in convolutional layers
    dropout_rate = DROPOUT_RATE


    torch.manual_seed(seed)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    tqdm.write("Loading data...")
    # Load Data Paths and reports
    df = load_ptbxl(basepath, reports_path, reports_language, remove_sinus, drop_no_reports)

    tqdm.write("Encoding Captions and Target...")

    # Encoder Target
    # Still compute classification target to create stratified splits
    if multilabel:
        agg_df = pd.read_csv(basepath + 'scp_statements.csv')  # csv with aggregate diagnostic
        agg_df.rename(columns={"Unnamed: 0": "class"}, inplace=True)
        labels = agg_df['class'].astype(str)
        label_enc = preprocessing.MultiLabelBinarizer()
        label_enc.fit([np.unique(labels)])
        target = label_enc.transform(df['labels'])
        n_classes = len(label_enc.classes_)
        print('\nnumber of classes:', n_classes)
    else:
        target = encode_target(df['diagnostic_superclass'].astype(str))
        n_classes = len(np.unique(target))

    df['encoder_target'] = [x for x in target]

    # Train, validation and test split
    train_test_split_folds = split_data(df['diagnostic_superclass'],
                                        encode_target(df['diagnostic_superclass'].astype(str)))
    train_validation_order_array = train_test_split_folds[0][0]  # at the moment just need first fold
    test_order_array = train_test_split_folds[0][1]

    test_df = df.iloc[test_order_array]

    # Tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.add_tokens("<SEP>")
    gpt_tokenizer.add_tokens("<END>")

    tqdm.write("Defining model...")
    # Load encoder model
    # Encoder
    encoder_model = PTResNet1d(input_dim=input_shape,
                               blocks_dim=list(zip(net_filter_size, net_seq_len)),
                               n_classes=n_classes,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate)
    # encoder_checkpoint = torch.load(encoder_model_path)
    # encoder_model.load_state_dict(encoder_checkpoint['model'])

    # Output Shapes
    test_input = torch.unsqueeze(torch.rand(*(np.array(input_shape))), dim=0)
    encoder_output_shape = encoder_model(test_input)[1].shape
    print('Encoder output size:')
    print(encoder_output_shape)
    print('NB: None dimension is batch_size')
    encoded_ecg_length = encoder_output_shape[2]  # 2500 ---> 5
    encoded_ecg_size = encoder_output_shape[1]  # 12 ---> 1024

    # Create model
    tqdm.write("Creating Caption Model")
    decoder_model = CaptionModel(tokenizer=gpt_tokenizer,
                                 encoded_ecg_length=encoded_ecg_length,
                                 encoded_ecg_size=encoded_ecg_size)
    # Load Model

    # Load decoder --- NOT NEEDED BECAUSE WE LOAD FULL MODEL
    # decoder_checkpoint = torch.load(decoder_model_path)
    # decoder_model.load_state_dict(decoder_checkpoint['model'])

    model = FullModel(caption_model=decoder_model, encoder_model=encoder_model, tokenizer=gpt_tokenizer)
    model.to(device=device)
    model_checkpoint = torch.load(model_path)
    model.load_state_dict(model_checkpoint['model'])

    tqdm.write("Creating test dataloader dataset...")
    test_dataloader = ECGDataset(test_df, basepath, test_mode=True, mode='decoder', tokenizer=gpt_tokenizer)

    tqdm.write("Testing...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])

    caption_list, report_list, filenames, bleu_1, bleu_4 = eval_test(model, test_dataloader=test_dataloader,
                                                                     encoded_ecg_length=encoded_ecg_length,
                                                                     tokenizer=gpt_tokenizer)
    tqdm.write("Done!")

