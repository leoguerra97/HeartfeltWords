import torch
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn import preprocessing
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5Tokenizer

from config import INPUT_SHAPE, SEED, SEQ_LENGTH, BATCH_SIZE, SCALE_MULT, NET_FILTER_SIZE, NET_SEQ_LEN, KERNEL_SIZE, \
    DROPOUT_RATE, \
    LR, LR_FACTOR, MIN_LR, PATIENCE
from decoder_model import FullModel, CaptionModel
from ecg_dataset import ECGDataset
from encoder_model import PTResNet1d
from helper_code import load_ptbxl_data_paths, correct_reports, open_reports, encode_target, \
    split_data, str2bool


def train(model, ep, train_dataset, encoded_ecg_length, batch_size):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_bar = tqdm(initial=0, leave=True, total=len(train_dataloader),
                     desc=train_desc.format(ep, 0, 0), position=0)

    for idx, (tokens, mask, ecg_element, _) in enumerate(train_dataloader):
        model.zero_grad()

        tokens, mask, prefix = tokens.to(device), mask.to(device), ecg_element.to(device, dtype=torch.float32)
        outputs = model(tokens, ecg_element.float(), mask)
        logits = outputs.logits[:, encoded_ecg_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        loss.backward()

        optimizer.step()

        bs = len(tokens)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs

        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(model, ep, valid_dataset, encoded_ecg_length, batch_size):
    model.zero_grad()
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_dataloader),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for idx, (tokens, mask, ecg_element, _) in enumerate(valid_dataloader):

        tokens, mask, prefix = tokens.to(device), mask.to(device), ecg_element.to(device, dtype=torch.float32)
        outputs = model(tokens, ecg_element.float(), mask)
        logits = outputs.logits[:, encoded_ecg_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        bs = len(ecg_element)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs

        # Print result
        eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
        eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


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
                        const=True, default="results/decoder/",
                        help="encoder output path")
    parser.add_argument("--output_name", type=str, nargs='?',
                        const=True, default="decoder_model",
                        help="model name")
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
                        const=True, default=1,
                        help="train epochs")
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use gpu")
    parser.add_argument("--encoder_model_path", type=str, nargs='?',
                        const=True, default='encoder_name/model_0.pth')
    parser.add_argument("--decoder_model_path", type=str, nargs='?',
                        const=True, default='_')

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
    encoder_model_path = './results/encoder/' + args.encoder_model_path
    decoder_model_path = args.decoder_model_path

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
    lr = LR
    patience = PATIENCE
    min_lr = MIN_LR
    lr_factor = LR_FACTOR

    torch.manual_seed(seed)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    tqdm.write("Loading Data...")
    # Load Data Paths and reports
    df = load_ptbxl(basepath, reports_path, reports_language, remove_sinus, drop_no_reports)

    tqdm.write("Encoding Target...")

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
    tqdm.write("Creating train, validation and test splits...")
    train_test_split_folds = split_data(df['diagnostic_superclass'], encode_target(df['diagnostic_superclass'].astype(str)))
    train_validation_order_array = train_test_split_folds[0][0]  # at the moment just need first fold
    test_order_array = train_test_split_folds[0][1]
    train_and_valid_df = df.iloc[train_validation_order_array]
    train_and_valid_target = target[train_validation_order_array]

    train_valid_folds = split_data(train_and_valid_df['diagnostic_superclass'], encode_target(train_and_valid_df['diagnostic_superclass'].astype(str)))
    train_order_array = train_valid_folds[0][0]  # at the moment just need first fold
    valid_order_array = train_valid_folds[0][1]

    test_df = df.iloc[test_order_array]
    train_df = train_and_valid_df.iloc[train_order_array]
    valid_df = train_and_valid_df.iloc[valid_order_array]

    tqdm.write("Creating Tokenizer...")
    # Tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #gpt_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    gpt_tokenizer.add_tokens("<SEP>")
    gpt_tokenizer.add_tokens("<END>")

    tqdm.write("Defining model...")
    # Encoder (from saved File)
    tqdm.write("Loading encoder model...")
    encoder_model = PTResNet1d(input_dim=input_shape,
                               blocks_dim=list(zip(net_filter_size, net_seq_len)),
                               n_classes=n_classes,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate)
    encoder_checkpoint = torch.load(encoder_model_path)
    encoder_model.load_state_dict(encoder_checkpoint['model'])

    # Output Shapes
    test_input = torch.unsqueeze(torch.rand(*(np.array(input_shape))), dim=0)
    encoder_output_shape = encoder_model(test_input)[1].shape
    print('Encoder output size:')
    print(encoder_output_shape)
    print('NB: None dimension is batch_size')
    encoded_ecg_length = encoder_output_shape[2]  # 2500 ---> 5
    encoded_ecg_size = encoder_output_shape[1]  # 12 ---> 1024

    num_steps = len(train_df['report']) // batch_size

    # Create Decoder and Full model
    tqdm.write("Creating Caption Model")
    decoder_model = CaptionModel(tokenizer=gpt_tokenizer,
                                 encoded_ecg_length=encoded_ecg_length,
                                 encoded_ecg_size=encoded_ecg_size)
    # Load Decoder
    if decoder_model_path != '_':
        decoder_model.load_state_dict(torch.load(decoder_model_path))

    model = FullModel(caption_model=decoder_model, encoder_model=encoder_model, tokenizer=gpt_tokenizer)
    model.to(device=device)

    tqdm.write("Creating Datasets...")
    train_dataset = ECGDataset(train_df, basepath, test_mode=False, mode='decoder', tokenizer=gpt_tokenizer, encoded_ecg_length=encoded_ecg_length)
    valid_dataset = ECGDataset(valid_df, basepath, test_mode=False, mode='decoder', tokenizer=gpt_tokenizer, encoded_ecg_length=encoded_ecg_length)
    test_dataset = ECGDataset(test_df, basepath, test_mode=True, mode='decoder', tokenizer=gpt_tokenizer, encoded_ecg_length=encoded_ecg_length)

    tqdm.write("Defining optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tqdm.write("Defining scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience,
                                                     min_lr=lr_factor * min_lr,
                                                     factor=lr_factor)

    tqdm.write("Training...")
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.makedirs(os.path.join(output_path, output_name))

    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])

    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    unfreeze_after = 5
    for ep in range(start_epoch, epochs):
        # Unfreeze encoder parameters
        if ep > unfreeze_after:
            for param in model.encoder.parameters():
                param.requires_grad = True

        train_loss = train(model, ep, train_dataset, encoded_ecg_length=encoded_ecg_length, batch_size=batch_size)
        valid_loss = eval(model, ep, valid_dataset, encoded_ecg_length=encoded_ecg_length, batch_size=batch_size)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(output_path, output_name, f'model_{ep}.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < min_lr:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                   '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                   .format(ep, train_loss, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(output_path, output_name, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)

    tqdm.write("Done!")
