import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset

from helper_code import bandpass_filter

class ECGDataset(Dataset):
    """
    ECG dataset class that loads the signal from folder, gets the correct segment (first or second),
    passes the signal through a bandpass filter and returns it along with the encoded caption
    (caption encoded not here but before and added as a column of df) or the target diagnostic class.

    The 'mode' parameter controls the dataset mode and if the returned target is a caption or a diagnostic class
    The 'test_mode' parameter sets a mode in which only the ECG is returned

    df needs to have 'filename_hr' column and 'segment' column
    """

    def __len__(self) -> int:
        return len(self.df['filename_hr'])

    def pad_tokens(self, encoded_caption):
        """
        padding to the encoded caption
        Note: It modifies the encoded caption!
        """
        tokens = encoded_caption
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(1)  # mask is zero where we are out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.encoded_ecg_length), mask),
                         dim=0)  # adding prefix mask so that it is of length embedding_dim + max token seq len
        return tokens, mask

    def __getitem__(self, item: int):
        filename = self.df['filename_hr'].iloc[item]
        segment = self.df['segment'].iloc[item]

        # Load ecg
        ecg_element = wfdb.rdsamp(self.basepath + filename)
        if segment == 1:
            ecg_element = ecg_element[0][2500:, :]
        else:
            ecg_element = ecg_element[0][:2500, :]
        # Pass ecg through bandpass filter
        if self.bandpass:
            ecg_element = bandpass_filter(np.swapaxes(ecg_element, 0, 1))
        else:
            ecg_element = np.swapaxes(ecg_element, 0, 1)
        #ecg_element = np.expand_dims(ecg_element, axis=0)  # Shape is 1,5000,12 or 1,2500,12

        if self.test_mode:
            # ecg_element = np.expand_dims(ecg_element, axis=0)  # Shape is 1,5000,12 or 1,2500,12
            return ecg_element, self.df['report'].iloc[item], filename
        elif self.mode == 'encoder':
            return ecg_element, self.df['encoder_target'].iloc[item]
        else:
            report = self.df['report'].iloc[item]
            # Add <SEP> and <END> tokens
            sep = torch.tensor(self.tokenizer.encode("<SEP>"), dtype=torch.int64)
            encoded_caption = torch.tensor(self.tokenizer.encode(report), dtype=torch.int64)
            encoded_caption = torch.cat((sep, encoded_caption), dim=0)
            end = torch.tensor(self.tokenizer.encode("<END>"), dtype=torch.int64)
            encoded_caption = torch.cat((encoded_caption, end), dim=0)
            # get tokens and mask
            tokens, mask = self.pad_tokens(encoded_caption)
            #return encoded_caption, mask, ecg_element, report
            return tokens, mask, ecg_element, report

    def set_test_mode(self, test_mode: bool):
        self.test_mode = test_mode
        print(f"Test mode set to: {self.test_mode}")

    def set_mode(self, mode: str):
        if mode not in self.mode_list:
            raise ValueError(f"Mode not in mode list: {self.mode_list}")
        self.mode = mode

    def __init__(self, df: pd.DataFrame, basepath: str, max_seq_len: int = 100, bandpass: bool = True,
                 test_mode: bool = False, mode: str = 'both', tokenizer = None, encoded_ecg_length=100):
        self.df = df
        self.max_seq_len = max_seq_len  # max sequence length of encoded captions
        self.basepath = basepath
        self.bandpass = bandpass
        self.test_mode = test_mode
        self.mode_list = ['encoder', 'decoder', 'both']
        self.mode = mode
        self.tokenizer = tokenizer
        self.encoded_ecg_length = encoded_ecg_length # placeholder
        if mode not in self.mode_list:
            raise ValueError(f"Mode not in mode list: {self.mode_list}")
        if mode != 'encoder' and tokenizer is None:
            raise (f"Need tokenizer!")

        'add checks that if mode is encoder, encoder_target col is present, ' \
        'while if mode is decoder encoded_caption is present'

    def set_encoded_ecg_length(self, encoded_ecg_length):
        self.encoded_ecg_length = encoded_ecg_length
