import numpy as np
import torch
import wfdb
from torch.nn import functional as nnf

from helper_code import bandpass_filter

def generate(model,
             tokenizer,
             path,
             top_p=0.8,
             entry_length=67,  # maximum number of words CHANGE
             temperature=1.,
             stop_token: str = "<END>"):
    device = next(model.parameters()).device

    segment = 0,
    embed = None,  # embedded ECG (ECG passed through CNN)
    entry_length = 67  # maximum number of words CHANGE
    bandpass = True

    # Load Ecg
    ecg_element = wfdb.rdsamp(path)
    if segment == 1:
        ecg_element = ecg_element[0][2500:, :]
    else:
        ecg_element = ecg_element[0][:2500, :]
    # Pass ecg through bandpass filter
    if bandpass:
        ecg_element = bandpass_filter(np.swapaxes(ecg_element, 0, 1))
    else:
        ecg_element = np.swapaxes(ecg_element, 0, 1)

    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    ecg_element = torch.tensor(ecg_element, dtype=torch.float32)
    torch.tensor(ecg_element).to(device, dtype=torch.float32)
    # encode ECG and pass through MLP in decoder
    _, encoded_ecg_data_element = model.encoder(torch.unsqueeze(ecg_element, 0))
    encoded_ecg_data_element = encoded_ecg_data_element.view(-1, model.decoder.encoded_ecg_length,
                                                             model.decoder.encoded_ecg_size)
    encoded_ecg_data_element = model.decoder.ecg_project(encoded_ecg_data_element).view(-1,
                                                                                        model.decoder.encoded_ecg_length,
                                                                                        model.decoder.gpt_embedding_size)
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


def upload_generate(model,
             tokenizer,
             path,
             top_p=0.8,
             entry_length=67,  # maximum number of words CHANGE
             temperature=1.,
             stop_token: str = "<END>",
            file = None):
    device = next(model.parameters()).device

    segment = 0,
    embed = None,  # embedded ECG (ECG passed through CNN)
    entry_length = 67  # maximum number of words CHANGE
    bandpass = True

    # Load Ecg
    if path is not None:
        ecg_element = wfdb.rdsamp(path)
        if segment == 1:
            ecg_element = ecg_element[0][2500:, :]
        else:
            ecg_element = ecg_element[0][:2500, :]
        # Pass ecg through bandpass filter
        if bandpass:
            ecg_element = bandpass_filter(np.swapaxes(ecg_element, 0, 1))
        else:
            ecg_element = np.swapaxes(ecg_element, 0, 1)
    else:
        ecg_element = file

    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    ecg_element = torch.tensor(ecg_element, dtype=torch.float32)
    torch.tensor(ecg_element).to(device, dtype=torch.float32)
    # encode ECG and pass through MLP in decoder
    _, encoded_ecg_data_element = model.encoder(torch.unsqueeze(ecg_element, 0))
    encoded_ecg_data_element = encoded_ecg_data_element.view(-1, model.decoder.encoded_ecg_length,
                                                             model.decoder.encoded_ecg_size)
    encoded_ecg_data_element = model.decoder.ecg_project(encoded_ecg_data_element).view(-1,
                                                                                        model.decoder.encoded_ecg_length,
                                                                                        model.decoder.gpt_embedding_size)
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