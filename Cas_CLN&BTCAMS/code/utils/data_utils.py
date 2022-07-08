import re

import numpy as np
import torch
# from transformers import BertTokenizer
from transformers import MBartTokenizer

from code.utils import extract_chinese_and_punct

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()


def batch_gather(data: torch.Tensor, index: torch.Tensor):  #bert_en[bsize, seqlen, hsize]; start_index[bsize, 1]
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])
    index_encoder = torch.from_numpy(np.array(result)).to(data.device)

    return index_encoder  #返回index指定token的编码表示。[bsize, 1, hsize]

def batch_gather_sub(data: torch.Tensor, index: torch.Tensor, sub_len):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = torch.zeros(1, data.size()[-1]).to(data.device)
    # print(length, data.shape[0])
    for i in range(length):
        # if i >= 12:
        #     print('error', i,length, data.shape[0])
        #     continue
        sub_encoder = torch.from_numpy(t_data[i, t_index[i]:t_index[i]+sub_len[i], :]).to(data.device)
        # print('\nsub_encoder', sub_encoder.size(), sub_len[i])
        sub_encoder_sum = torch.cumsum(sub_encoder, 0)
        # print('\nsub_encoder_sum', sub_encoder_sum.size())
        sub_encoder_sum_last = sub_encoder_sum[-1, :]  #一维向量
        sub_average = sub_encoder_sum_last / sub_len[i]
        result = torch.cat((result, sub_average.unsqueeze(0)), 0)
    return result[1:, :].unsqueeze(1)

        # result.append(t_data[i, t_index[i]:t_index[i]+sub_len[i]-1, :])
    # sub_encoder = torch.from_numpy(np.array(result)).to(data.device)  #返回index指定的实体的编码表示。[bsize, sub_len, hsize]
    # print('\nsub_encoder.size()', sub_encoder.size(), length, sub_len)
    # sub_encoder_sum = torch.cumsum(sub_encoder, 1)  #[bsize, sub_len, hsize]
    # sub_encoder_sum_last = sub_encoder_sum[:, -1, :]
    # sub_average = sub_encoder_sum_last / sub_len  #[bsize, hsize]
    # return sub_average.unsqueeze(1)  #返回index指定的实体的平均编码表示。[bsize, 1, hsize]

def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return torch.tensor(out_tensor)


def covert_to_tokens(text, tokenizer=None, return_orig_index=False, max_seq_length=512):
    if not tokenizer:
        # tokenizer =BertTokenizer.from_pretrained('transformer_cpt/bert/', do_lower_case=True)
        tokenizer = MBartTokenizer.from_pretrained('')
    sub_text = []
    buff = ""
    flag_en = False
    flag_digit = False
    for char in text:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
            flag_en = False
            flag_digit = False
        else:
            if re.compile('\d').match(char):
                if buff != "" and flag_en:
                    sub_text.append(buff)
                    buff = ""
                    flag_en = False
                flag_digit = True
                buff += char
            else:
                if buff != "" and flag_digit:
                    sub_text.append(buff)
                    buff = ""
                    flag_digit = False
                flag_en = True
                buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    tokens = []
    # text_tmp = ''
    text_tmp = []
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token) if token != ' ' else []
        # text_tmp += token
        text_tmp.append(token)
        for sub_token in sub_tokens:
            #此处应更改为按照单词级索引词位置。中文文本是按照字级。
            #######改！！！！！！
            # tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_start_index.append(text_tmp.count(' ') * 2)
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_seq_length - 2:
            # if len(tokens) >= max_seq_length - 2:
                break
        else:
            continue
        break
    if return_orig_index:
        return tokens, tok_to_orig_start_index, tok_to_orig_end_index
    else:
        return tokens


def search_spo_index(tokens, subject_sub_tokens, object_sub_tokens):
    subject_start_index, object_start_index = -1, -1
    forbidden_index = None
    if len(subject_sub_tokens) > len(object_sub_tokens):
        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                subject_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                if forbidden_index is None:
                    object_start_index = index
                    break
                # check if labeled already
                elif index < forbidden_index or index >= forbidden_index + len(
                        subject_sub_tokens):
                    object_start_index = index
                    break

    else:
        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                object_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                if forbidden_index is None:
                    subject_start_index = index
                    break
                elif index < forbidden_index or index >= forbidden_index + len(
                        object_sub_tokens):
                    subject_start_index = index
                    break
    return subject_start_index, object_start_index

# def search_spo_index_en(tokens, subject_sub_tokens, object_sub_tokens):

