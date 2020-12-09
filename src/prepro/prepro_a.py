from tqdm import tqdm
import os
import torch
import pandas as pd

oj = os.path.join


def generate_samples(config, tokenize, lg):
    lg.info('Load train raw file')
    train_triple = get_raw_file(config.train_path, config)
    train_data = preprocess(train_triple, tokenize, config)
    train_path = oj(config.target_path, 'train.pkl')
    with open(train_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(train_path, len(train_data)))
        torch.save(train_data, f)

    lg.info('Load dev raw file')
    dev_raw = get_raw_file(config.dev_path, config)
    dev_data = preprocess(dev_raw, tokenize, config)
    dev_path = oj(config.target_path, 'dev.pkl')
    with open(dev_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(dev_path, len(dev_data)))
        torch.save(dev_data, f)

    lg.info('Load test raw file')
    test_raw = get_raw_file(config.test_path, config)
    test_data = preprocess(test_raw, tokenize, config)
    test_path = oj(config.target_path, 'test.pkl')
    with open(test_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(test_path, len(test_data)))
        torch.save(test_data, f)


def get_raw_file(path_list, config):
    triples = []
    for path in path_list:
        data = pd.read_csv(path)
        for row in data.iloc:
            row = dict(row)
            for rel in config.relations:
                obj = cln_row(row[rel])
                if len(obj) == 0 or obj[0] == 'none':
                    continue
                for token in obj:
                    triples.append((row['event'], rel, token))

    return triples


def cln_row(row):
    row = row.replace('[', '').replace(']', '').replace('"','').strip()
    row = row.split(', ')
    try:
        row.remove('')
    except:
        row = row
    return row


def preprocess(triples, tokenize, config):

    tokn_triple = list()
    for tri in tqdm(triples):
        s = tri[0].replace('___', config.blank_token)
        r = tri[1]
        o = tri[2]

        id_s = s.find(config.blank_token)
        id_e = s.find(config.blank_token) + len(config.blank_token)

        if id_s >= 0:
            s_tokens = tokenize.encode([s[:id_s]])[0]
            s_tokens.append(tokenize.encoder[config.blank_token])
            s_tokens.extend(tokenize.encode([s[id_e:]])[0])
        else:
            s_tokens = tokenize.encode([s])[0]

        s_tokens = s_tokens
        r_tokens = tokenize.encoder['<{}>'.format(r)]
        o_tokens = tokenize.encode([o])[0] + [tokenize.encoder[config.end_token]]

        if len(s_tokens) > config.max_s or len(o_tokens) > config.max_o:
            continue

        tokn_triple.append( ([s_tokens, [r_tokens], o_tokens], tri) )

    final_data = list()
    for tri, origin in tokn_triple:
        pad_h = [0] * (config.max_s - len(tri[0]))
        pad_r = [0] * (config.max_r - len(tri[1]))
        pad_t = [0] * (config.max_o - len(tri[2]))

        tmp_input = torch.tensor(tri[0] + pad_h + tri[1] + pad_r + tri[2] + pad_t + [0])
        input_ = tmp_input[:-1]
        output_ = tmp_input[1:]

        seq_mask = input_.clone().bool().long()
        loss_mask = output_.clone().bool().long()
        loss_mask[:config.max_s+config.max_r-1] = 0

        data_point = {'input':input_, 'output':output_, 'seq_mask':seq_mask, 'loss_mask':loss_mask, 'origin':origin}
        final_data.append(data_point)

    return final_data