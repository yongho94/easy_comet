from tqdm import tqdm
import os
import torch

pj = os.path.join

def generate_samples(config, tokenize, lg):
    lg.info('Load train raw file')
    train_raw = get_raw_file(config.train_path)
    train_data = preprocess(train_raw, tokenize, config)
    train_path = pj(config.target_path, 'train.pkl')
    with open(train_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(train_path, len(train_data['pos'])))
        torch.save(train_data, f)

    lg.info('Load dev raw file')
    dev_raw = get_raw_file(config.dev_path)
    dev_data = preprocess(dev_raw, tokenize, config)
    dev_path = pj(config.target_path, 'dev.pkl')
    with open(dev_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(dev_path, len(dev_data['pos'])))
        torch.save(dev_data, f)

    lg.info('Load test raw file')
    test_raw = get_raw_file(config.test_path)
    test_data = preprocess(test_raw, tokenize, config)
    test_path = pj(config.target_path, 'test.pkl')
    with open(test_path, 'wb') as f:
        lg.info("Save {}, {} lines".format(test_path, len(test_data['pos'])))
        torch.save(dev_data, f)

    
def get_raw_file(path_list):
    result = []
    for f in path_list:
        with open(f, 'r') as f:
            for line in f.readlines():
                result.append(line)
    return result

def preprocess(raw_data, text_encoder, config):

    # Produce preprocessed file
    a = 0
    data = {'pos':list(), 'neg':list()}
    for line in tqdm(raw_data, desc="Preprocess data"):
        r, h, t, s = line.replace('\n','').split('\t')
        r, h, t = r.strip(), h.strip().lower(), t.strip().lower()
        h = text_encoder.encode([h], False)[0]
        t = text_encoder.encode([t], False)[0]
        t.append( text_encoder.encoder['<END>'])
        if config.rel == 'language':
            r = config.split_into_words[r]
            r = text_encoder.encode([r], False)[0]
            b = len(r)
            if b > a:
                a = b
        else:
            r = [ text_encoder.encoder['<'+r+'>'] ]
        if len(h) > config.max_s or len(t) > config.max_o or len(r) > config.max_r:
            continue
        if float(s) >= 1:
            data['pos'].append( ([h,r,t], line) )
        else:
            data['neg'].append( ([h,r,t], line) )

    final_data = {'pos': list(), 'neg':list()}

    for key in final_data:
        for line, origin in data[key]:
            pad_h = [0] * (config.max_s - len(line[0]))
            pad_r = [0] * (config.max_r - len(line[1]))
            pad_t = [0] * (config.max_o - len(line[2]))

            tmp_input = torch.tensor(line[0] + pad_h + line[1] + pad_r + line[2] + pad_t + [0])
            input_ = tmp_input[:-1]
            output_ = tmp_input[1:]

            seq_mask = input_.clone().bool().long()
            loss_mask = output_.clone().bool().long()
            loss_mask[:config.max_s+config.max_r-1] = 0

            data_point = {'input': input_, 'output': output_, 'seq_mask': seq_mask, 'loss_mask': loss_mask, 'origin':origin}
            final_data[key].append(data_point)

    return final_data