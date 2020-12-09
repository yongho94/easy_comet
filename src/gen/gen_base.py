import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_generator(config, device, loader, tokenize):
    return Generator(config, device, loader, tokenize)


class Generator:
    def __init__(self, config, device, loader, tokenize):
        self.config = config
        self.device = device
        self.loader = loader
        self.tokenize = tokenize
        self.end_idx = tokenize.encoder[self.config.end_token]
        self.o_start = self.config.max_s + self.config.max_r
        self.o_end = self.o_start + self.config.max_o

    def _truncate_obj(self, x):
        obj_mask = torch.zeros_like(x)
        obj_mask[0][:self.o_start] = 1
        return x * obj_mask

    def generate(self, model):
        model.eval()
        model.to(self.device)
        gen_triples = list()
        for data in tqdm(self.loader):
            input_ = data['input'].to(self.device)
            seq_mask_ = data['seq_mask'].to(self.device)
            input_ = self._truncate_obj(input_)
            seq_mask_ = self._truncate_obj(seq_mask_)

            gen_prob = list()
            for idx in range(self.o_start, self.o_end):
                pred_score, pred_prob = model(input_, seq_mask_)
                pred_prob = pred_prob.squeeze(0)[idx-1]
                pred_token_idx = torch.argmax(pred_prob)
                seq_mask_[0][idx] = 1
                gen_prob.append((int(pred_token_idx), float(pred_prob[pred_token_idx])))
                input_[0][idx] = pred_token_idx
                if int(pred_token_idx) == self.end_idx:
                    break
            gen_triples.append({'id':len(gen_triples), 'tensor':input_.to('cpu'), 'prob':gen_prob})

        return gen_triples

    def convert_to_str(self, gen_triples):
        triple_list = list()
        config = self.config
        for triple in gen_triples:
            one_triple = list()
            triple_tensor = triple['tensor'].squeeze().tolist()
            for elem in triple_tensor:
                one_triple.append(self.tokenize.decoder[elem])

            assert config.max_s + config.max_r + config.max_o == len(one_triple)

            s, r, o = '', '', ''
            for i in range(0, config.max_s):
                s += _cln_token(one_triple[i])
            for i in range(config.max_s, config.max_s+config.max_r):
                r += _cln_token(one_triple[i])
            for i in range(config.max_s+config.max_r, config.max_s + config.max_r + config.max_o):
                o += _cln_token(one_triple[i])

            triple_list.append({'id':triple['id'], 'str':[s.strip(), r.strip(), o.strip()]})
        return triple_list

def _cln_token(token):
    result = ''
    if token == '<unk>' or token == '<END>':
        return ''
    if token[-4:] == '</w>':
        result += token[:-4] + ' '
    else:
        result += token

    return result
