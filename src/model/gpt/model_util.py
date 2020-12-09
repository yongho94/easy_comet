from src.model.gpt import model
import numpy as np
import json, os, torch

def get_model(config, vocab_num):
    hyper_params = {
        'seq_len':config.max_s + config.max_r + config.max_o,
        'vocab_num': vocab_num,
        'pos_vocab_num':config.model.pos_vocab_num,
        'hidden': config.model.hidden,
        'num_layers': config.model.num_layers,
        'num_heads': config.model.num_heads,
        'emb_dropout': config.model.emb_dropout,
        'att_dropout': config.model.att_dropout,
        'res_dropout': config.model.res_dropout,
        'activation': config.model.activation
    }
    gpt_model = model.GPT_model(hyper_params)
    return gpt_model

def mul_self(data):
    result = 1
    assert len(data) > 0
    for elem in data:
        result *= elem
    return result

def load_params(model):
    with open('data/model/params_shapes.json', 'r') as f:
        shape_list = json.load(f)

    flat_params = list()
    for i in range(10):
        param_f = os.path.join('data/model/params_{}.npy'.format(i))
        flat_params.append(np.load(param_f))

    flat_params = np.concatenate(tuple(flat_params))
    gpt_params = list()

    offset = 0
    for shape in shape_list:
        layer_size = mul_self(shape)
        layer_param = np.array(flat_params[offset:offset+layer_size]).reshape(shape)
        gpt_params.append(layer_param)
        offset = offset + layer_size

    emb_layer = gpt_params[1]
    comet_vocab = model.embedding_layer.bpe_emb.weight.shape[0]
    extra_vocab = comet_vocab - emb_layer.shape[0]

    emb_layer = np.concatenate(
        (emb_layer, np.random.random([extra_vocab, emb_layer.shape[1]])), axis=0
    )
    gpt_params[1] = emb_layer

    for load_layer, model_layer in zip(gpt_params, model.parameters()):
        load_layer = np.squeeze(load_layer)
        assert mul_self(load_layer.shape) == mul_self(model_layer.shape)
        if model_layer.shape == load_layer.shape:
            model_layer.data = torch.from_numpy(load_layer)
        elif model_layer.shape == np.transpose(load_layer).shape:
            model_layer.data = torch.from_numpy(load_layer).T
        else:
            return False
    model.float()
    return True
