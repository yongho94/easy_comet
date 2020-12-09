import logging
import os
import json
from src.bpe_encoder import TextEncoder

class DictObj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)
    
    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return DictObj(value) if isinstance(value, dict) else value

def get_logger(level="info", name='log'):
    lg = logging.getLogger(name)
    if level == 'debug':
        lg.setLevel(logging.DEBUG)
    else:
        lg.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] :: %(message)s')
    stream_handler.setFormatter(formatter)
    lg.addHandler(stream_handler)

    lg.info('Logger Module Initialized, Set log level {}'.format(level))

    return lg

def load_config(kg, conf):
    with open(os.path.join('config', '{}_{}.json'.format(kg, conf)), 'r') as f:
        config = json.load(f)
    return DictObj(config)

def get_atomic_encoder(kg, config):
    assert kg == 'atomic'
    encoder_path = 'data/model/encoder_bpe_40000.json'
    bpe_path = 'data/model/vocab_40000.bpe'

    text_encoder = TextEncoder(encoder_path, bpe_path)
    special = [config.start_token, config.end_token, config.blank_token]
    special += ["<{}>".format(relation) for relation in config.relations]

    for special_token in special:
        text_encoder.decoder[len(text_encoder.encoder)] = special_token
        text_encoder.encoder[special_token] = len(text_encoder.encoder)

    return text_encoder

def get_conceptnet_encoder(kg, config):
    assert kg == 'conceptnet'
    encoder_path = 'data/model/encoder_bpe_40000.json'
    bpe_path = 'data/model/vocab_40000.bpe'
    
    text_encoder = TextEncoder(encoder_path, bpe_path)
    special = [config.start_token, config.end_token]
    special += ["<{}>".format(relation) for relation in config.relations]

    for special_token in special:
        text_encoder.decoder[len(text_encoder.encoder)] = special_token
        text_encoder.encoder[special_token] = len(text_encoder.encoder)

    return text_encoder
