import sys, os

sys.path.append(os.getcwd())

import argparse
from src.util import *
from src.prepro import prepro_c  # ConceptNet Preprocessor
from src.prepro import prepro_a  # Atomic Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--kg', type=str, default='atomic')
parser.add_argument('--mode', type=str, default='???')
parser.add_argument('--config', type=str, default='default')
args = parser.parse_args()

config = load_config(args.kg, args.config)
lg = get_logger()

if args.kg == 'conceptnet':
    tokenize = get_conceptnet_encoder(args.kg, config)
    prepro_c.generate_samples(config, tokenize, lg)

elif args.kg == 'atomic':
    tokenize = get_atomic_encoder(args.kg, config)
    prepro_a.generate_samples(config, tokenize, lg)

lg.info('Done')
