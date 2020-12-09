import sys, os

sys.path.append(os.getcwd())

import argparse
from src.util import *

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--kg', type=str, default='atomic')
parser.add_argument('--mode', type=str, default='???')
parser.add_argument('--model', type=str, default='gpt')
parser.add_argument('--use_pretrained', type=bool, default=True)
parser.add_argument('--config', type=str, default='default')
parser.add_argument('--log', type=str, default='test2')
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--total_steps', type=int, default=100000)
parser.add_argument('--eval_period', type=int, default=10000)

parser.add_argument('--load', type=str, default='model.1.ckpt')

args = parser.parse_args()

config = load_config(args.kg, args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

lg = get_logger()
oj = os.path.join


evaluator, generator = None, None

if args.kg == 'conceptnet':
    from src.eval import eval_c as eval
    from src.gen import gen_base as gen
    from src.train import train_c as train
    tokenize = get_conceptnet_encoder(args.kg, config)
elif args.kg == 'atomic':
    from src.eval import eval_a as eval
    from src.gen import gen_base as gen
    from src.train import train_a as train
    tokenize = get_atomic_encoder(args.kg, config)
else:
    raise NotImplementedError

args.log = 'log/{}/{}'.format(args.kg, args.log)
if args.model == 'gpt':
    from src.model.gpt import model_util as gpt_loader
    model = gpt_loader.get_model(config, len(tokenize.encoder))
    if args.use_pretrained:
        print("Load pre-trained parameter : {}".format(gpt_loader.load_params(model)))
else:
    raise NotImplementedError

assert args.load is not None

chkpnt = oj(args.log, 'chkpnt', args.load)
model.load_state_dict(torch.load(chkpnt)['model_state_dict'])

#dev_loader = train.get_data_loader(config, 'dev', shuffle=False)
test_loader = train.get_data_loader(config, 'test', shuffle=False, small=30)

#dev_generator = gen.get_generator(config, device, dev_loader, tokenize)
test_generator = gen.get_generator(config, device, test_loader, tokenize)

#dev_triples = dev_generator.generate(model)
test_triples = test_generator.generate(model)

#dev_triples_str = dev_generator.convert_to_str(dev_triples)
test_triples_str = test_generator.convert_to_str(test_triples)

evaluator = eval.Evaluator(config, config.train_path[0])
evaluator.init_bleu('test')

k = evaluator.evaluate(test_triples, test_triples_str)

from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

