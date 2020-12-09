import sys, os
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
sys.path.append(os.getcwd())

from src.util import *

parser = argparse.ArgumentParser()
parser.add_argument('--kg', type=str, default='atomic')
parser.add_argument('--mode', type=str, default='???')
parser.add_argument('--model', type=str, default='gpt')
parser.add_argument('--use_pretrained', type=bool, default=False)
parser.add_argument('--config', type=str, default='default')
parser.add_argument('--log', type=str, default='no_prtn')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--total_steps', type=int, default=150000)
parser.add_argument('--eval_period', type=int, default=4000)
parser.add_argument('--use_earlystop', type=int, default=0)
args = parser.parse_args()

assert args.kg in ['conceptnet', 'atomic']

config = load_config(args.kg, args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

lg = get_logger()
oj = os.path.join

if args.kg == 'conceptnet':
    from src.train import train_c as train
    from src.eval import eval_c as eval
    from src.gen import gen_base as gen
    tokenize = get_conceptnet_encoder(args.kg, config)
elif args.kg == 'atomic':
    from src.train import train_a as train
    from src.eval import eval_a as eval
    from src.gen import gen_base as gen
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

trainer, evaluator, generator = None, None, None

tb_loc = oj(args.log, 'tb')
chkpnt_loc = oj(args.log, 'chkpnt')

if not os.path.exists(args.log):
    os.mkdir(args.log)
    os.mkdir(tb_loc)
    os.mkdir(chkpnt_loc)

writer = SummaryWriter(tb_loc)
train_loader = train.get_data_loader(config, 'train')
dev_loader = train.get_data_loader(config, 'dev', small=3000)
test_loader = train.get_data_loader(config, 'test')

evaluator = eval.Evaluator(config, config.train_path)

trainer = train.get_trainer(config, device, train_loader, writer, mode='train')
dev_trainer = train.get_trainer(config, device, dev_loader, writer, mode='valid')
dev_generator = gen.get_generator(config, device, dev_loader, tokenize)
test_generator = gen.get_generator(config, device, test_loader, tokenize)

optimizer = train.get_optimizer(model, config)
scheduler = train.get_lr_scheduler(optimizer, config, args.total_steps)

trainer.init_opt(optimizer)
trainer.init_lr_schedule(scheduler)

total_epoch = int(args.total_steps / len(train_loader))+1
eval_epoch = max(int(args.eval_period / len(train_loader))+1, 0)

print("Total Epoch : {}".format(total_epoch))

early_stop_loss = [1e10]
if args.kg == 'atomic':
    evaluator.init_bleu('dev')

for epoch in range(1, total_epoch+1):
    trainer.train_epoch(model, epoch)
    valid_loss = dev_trainer.train_epoch(model, epoch, trainer.global_step)
    early_stop_loss.append(valid_loss)

    if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]:
        break
    
    if epoch % eval_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': trainer.mean_loss},
            os.path.join(chkpnt_loc, 'model.{}.ckpt'.format(epoch)))

        dev_triples = dev_generator.generate(model)
        dev_triples_str = dev_generator.convert_to_str(dev_triples)
        with open(oj(args.log, 'dev_gen_triple_{}.pkl'.format(epoch)), 'wb') as f:
            pkl.dump(dev_triples_str, f)
        eval_result = evaluator.evaluate(dev_triples, dev_triples_str)
        for key, val in eval_result.items():
            if type(val) is list or type(val) is tuple:
                writer.add_scalar('valid/{}'.format(key), val[-1], trainer.global_step)
            else:
                writer.add_scalar('valid/{}'.format(key), val, trainer.global_step)

if args.kg == 'atomic':
    evaluator.init_bleu('test')

test_triples = test_generator.generate(model)
test_triples_str = test_generator.convert_to_str(test_triples)
test_result = evaluator.evaluate(test_triples, test_triples_str)

for key, val in test_result.items():
    if type(val) is list or type(val) is tuple:
        print(key, ':', val[-1])
    else:
        print(key, ':', val)