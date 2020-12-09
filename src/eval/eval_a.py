from src.eval.eval_base import *
from src.prepro.prepro_a import get_raw_file
import pandas as pd
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

def get_Evaluator(config,  tb_loc):
    evaluator = Evaluator(config, tb_loc)
    return evaluator


class Evaluator(BaseEvaluator):
    def __init__(self, config, seed_triples):
        super().__init__(config, seed_triples)
        self.bleu_ref = defaultdict(list)
        self.tknz = TweetTokenizer()

    def _load_triples(self, seed_triples):
        seeds = {'sro': list(), 's': list(), 'o': list(), 's+o': list()}
        triples = []

        if type(seed_triples) is list:
            data = pd.read_csv(seed_triples[0])
            for seed_triple in seed_triples[1:]:
                seed_triple = pd.read_csv(seed_triple)
                data = pd.concat(data, seed_triple)
        elif type(seed_triples) is str:
            data = pd.read_csv(seed_triples)
        else:
            raise NotImplementedError

        for row in data.iloc:
            row = dict(row)
            for rel in self.config.relations:
                obj = cln_row(row[rel])
                if len(obj) == 0 or obj[0] == 'none':
                    continue
                for token in obj:
                    triples.append((row['event'], rel, token))

        for triple in triples:
            seeds['sro'].append(triple)
            seeds['s'].append(triple[0])
            seeds['o'].append(triple[2])
            seeds['s+o'].append(triple[0])
            seeds['s+o'].append(triple[2])
        return seeds

    def init_bleu(self, _type):
        triples = get_raw_file(self.config[_type+'_path'], self.config)
        for triple in triples:
            key = triple[0] + '_' + triple[1]
            key = key.lower().replace(' ','').replace('___','').strip()
            self.bleu_ref[key].append(self.tknz.tokenize(triple[2].lower()))

    def evaluate(self, triples, triples_str):
        ppl = self.PPL(triples)
        bleu = self.get_bleu_score(triples_str)
        nt_sro = self.NT_sro(triples_str)
        nt_o = self.NT_o(triples_str)

        return {'ppl':ppl, 'bleu':bleu, 'nt_sro':nt_sro, 'nt_o':nt_o}

    def get_bleu_score(self, triples_str):
        bleu_list = list()
        for triple in triples_str:
            triple = triple['str']
            key = triple[0]+'_'+triple[1]
            key = key.replace(' ','').replace('<BLANK>','').replace('<','').replace('>','').lower()
            referee = self.bleu_ref[key]
            bleu_score = bleu(referee, self.tknz.tokenize(triple[2]), [0.5, 0.5],
                              smoothing_function=SmoothingFunction().method1)
            bleu_list.append(bleu_score)

        return sum(bleu_list) / len(bleu_list)


def cln_row(row):
    row = row.replace('[', '').replace(']', '').replace('"','').strip()
    row = row.split(', ')
    try:
        row.remove('')
    except:
        row = row
    return row

