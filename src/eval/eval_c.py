from src.eval.eval_base import BaseEvaluator


def get_Evaluator(config,  tb_loc):
    evaluator = Evaluator(config, tb_loc)
    return evaluator


class Evaluator(BaseEvaluator):
    def __init__(self, config, seed_triples):
        super().__init__(config, seed_triples)

    def _load_triples(self, seed_triples):
        seeds = {'sro': list(), 's': list(), 'o': list(), 's+o': list()}

        data = list()
        if type(seed_triples) is list:
            for seed_file in seed_triples:
                with open(seed_file, 'r') as f:
                    data += f.readlines()
        elif type(seed_triples) is str:
            with open(seed_triples, 'r') as f:
                data += f.readlines()
        else:
            raise NotImplementedError

        for line in data:
            line = line.split('\t')
            line[0] = self.config.split_into_words[line[0]]
            seeds['sro'].append((line[1], line[0], line[2]))
            seeds['s'].append(line[1])
            seeds['o'].append(line[2])
            seeds['s+o'].append(line[1])
            seeds['s+o'].append(line[2])
        return seeds

    def evaluate(self, triples, triples_str):
        ppl = self.PPL(triples)
        nt_sro = self.NT_sro(triples_str)
        nt_o = self.NT_o(triples_str)
        return {'ppl':ppl, 'nt_sro':nt_sro, 'nt_o':nt_o}

    def plausible_score(self):
        pass
