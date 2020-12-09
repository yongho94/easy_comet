import abc
import numpy as np

class BaseEvaluator:
    '''
    Generation result :
    [ ( [ idx_s1, idx_s2, ... idx_s|S|, idx_r1, ..., idx_o1, ... idx_o|O| ],
        [ prob_o1, prob_o2, ... prob_oK ] ),
      ( [ ], [ ] ), ... , ]
    '''
    def __init__(self, config, seed_triples):
        self.config = config
        self.seed_triples = self._load_triples(seed_triples)
        self.deduplicate_seeds()

    @abc.abstractmethod
    def _load_triples(self, seed_triples):
        pass

    @abc.abstractmethod
    def evaluate(self, triples):
        pass

    def deduplicate_seeds(self):
        for key, val in self.seed_triples.items():
            self.seed_triples[key] = list(set(val))

    def PPL(self, triples):
        ppl_list = list()
        for triple in triples:
            probs = triple['prob']
            ppl = 1
            for prob in probs:
                ppl *= prob[1]
            ppl = np.reciprocal(ppl) ** (1 / len(probs))
            ppl_list.append(ppl)

        return sum(ppl_list) / len(ppl_list)

    def NT_sro(self, gen_list):
        seed_sro = list()
        gen_sro = list()
        for i in self.seed_triples['sro']:
            seed_sro.append(' '.join(i).lower().strip())
        seed_sro = list(set(seed_sro))

        for i in gen_list:
            gen_sro.append(' '.join(i['str']).lower().strip())

        gen_sro = list(set(gen_sro))
        cnt = 0

        novel_sro = list()
        for sro in gen_sro:
            if sro not in seed_sro:
                novel_sro.append(sro)
                cnt += 1

        return novel_sro, cnt/len(set(gen_sro))

    def NT_o(self, gen_list):
        seed_so = list()
        gen_o = list()
        for seed in self.seed_triples['s+o']:
            seed_so.append(seed.lower())
        for i in gen_list:
            gen_o.append(i['str'][2].lower())

        gen_o = list(set(gen_o))
        cnt = 0

        novel_o = list()
        for o in gen_o:
            if o not in seed_so:
                novel_o.append(o)
                cnt += 1

        return novel_o, cnt/len(set(gen_o))
