from .accumulator import Accumulator
from .logger import log
import itertools
import copy
import numpy as np

class Hyper(object):
    def __init__(self, instructions):
        self.instr = instructions
        self.n_states = len(self.instr[list(self.instr.keys())[0]])
    def __iter__(self):
        for s in range(self.n_states):
            updates = {}
            for addr, val in self.instr.items():
                updates[addr] = val[s]
            yield updates

class GridHyper(object):
    def __init__(self, *hypers, **kwargs):
        self.hypers = hypers
    def getFields(self):
        return [ field for h in self.hypers for field in h.instr ]
    def __iter__(self):
        def merge(*updates):
            merged = {}
            for upd in updates: merged.update(upd)
            return merged
        update_cache = []
        for hyperidx, updates in enumerate(
                itertools.product(*tuple(self.hypers))):
            updates = merge(*updates)
            yield updates
    def optimize(self, module, stream, 
            split_args, 
            accu_args, 
            target, 
            target_ref,
            log=None):
        update_cache = []
        fields = self.getFields()
        ln_length = 12*(len(fields)+2)+3
        if log:
            log << "="*ln_length << log.endl
            log << "|   iter    |   target    |" + "|".join(
                map(lambda f: " %8s  " % f if len(f) <= 8 else " %5s...  " % f[0:5], 
                fields))+"|" << log.endl
            log << "-"*ln_length << log.endl
        prev = None
        invert = -1 if Accumulator.select(**accu_args) == "smallest" else +1
        for hyperidx, updates in enumerate(self):
            metric = module.hyperEval(stream, updates, 
                split_args, accu_args, target, target_ref)
            update_cache.append({
                "metric": metric,
                "updates": updates
            })
            if prev is None:
                prev = invert*metric
            if invert*metric >= prev:
                if log: colour = log.pp
                prev = invert*metric
            else:
                if log: colour = log.ww
            if log:
                log << colour << "|   %-5d   |  %+1.2e  |" % (hyperidx+1, metric) + "|".join(map(
                    lambda f: (" %+1.2e " % float(updates[f])) \
                        if type(updates[f]) is not list else "  [ ... ]  ", fields))+"|" << log.endl
        update_cache = sorted(update_cache, key=lambda cache: cache["metric"])
        best = update_cache[0] if (Accumulator.select(**accu_args) == "smallest") \
            else update_cache[-1]
        return best["updates"]

class BayesianHyper(object):
    def __init__(self, *hypers, convert={}, seed=0):
        self.hypers = hypers
        self.seed = seed
        self.convert = convert
    def convertUpdates(self, updates):
        for field in self.convert:
            updates[field] = self.convert[field](updates[field])
        return updates
    def optimize(self, module, stream,
            split_args, 
            accu_args, 
            target, 
            target_ref,
            log=None):
        all_updates = [ upd for upd in GridHyper(*self.hypers) ]
        bounds = copy.deepcopy(all_updates[0])
        for key in bounds.keys():
            bounds[key] = (bounds[key], all_updates[-1][key])
        def f(**kwargs):
            self.convertUpdates(kwargs)
            return module.hyperEval(stream, kwargs,
                split_args, accu_args, target, target_ref, verbose=False)*(-1. if \
                    Accumulator.select(accu_args['metric']) == 'smallest' else +1.)
        from bayes_opt import BayesianOptimization
        optimizer = BayesianOptimization(
            f=f, pbounds=bounds, random_state=self.seed)
        optimizer.maximize(
            init_points=4,
            n_iter=10,
        )
        return self.convertUpdates(optimizer.max["params"])

