from .accumulator import Accumulator
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
            target_ref):
        log << log.mb << "Start hyper loop on stream" << stream.tag << log.endl
        update_cache = []
        for hyperidx, updates in enumerate(self):
            log << "  Hyper #%d" % hyperidx << log.endl
            metric = module.hyperEval(stream, updates, 
                split_args, accu_args, target, target_ref)
            update_cache.append({
                "metric": metric,
                "updates": updates
            })
        update_cache = sorted(update_cache, key=lambda cache: cache["metric"])
        best = update_cache[0] if (Accumulator.select(**accu_args) == "smallest") \
            else update_cache[-1]
        log << "  Select hyper parameters:" << log.endl
        log << "    Metrics = [ %+1.4f ... %+1.4e ]" % (
                update_cache[0]["metric"], update_cache[-1]["metric"]) \
            << log.endl
        return best["updates"]

class BayesianHyper(object):
    def __init__(self, *hypers, convert={}, seed=None):
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
            target_ref):
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

