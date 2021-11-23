import copy
import itertools

import numpy as np

from benchml.accumulator import Accumulator

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None


class Hyper(object):
    def __init__(self, instructions):
        self.instr = instructions
        self.n_states = len(self.instr[list(self.instr.keys())[0]])

    def random(self):
        s = np.random.randint(0, self.n_states)
        updates = {}
        for addr, val in self.instr.items():
            updates[addr] = val[s]
        return updates

    def __iter__(self):
        for s in range(self.n_states):
            updates = {}
            for addr, val in self.instr.items():
                updates[addr] = val[s]
            yield updates


class GridHyper(object):
    def __init__(self, *hypers, **kwargs):
        self.hypers = list(hypers)

    def add(self, grid_hyper):
        for h in grid_hyper.hypers:
            self.hypers.append(h)

    def getFields(self):
        return [field for h in self.hypers for field in h.instr]

    def __iter__(self):
        def merge(*updates):
            merged = {}
            for upd in updates:
                merged.update(upd)
            return merged

        for hyperidx, updates in enumerate(itertools.product(*tuple(self.hypers))):
            updates = merge(*updates)
            yield updates

    def random(self):
        def merge(*updates):
            merged = {}
            for upd in updates:
                merged.update(upd)
            return merged

        return merge(*[h.random() for h in self.hypers])

    def optimize(
        self, module, stream, split_args, accu_args, target, target_ref, log=None, **kwargs
    ):
        update_cache = []
        fields = self.getFields()
        ln_length = 12 * (len(fields) + 2) + 3
        if log:
            log << "=" * ln_length << log.endl
            (
                log
                << "|   iter    |   target    |"
                + "|".join(
                    map(lambda f: " %8s  " % f if len(f) <= 8 else " %5s...  " % f[0:5], fields)
                )
                + "|"
                << log.endl
            )
            log << "-" * ln_length << log.endl
        prev = None
        invert = -1 if Accumulator.select(**accu_args) == "smallest" else +1
        for hyperidx, updates in enumerate(self):
            log_colour = log.ww
            metric = module.hyperEval(
                stream, updates, split_args, accu_args, target, target_ref, **kwargs
            )
            update_cache.append({"metric": metric, "updates": updates})
            if prev is None:
                prev = invert * metric
            if invert * metric >= prev:
                log_colour = log.pp
                prev = invert * metric
            if log:
                (
                    log
                    << log_colour
                    << "|   %-5d   |  %+1.2e  |" % (hyperidx + 1, metric)
                    + "|".join(
                        map(
                            lambda f: (" %+1.2e " % float(updates[f]))
                            if (updates[f] is not None and (not type(updates[f]) in {str, list}))
                            else "  [ ... ]  ",
                            fields,
                        )
                    )
                    + "|"
                    << log.endl
                )
        update_cache = sorted(update_cache, key=lambda cache: cache["metric"])
        best = (
            update_cache[0] if (Accumulator.select(**accu_args) == "smallest") else update_cache[-1]
        )
        return best["updates"]


class BayesianHyper(object):
    def __init__(self, *hypers, convert=None, seed=0, init_points=5, n_iter=10):
        if convert is None:
            convert = {}
        if BayesianOptimization is None:
            raise ImportError(
                "bayesian-optimization missing, try 'pip install bayesian-optimization'"
            )
        self.hypers = hypers
        self.convert = convert
        self.arrays = []
        self.array_lengths = {}
        self.seed = seed
        self.init_points = init_points
        self.n_iter = n_iter

    def findBounds(self):
        all_updates = [upd for upd in GridHyper(*self.hypers)]
        bounds = copy.deepcopy(all_updates[0])
        for key in bounds.keys():
            bounds[key] = (bounds[key], all_updates[-1][key])
        return bounds

    def convertUpdates(self, updates):
        for field in self.convert:
            fct = self.convert[field]
            # Lambda fcts may be written as str to support pickling:
            # E.g., fct = "lambda x: 10**x"
            if isinstance(fct, str):
                fct = eval(fct)
            updates[field] = fct(updates[field])
        return updates

    def detectArrays(self, bounds):
        self.arrays = []
        self.array_lengths = {}
        for key, val in bounds.items():
            if isinstance(val[0], (list, np.ndarray)):
                self.arrays.append(key)
                self.array_lengths[key] = len(val[0])

    def atomizeArrays(self, bounds):
        self.detectArrays(bounds)
        for arr in self.arrays:
            bound = bounds.pop(arr)
            length = len(bound[0])
            self.array_lengths[arr] = length
            for al in range(length):
                bounds["%s[%d]" % (arr, al)] = [bound[0][al], bound[1][al]]
        return bounds

    def joinArrays(self, updates):
        for arr in self.arrays:
            vals = np.array(
                [updates.pop("%s[%d]" % (arr, al)) for al in range(self.array_lengths[arr])]
            )
            updates[arr] = vals
        return updates

    def optimize(
        self, module, stream, split_args, accu_args, target, target_ref, log=None, **kwargs
    ):
        bounds = self.findBounds()
        bounds = self.atomizeArrays(bounds)

        def f(**kwargs):
            self.joinArrays(kwargs)
            self.convertUpdates(kwargs)
            return module.hyperEval(stream, kwargs, split_args, accu_args, target, target_ref) * (
                -1.0 if Accumulator.select(accu_args["metric"]) == "smallest" else +1.0
            )

        optimizer = BayesianOptimization(f=f, pbounds=bounds, random_state=self.seed)
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )
        return self.convertUpdates(self.joinArrays(optimizer.max["params"]))
