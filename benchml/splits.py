import numpy as np
import json
SEED = None

def synchronize(seed):
    global SEED
    SEED = seed

class SplitBase(object):
    tag = "__none__"
    def __init__(self, dset):
        self.step = 0
        self.n_reps = 0
        self.n_samples = dset if (type(dset) is int) else len(dset)
    def isDone(self):
        return self.step >= self.n_reps
    def _next(self):
        split = self.next() 
        self.step += 1
        return split
    def next(self):
        raise NotImplementedError("'next' function not defined")
    def __iter__(self):
        while not self.isDone():
            yield self._next()

class SplitJson(SplitBase):
    tag = "json"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.splits = json.load(open(kwargs["json"]))
        self.n_reps = len(self.splits)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        return info, self.splits[self.step]["train"], self.splits[self.step]["test"]

class SplitLOO(SplitBase):
    tag = "loo"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.n_reps = dset if (type(dset) is int) else len(dset)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = list(np.arange(self.step)) + list(np.arange(self.step+1, self.n_samples))
        idcs_test = [ self.step ]
        return info, np.array(idcs_train), np.array(idcs_test)

class SplitKfold(SplitBase):
    tag = "kfold"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.n_reps = kwargs["k"]
        self.length = dset if (type(dset) is int) else len(dset)
        self.stride = self.length//self.n_reps + (1 if ((self.length % self.n_reps) > 0) else 0)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = list(np.arange(0, self.step*self.stride)) + list(np.arange((self.step+1)*self.stride, self.length))
        idcs_test = list(np.arange(self.step*self.stride, min(self.length, (self.step+1)*self.stride)))
        return info, idcs_train, idcs_test

class SplitChronological(SplitBase):
    tag = "chrono"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.f = kwargs["train_fraction"]
        if not isinstance(self.f, list): self.f = [ self.f ]
        self.n_reps = len(self.f)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        idcs = np.arange(self.n_samples)
        t = int(self.n_samples*self.f[self.step])
        return info, idcs[0:t], idcs[t:]

class SplitMC(SplitBase):
    tag = "random"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.n_reps = kwargs["n_splits"]
        self.f_mccv = kwargs["train_fraction"]
        self.rng = np.random.RandomState(SEED)
        self.n_train = int(self.f_mccv*self.n_samples)
        if self.n_train < 1 or self.n_train > (self.n_samples-1):
            raise ValueError("Invalid split with n_samples=%d, f=%f" % (
                self.n_samples, self.f_mccv))
        self.tag = "mc_n%d_f%1.1f" % (self.n_reps, self.f_mccv)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        idcs = np.arange(self.n_samples)
        self.rng.shuffle(idcs)
        idcs_train = np.sort(idcs[0:self.n_train])
        idcs_test = np.sort(idcs[self.n_train:])
        return info, idcs_train, idcs_test

class SplitSequentialMC(SplitBase):
    tag = "sequential"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.rng = np.random.RandomState(SEED)
        self.f_train = eval(kwargs["train_fraction"])
        self.repeat_fct = eval(kwargs["repeat_fraction_fct"])
        self.n_train_sequence = []
        for f in self.f_train:
            n_train = int(f*self.n_samples)
            n_test = self.n_samples - n_train
            self.n_train_sequence.extend([n_train]*int(
                self.repeat_fct(self.n_samples, n_train, n_test, f)))
        self.n_reps = len(self.n_train_sequence)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        n_train = self.n_train_sequence[self.step]
        idcs = np.arange(self.n_samples)
        self.rng.shuffle(idcs)
        idcs_train = np.sort(idcs[0:n_train])
        idcs_test = np.sort(idcs[n_train:])
        return info, idcs_train, idcs_test

class SplitLambda(SplitBase):
    tag = "lambda"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.lambda_funcs = kwargs["lambdas"]
        self.n_reps = len(self.lambda_funcs)
        self.dset = dset
        self.rng = np.random.RandomState(SEED)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        train = []
        test = []
        func = eval(self.lambda_funcs[self.step])
        for i in range(len(self.dset)):
            if func(self.dset.data[i]): train.append(i)
            else: test.append(i)
        return info, train, test

class SplitGrouped(SplitBase):
    tag = "grouped"
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self, dset)
        self.index = kwargs["group_index"]
        if isinstance(self.index, list):
            self.index = np.array(self.index)
        self.groups = list(sorted(list(set(self.index.tolist()))))
        self.n_reps = len(self.groups)
    def next(self):
        info = "%s_i%03d" % (self.tag, self.step)
        train = []
        test = []
        g = self.groups[self.step]
        train = np.where(self.index != g)[0]
        test = np.where(self.index == g)[0]
        return info, train, test

def Split(dset, **kwargs):
    return split_generators[kwargs["method"]](dset, **kwargs)

split_generators = {
  "loo": SplitLOO,
  "mc": SplitMC,
  "json": SplitJson,
  "lambda": SplitLambda,
  "chrono": SplitChronological,
  "kfold": SplitKfold,
  "random": SplitMC,
  "grouped": SplitGrouped,
  "sequential": SplitSequentialMC,
}
