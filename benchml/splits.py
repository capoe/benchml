import numpy as np
import json
SEED = None

def synchronize(seed):
    global SEED
    SEED = seed

class SplitBase(object):
    def __init__(self):
        self.step = 0
        self.n_reps = 0
    def isDone(self):
        return self.step >= self.n_reps
    def next(self):
        raise NotImplementedError("'next' function not defined")
    def __iter__(self):
        while not self.isDone():
            yield self.next()

class SplitLOO(SplitBase):
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self)
        self.tag = "loo"
        self.n_samples = dset if (type(dset) is int) else len(dset)
        self.n_reps = dset if (type(dset) is int) else len(dset)
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = list(np.arange(self.step)) + list(np.arange(self.step+1, self.n_samples))
        idcs_test = [ self.step ]
        self.step += 1
        return info, idcs_train, idcs_test

class SplitMC(SplitBase):
    def __init__(self, dset, **kwargs):
        SplitBase.__init__(self)
        self.n_samples = dset if (type(dset) is int) else len(dset)
        self.n_reps = kwargs["n_splits"]
        self.f_mccv = kwargs["train_fraction"]
        self.rng = np.random.RandomState(SEED)
        self.n_train = int(self.f_mccv*self.n_samples)
        if self.n_train < 1 or self.n_train > (self.n_samples-1):
            raise ValueError("Invalid split with n_samples=%d, f=%f" % (
                self.n_samples, self.f_mccv))
        self.tag = "mc_n%d_f%1.1f" % (self.n_reps, self.f_mccv)
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs = np.arange(self.n_samples)
        self.rng.shuffle(idcs)
        idcs_train = idcs[0:self.n_train]
        idcs_test = idcs[self.n_train:]
        self.step += 1
        return info, idcs_train, idcs_test

def Split(dset, **kwargs):
    return split_generators[kwargs["method"]](dset, **kwargs)

split_generators = {
  "loo": SplitLOO,
  "mc": SplitMC,
  "random": SplitMC
}
