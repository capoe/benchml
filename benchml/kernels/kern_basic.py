import numpy as np
from ..pipeline import Transform

class KernelBase(object):
    def __init__(self, **kwargs):
        self.name = "base"
        self.X_fit = None
        self.K_fit = None
    def evaluate(self, X1, X2, symmetric, **kwargs):
        raise NotImplementedError("<evaluate> not defined")
    def evaluateFit(self, X, y='not_used', **kwargs):
        self.X_fit = X
        self.K_fit = self.evaluate(X, X, symmetric=True, **kwargs)
        return self.K_fit
    def evaluatePredict(self, X, **kwargs):
        return self.evaluate(X, self.X_fit, symmetric=False, **kwargs)

class KernelDotDeprecated(KernelBase):
    def __init__(self, **kwargs):
        KernelBase.__init__(self, **kwargs)
        self.power = kwargs["power"]
        self.name = "dot"
    def evaluate(self, X1, X2, symmetric='not_used', **kwargs):
        k = X1.dot(X2.T)
        k = k**self.power
        return k

class KernelDot(Transform):
    default_args = {
        'power': 1,
        'self_kernel': False
    }
    req_inputs = ('X',)
    allow_params = {'X'}
    allow_stream = {'K', 'K_self'}
    stream_kernel = ('K',)
    stream_samples = ('K_self',)
    precompute = True
    def evaluate(self, x1, x2=None, diagonal_only=False):
        if x2 is None: x2 = x1
        if diagonal_only:
            return np.einsum('ai,ai->a', x1, x2, 
                optimize='greedy')**self.args["power"]
        else:
            return x1.dot(x2.T)**self.args["power"]
    def _fit(self, inputs, stream, params):
        K = self.evaluate(inputs["X"])
        params.put("X", np.copy(inputs["X"]))
        stream.put("K", K)
        if self.args["self_kernel"]:
            stream.put("K_self", K.diagonal())
    def _map(self, inputs, stream):
        K = self.evaluate(inputs["X"], self.params().get("X"))
        stream.put("K", K)
        if self.args["self_kernel"]:
            K_self = self.evaluate(inputs["X"], inputs["X"], 
                diagonal_only=True)
            stream.put("K_self", K_self)

class KernelGaussian(Transform):
    default_args = {
        'scale': 1,
        'self_kernel': False,
        'epsilon': 1e-10
    }
    req_inputs = ('X',)
    allow_params = {'X','sigma'}
    allow_stream = {'K', 'K_self'}
    stream_kernel = ('K',)
    stream_samples = ('K_self',)
    precompute = True
    def evaluate(self, x1, x2=None, sigma=None, diagonal_only=False):
        x1s = x1/(sigma+self.args["epsilon"])
        z1 = np.sum(x1s**2, axis=1)
        if x2 is None:
            x2s = x1s
            z2 = z1
        else:
            x2s = x2/sigma
            z2 = np.sum(x2s**2, axis=1)
        if diagonal_only:
            zz = -0.5*(z1+z2)
            xx = np.einsum('ai,ai->a', x1s, x2s, 
                optimize='greedy')
        else:
            zz = -0.5*np.add.outer(z1, z2)
            xx = x1s.dot(x2s.T)
        return np.exp(zz+xx)
    def _fit(self, inputs, stream, params):
        X = inputs["X"]
        sigma = self.args["scale"]*np.std(X, axis=0)
        K = self.evaluate(x1=inputs["X"], sigma=sigma)
        params.put("sigma", sigma)
        params.put("X", np.copy(inputs["X"]))
        stream.put("K", K)
        if self.args["self_kernel"]:
            stream.put("K_self", K.diagonal())
    def _map(self, inputs, stream):
        K = self.evaluate(x1=inputs["X"], x2=self.params().get("X"), 
            sigma=self.params().get("sigma"))
        stream.put("K", K)
        if self.args["self_kernel"]:
            K_self = self.evaluate(x1=inputs["X"], x2=inputs["X"], 
                sigma=self.params().get("sigma"),
                diagonal_only=True)
            stream.put("K_self", K_self)

