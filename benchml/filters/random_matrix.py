from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np

def dist_mp(x, gamma):
    # NOTE gamma = #dim / #samples
    l, u = dist_mp_bounds(gamma)
    if x <= l or x >= u:
        return 0.
    else:
        return ( (u - x)*(x - l) )**0.5 / (2*np.pi*gamma*x)

def dist_mp_bounds(gamma):
    if gamma > 1.:
        return 0.0, (1.+gamma**0.5)**2
    else:
        return (1.-gamma**0.5)**2, (1.+gamma**0.5)**2

def dist_mp_sample(xs, gamma):
    ys = np.array([ dist_mp(x, gamma) for x in xs ])
    return ys

def dist_mp_test():
    xs = np.arange(0.,100.,0.01)
    ys = np.array([ [ dist_mp(x, 0.25), dist_mp(x, 0.5), dist_mp(x, 1.0), dist_mp(x, 2.0) ] for x in xs ])
    xys = np.zeros((xs.shape[0],5))
    xys[:,0] = xs
    xys[:,1:5] = ys
    np.savetxt('dist_mp_test.txt', xys)
    int_, err = scipy.integrate.quad( lambda x: dist_mp(x, 0.25), 0, 10.)
    print(int_, "+/-", err)
    return

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
        return c

def pca_compute(X, norm_div_std=True, norm_sub_mean=True, ddof=1, eps=1e-10):
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0, ddof=ddof)
    X_norm = X
    if norm_sub_mean:
        X_norm = X - x_mean
    else:
        x_mean = 0.0
    if norm_div_std:
        X_norm = div0(X_norm, x_std+eps)
    else:
        x_std = 1.0
    # Correlation matrix, diagonalize
    S = X_norm.T.dot(X_norm)/(X_norm.shape[0]-ddof)
    lambda_, U = np.linalg.eigh(S)
    idcs = lambda_.argsort()[::+1]
    lambda_ = lambda_[idcs]
    U = U[:,idcs]
    L = np.identity(lambda_.shape[0])*lambda_
    # Transform
    X_norm_pca = U.T.dot(X_norm.T).T
    return X_norm_pca, X_norm, x_mean, x_std, S, L, U

class CleanMatrix(Transform):
    req_inputs = {"X",}
    allow_stream = {"X",}
    allow_params = {"slice","rank"}
    stream_samples = {"X",}
    default_args = {
        "std_threshold": 1e-10,
        "axis": 0
    }
    def _fit(self, inputs, stream, params):
        X = inputs["X"]
        slice = np.where(np.std(X, axis=self.args["axis"]) \
            > self.args["std_threshold"])[0]
        if not len(slice):
            raise RuntimeError("CleanMatrix: Returned empty array")
        log << log.debug << "CleanMatrix: Removed %d columns" % (
            X.shape[self.args["axis"]] - len(slice)) << log.endl
        params.put("slice", slice)
        params.put("rank", len(X.shape))
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        s = [ slice(None) ]*self.params().get("rank")
        s[int(not self.args["axis"])] = self.params().get("slice")
        stream.put("X", inputs["X"][tuple(s)])

class MarchenkoPasturFilter(Transform):
    req_inputs = {"X",}
    allow_stream = {"X",}
    allow_params = {"V_upper", "L_upper", "x_mean", "x_std"}
    stream_samples = {"X",}
    def _fit(self, inputs, stream, params):
        X = inputs["X"]
        X_trafo, X_norm, x_mean, x_std, S, L, V = pca_compute(X)
        gamma = float(X.shape[1])/X.shape[0]
        lower, upper = dist_mp_bounds(gamma)
        sel_upper = np.where(L.diagonal() > upper)[0]
        if len(sel_upper) < 1:
            raise RuntimeError("MarchenkoPasturFilter returned empty")
        log << log.debug << "MarchenkoPasturFilter: %d components" \
            % len(sel_upper) << log.endl
        L_upper = L.diagonal()[sel_upper]
        V_upper = V[:,sel_upper]
        params.put("V_upper", V_upper)
        params.put("L_upper", L_upper)
        params.put("x_mean", x_mean)
        params.put("x_std", x_std)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        X = inputs["X"]
        X_proj = div0(X - self.params().get("x_mean"), self.params().get("x_std"))
        X_proj = X_proj.dot(self.params().get("V_upper"))
        X_proj = np.concatenate([X_proj, X_proj**2], axis=1)
        stream.put("X", X_proj)

