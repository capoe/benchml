from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np

class GaussianProcess(Transform):
    req_args = ('alpha',)
    default_args = {
        'power': 1,
        'alpha': 1.0,
        'predict_variance': True
    }
    req_inputs = ('K', 'y')
    allow_params = {'K_inv', 'w', 'y_mean', 'y_std', 'y', 'dy_sorted', 'dy_std', 'dy_mean'}
    allow_stream = {'y', 'dy', 'dy_rank', 'dy_zscore'}
    def _fit(self, inputs, stream, params):
        # Read
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        K = inputs["K"]
        # Invert
        K_inv = np.linalg.inv(K**self.args["power"] + self.args["alpha"]*np.identity(K.shape[0]))
        w = K_inv.dot(y_train)
        # Store
        params.put("K_inv", K_inv)
        params.put("w", w)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        y, dy, dr, dz = self._map(inputs, stream)
        if dy is not None:
            params.put("dy_sorted", np.sort(dy))
            params.put("dy_std", np.std(dy))
            params.put("dy_mean", np.mean(dy))
    def _map(self, inputs, stream):
        p = self.args["power"]
        k = inputs["K"]
        # Mean
        mean = self.params().get("y_std")*(
            inputs["K"]**p).dot(self.params().get("w")) \
            + self.params().get("y_mean")
        stream.put("y", mean)
        # Variance
        if self.args["predict_variance"]:
            dy = self.predictError(k, inputs["K_self"])
            dr = self.rankError(dy)
            dz = self.zscoreError(dy)
        else:
            dy = None
            dr = None
            dz = None
        stream.put("dy", dy)
        stream.put("dy_rank", dr)
        stream.put("dy_zscore", dz)
        return mean, dy, dr, dz
    def predictError(self, k, k_self):
        p = self.args["power"]
        var = k_self**p - np.einsum('ab,bc,ac->a',
            k**p, self.params().get("K_inv"), k**p,
            optimize='greedy')
        dy = self.params().get("y_std")*var**0.5
        return dy
    def rankError(self, dy):
        if self.params().has("dy_sorted"):
            dy_sorted = self.params().get("dy_sorted")
            ranks = 1.*np.searchsorted(dy_sorted, dy)/len(dy_sorted)
        else:
            ranks = None
        return ranks
    def zscoreError(self, dy):
        if self.params().has("dy_std"):
            z = (dy-self.params().get("dy_mean"))/self.params().get("dy_std")
        else:
            z = None
        return z

class ResidualGaussianProcess(Transform):
    req_args = ('alpha',)
    default_args = {
        'power': 1,
        'alpha': 1.0,
        'predict_variance': False,
        'fit_residuals': False
    }
    req_inputs = ('K', 'y')
    allow_params = {'K_inv', 'w', 'y_mean', 'y_std', 'y', 'res_model', 'res'}
    allow_stream = {'y', 'dy', 'dk'}
    def fitResiduals(self, inputs, stream):
        # Centre
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        K = inputs["K"]
        residuals = []

        # Residuals on folds
        from ..splits import SplitKfold
        split = SplitKfold(K, k=20)
        for info, train, test in split:
            log << info << log.flush
            K_i = K[train][:,train]
            y_i = y_train[train]
            y_i_test = y_train[test]
            K_i_inv = np.linalg.inv(K_i**self.args["power"] + self.args["alpha"]*np.identity(K_i.shape[0]))
            w_i = K_i_inv.dot(y_i)
            r_i = y_std*(y_i_test - (K[test][:,train]**self.args["power"]).dot(w_i))
            residuals.append(np.abs(r_i))

        #K_i_inv = np.linalg.inv(K**self.args["power"] + self.args["alpha"]*np.identity(K.shape[0]))
        #w_i = K_i_inv.dot(y_train)
        #r_i = y_std*(y_train - (K**self.args["power"]).dot(w_i))
        #residuals.append(r_i)

        # Train residual GP
        residuals = np.concatenate(residuals)
        rsd_gp = GaussianProcess(
            tag="rsd_gp", 
            args=dict(
                alpha=self.args["alpha"],
                power=self.args["power"],
                predict_variance=False),
            inputs=dict(
                K=None,
                y=None))
        rsd_gp.openPrivateStream("temp")
        rsd_gp.openParams("temp")
        rsd_gp._fit({"K": K, "y": residuals})
        rsd_gp._map({"K": K, "y": residuals})
        y = rsd_gp.stream().get("y")
        self.params().put("res", residuals)
        self.params().put("res_model", rsd_gp)
    def _fit(self, inputs, stream, params):
        if self.args["fit_residuals"]:
            self.fitResiduals(inputs, stream)
        # Read
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        K = inputs["K"]
        # Invert
        K_inv = np.linalg.inv(K**self.args["power"] + self.args["alpha"]*np.identity(K.shape[0]))
        w = K_inv.dot(y_train)
        # Store
        params.put("K_inv", K_inv)
        params.put("w", w)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        p = self.args["power"]
        k = inputs["K"]
        # Mean
        mean = self.params().get("y_std")*(
            inputs["K"]**p).dot(self.params().get("w")) \
            + self.params().get("y_mean")
        stream.put("y", mean)
        # Variance
        if self.args["predict_variance"]:
            k_self = inputs["K_self"]
            var = k_self**p - np.einsum('ab,bc,ac->a',
                k**p, self.params().get("K_inv"), k**p,
                optimize='greedy')
            dy = self.params().get("y_std")*var**0.5
            stream.put("dy", dy)

            self.params().get("res_model")._map({"K": k})
            dk = self.params().get("res_model").stream().get("y")
            #n = 1
            #dk = []
            #for i in range(k.shape[0]):
            #    k_sorted = np.sort(k[i])[::-1]
            #    dk_i = np.sum(k_sorted[0:n])
            #    dk.append(dk_i)
            #dk = np.array(dk)

            stream.put("dk", dk)
        else:
            stream.put("dy", None)
            stream.put("dk", None)
