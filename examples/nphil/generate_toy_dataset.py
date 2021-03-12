import nphil
import numpy as np
np.random.seed(0)

def generate_data(
        n_samples=20, 
        dim=10, 
        noise_scale=1.0, 
        yfunc=lambda X: np.exp(X[:,0])*X[:,2]):
    X = np.random.normal(
        np.random.normal(size=(dim,)),
        0.5*np.random.uniform(size=(dim,)),
        size=(n_samples, dim))
    y = yfunc(X) + noise_scale*np.random.normal(size=(X.shape[0],))
    return X, y

if __name__ == "__main__":
    X, y = generate_data()
    variables = nphil.generate.infer_variable_properties(X)
    extt = nphil.save_extt(
        "example.extt",
        arrays={"X": X, "Y": y}, 
        meta={"variables": variables})

