import numpy as np

import benchml as bml

default_config = {
    "dim": 100,
    "n_modes": 1,
    "J_min": 0.0,
    "J_max": 1.0,
    "J_scale": 1.0,
    "B_min": 0.0,
    "B_max": 2.0,
    "B_scale": 0.0,
    "spawn": "random",
}


class IsingModel(object):
    def __init__(self, config):
        self.config = {k: config[k] for k in config}
        dim = config["dim"]
        J_min = config["J_min"]
        J_max = config["J_max"]
        J_sc = config["J_scale"]
        B_min = config["B_min"]
        B_max = config["B_max"]
        B_sc = config["B_scale"]
        # J & B matrices
        J = np.zeros((dim, dim))
        J_modes = []
        for mode in range(config["n_modes"]):
            v = np.random.normal(0, 1, size=(dim,))
            J_modes.append(v)
            J = J + np.outer(v, v)
        self.J = J_sc * J
        self.J_modes = np.array(J_modes)
        self.B = B_sc * np.random.uniform(low=B_min, high=B_max, size=dim)
        # Zero diagonals
        np.fill_diagonal(self.J, 0.0)
        return

    def spawn(self):
        S = IsingState(self.config)
        return S

    def energy(self, state):
        return -0.5 * state.S.dot(self.J).dot(state.S) - self.B.dot(state.S)

    def energy_split(self, state):
        return -self.B.dot(state.S), -0.5 * state.S.dot(self.J).dot(state.S)


class IsingEnsemble(object):
    def __init__(self, model, integrator):
        self.model = model
        self.integrator = integrator
        self.states = []

    def append(self, S, E):
        self.states.append(IsingState(config={}, S=S, E=E))
        return

    def exportState(self):
        # Embed index, label
        for idx, s in enumerate(self.states):
            s.info["idx"] = idx
            s.info["label"] = "ISING%06d" % idx
        # Descriptor matrix
        n_samples = len(self.states)
        n_dim = self.model.config["dim"]
        IX = np.zeros((n_samples, n_dim), dtype="float64")
        for i in range(n_samples):
            IX[i, :] = np.copy(self.states[i].S)
        # Setup state
        state = State(ising_J=self.model.J, ising_B=self.model.B)
        state.register("generate_ising", self.model.config)
        state["configs"] = self.states
        state["labels"] = [s.info for s in self.states]
        state["IX"] = IX
        state["n_samples"] = n_samples
        state["n_dim"] = n_dim
        return state

    def pickle(self, pfile="kernel.svmbox.pstr"):
        pstr = pickle.dumps(self)
        with open(pfile, "w") as f:
            f.write(pstr)
        return


class IsingState(object):
    def __init__(self, config={}, S=np.array([]), E=None):
        if S.shape[0] > 0:
            self.S = np.copy(S)
            self.dim = self.S.shape[0]
            self.E = E
        else:
            method = config["spawn"]
            self.config = {k: config[k] for k in config}
            dim = config["dim"]
            if method == "random":
                S = np.random.randint(low=0, high=2, size=dim)
            elif method == "zeros":
                S = np.zeros((dim,))
            S = (S - 0.5) * 2
            self.dim = dim
            self.S = S
            self.E = E
        self.info = {"E": self.E, "dim": self.dim}
        return


class IsingIntegrator(object):
    def __init__(self, config):
        self.kT = config["kT"]

    def integrate(
        self,
        model,
        state,
        n_steps,
        anneal=np.array([]),
        sample_every=1,
        sample_start=0,
        verbose=True,
    ):
        ensemble = IsingEnsemble(model, self)
        # Energy of initial configuration
        e_B, e_J = model.energy_split(state)
        E0 = e_J + e_B
        if verbose:
            print("Step= %4d Energy= %+1.4e [e_B=%+1.4e e_J=%+1.4e]" % (0, E0, e_B, e_J))
        # Annealing
        if anneal.shape[0] == 0:
            anneal = [self.kT for n in range(n_steps)]
        else:
            assert anneal.shape[0] == n_steps
        # Integrate
        for n in range(n_steps):
            kT = anneal[n]
            T = -1 * state.S
            for i in range(state.dim):
                Jij_Sj = model.J[i].dot(state.S)
                e0 = -state.S[i] * (Jij_Sj + model.B[i])
                e1 = -T[i] * (Jij_Sj + model.B[i])
                accept = False
                if e1 < e0:
                    accept = True
                else:
                    p_acc = np.exp(-(e1 - e0) / kT)
                    p = np.random.uniform()
                    if p < p_acc:
                        accept = True
                if accept:
                    state.S[i] = T[i]
            e_B, e_J = model.energy_split(state)
            E = e_J + e_B
            if verbose:
                print(
                    "Step= %4d Energy= %+1.4e [e_B=%+1.4e e_J=%+1.4e]   T=%+1.3e"
                    % (n, E, e_B, e_J, kT)
                )
            if n >= sample_start and n % sample_every == 0:
                ensemble.append(state.S, E)
        return ensemble


if __name__ == "__main__":
    np.random.seed(0)
    model = IsingModel(config=default_config)
    state = model.spawn()
    integrator = IsingIntegrator({"kT": 20.0})
    n_steps = 10100
    sample_every = 100
    data = integrator.integrate(
        model, state, n_steps, sample_every=sample_every, anneal=np.linspace(100.0, 0.1, n_steps)
    )
    X = np.array([d.S for d in data.states])
    y = np.array([d.E for d in data.states])
    bml.write(
        "data.extt", arrays={"X": X, "Y": y, "J": model.J, "V": model.J_modes}, meta=model.config
    )
