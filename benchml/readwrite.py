"""Module for BenchML Read/Write functionality."""
import copy
import json
import os
import pickle
import types

import numpy as np

from benchml import ptable, utils
from benchml.logger import Mock, log

ase = Mock()
ase.io = None


def disable_ase():
    global ase
    ase = Mock()
    ase.io = None
    log << log.mb << "[readwrite: Using built-in parser]" << log.endl  # pylint: disable=W0104
    return ase, ase.io


def configure(use_ase):
    global ase
    del ase.io
    if use_ase:
        import ase.io

        log << log.mb << "[readwrite: Using ASE]" << log.endl  # pylint: disable=W0104
    else:
        disable_ase()


class ExtendedTxt:
    def __init__(self, arrays=None, meta=None):
        if arrays is None:
            arrays = {}
        if meta is None:
            meta = {}
        self.arrays = arrays
        self.meta = meta

    def save(self, extt_file):
        write_extt(extt_file, self.arrays, self.meta)

    def load(self, extt_file):
        self.arrays, self.meta = read_extt(extt_file)

    def clone(self):
        return ExtendedTxt(
            arrays={a: np.copy(arr) for a, arr in self.arrays.items()},
            meta=copy.deepcopy(self.meta),
        )

    def __getitem__(self, key):
        return self.arrays[key]


def write_extt(extt_file, arrays, meta=None):
    if meta is None:
        meta = {}
    if isinstance(arrays, dict):
        with open(extt_file, "w", encoding="utf-8") as f:
            for k, v in arrays.items():
                shape = str(v.shape).replace(" ", "")
                f.write(f"{k}:{shape} ")
            f.write("\n")
            f.write(json.dumps(meta))
            f.write("\n")
            for k, v in arrays.items():  # order is guaranteed to match loop above
                np.savetxt(f, v)
            f.close()
    else:
        arrays.save(extt_file)


def read_extt(extt_file):
    with open(extt_file, "r", encoding="utf-8") as f:
        tuples = [item.split(":") for item in f.readline().strip().split()]
        array_rows = {t[0]: int(t[1].replace("(", "").split(",")[0]) for t in tuples}
        meta_str = f.readline().strip()
        meta = json.loads(meta_str)
        arrays = {}
        for array_name, n_rows in array_rows.items():
            arrays[array_name] = np.loadtxt(f, max_rows=n_rows)
    return ExtendedTxt(arrays=arrays, meta=meta)


class ExtendedXyz:
    def __init__(
        self, pos=None, symbols=None, cell=None, positions=None
    ):  # For compatibility with ASE
        if pos is None:
            pos = []
        if symbols is None:
            symbols = []
        self.info = {}
        self.cell = cell
        self.pbc = self.set_pbc()
        self.atoms = []
        self.positions = pos if positions is None else positions
        self.symbols = symbols
        self.heavy = None

    def __len__(self):
        return len(self.symbols)

    def set_pbc(self, booleans=None):
        if booleans is not None:
            self.pbc = np.array(booleans)
        elif self.cell is not None:
            self.pbc = np.array([True, True, True])
        else:
            self.pbc = np.array([False, False, False])
        return self.pbc

    def get_positions(self):
        return self.positions

    def get_cell(self):
        if self.cell is None and "Lattice" in self.info:
            self.cell = np.array(list(map(float, self.info["Lattice"].split()))).reshape((3, 3))
            self.set_pbc()
        return self.cell

    @staticmethod
    def _calculate_number_of_replicates(cell, r_cut):
        u, v, w = cell[0], cell[1], cell[2]
        a = np.cross(v, w, axis=0)
        b = np.cross(w, u, axis=0)
        c = np.cross(u, v, axis=0)
        ua = np.dot(u, a) / np.dot(a, a) * a
        vb = np.dot(v, b) / np.dot(b, b) * b
        wc = np.dot(w, c) / np.dot(c, c) * c
        proj = np.linalg.norm(np.array([ua, vb, wc]), axis=1)
        nkl = np.ceil(r_cut / proj).astype("int")
        return nkl

    def padToCutoff(self, r_cut):
        cell = np.array(self.get_cell())
        if cell is None:
            return self
        # Calculate # replicates
        nkl = self._calculate_number_of_replicates(cell, r_cut)
        # Replicate
        n_atoms = len(self)
        n_images = np.product(2 * nkl + 1)
        positions_padded = np.tile(self.positions, (n_images, 1))
        offset = 0
        for i in np.append(np.arange(0, nkl[0] + 1), np.arange(-nkl[0], 0)):
            for j in np.append(np.arange(0, nkl[1] + 1), np.arange(-nkl[1], 0)):
                for k in np.append(np.arange(0, nkl[2] + 1), np.arange(-nkl[2], 0)):
                    ijk = np.array([i, j, k])
                    positions_padded[offset : offset + n_atoms] += np.sum((cell.T * ijk).T, axis=0)
                    offset += n_atoms
        symbols_padded = np.tile(np.array(self.symbols), n_images)
        config_padded = self.__class__(
            positions=positions_padded, symbols=symbols_padded, cell=nkl * cell
        )
        return config_padded

    def get_chemical_symbols(self):
        return self.symbols

    def get_atomic_numbers(self):
        return np.array([ptable.lookup[s].z for s in self.get_chemical_symbols()])

    def getHeavy(self, recalculate=False):
        if recalculate or self.heavy is None:
            self.symbols = np.array(self.symbols)
            self.heavy = np.where(np.array(self.symbols != "H"))[0]
        return self.heavy, self.symbols[self.heavy], self.positions[self.heavy]

    def create(self, n_atoms, fs):
        self.info = tokenize_extxyz_meta(fs.readline())
        self.positions = []
        self.symbols = []
        for _ in range(n_atoms):
            new_atom = self.create_atom(fs.readline())
            self.positions.append(new_atom.pos)
            self.symbols.append(new_atom.name)
        self.positions = np.array(self.positions)
        self.get_cell()

    def create_atom(self, ln):
        ln = ln.split()
        name = ln[0]
        pos = list(map(float, ln[1:4]))
        pos = np.array(pos)
        new_atom = ExtendedXyzAtom(name, pos)
        self.atoms.append(new_atom)
        return new_atom


class ExtendedXyzAtom:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


def _parse_line_to_token_list(header):
    tokens = []
    pos0 = 0
    pos1 = 0
    status = "<"
    quotcount = 0
    while pos1 < len(header):
        status_out = status
        # On the lhs of the key-value pair?
        if status == "<":
            if header[pos1] == "=":
                tokens.append(header[pos0:pos1])
                pos0 = pos1 + 1
                pos1 = pos1 + 1
                status_out = ">"
                quotcount = 0
            else:
                pos1 += 1
        # On the rhs of the key-value pair?
        elif status == ">":
            if header[pos1 - 1 : pos1] == '"':
                quotcount += 1
            if quotcount == 0 and header[pos1] == " ":
                quotcount = 2
            if quotcount <= 1:
                pos1 += 1
            elif quotcount == 2:
                tokens.append(header[pos0:pos1])
                pos0 = pos1 + 1
                pos1 = pos1 + 1
                status_out = ""
                quotcount = 0
            else:
                assert False
        # In between key-value pairs?
        elif status == "":
            if header[pos1] == " ":
                pos0 += 1
                pos1 += 1
            else:
                status_out = "<"
        else:
            assert False
        status = status_out
    return tokens


def tokenize_extxyz_meta(header, allow_json=True):
    # Parse header: key1="str1" key2=123 key3="another value" ...
    header = header.replace("\n", "")
    if allow_json and header.startswith("{"):
        return json.loads(header)
    tokens = _parse_line_to_token_list(header)
    kvs = []
    for i in range(len(tokens) // 2):
        kvs.append([tokens[2 * i], tokens[2 * i + 1]])
    # Process key-value pairs
    info = {}
    for kv in kvs:
        value = "=".join(kv[1:])
        value = value.replace('"', "").replace("'", "")
        # Float?
        if "." in value:
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            # Int?
            try:
                value = int(value)
            except ValueError:
                pass
        info[kv[0]] = value
    return info


def read_extxyz_meta_only(config_file):
    with open(config_file, "r", encoding="utf-8") as ifs:
        while True:
            header = ifs.readline().split()
            if header:
                assert len(header) == 1
                n_atoms = int(header[0])
                info = tokenize_extxyz_meta(ifs.readline())
                for _ in range(n_atoms):
                    ifs.readline()
                yield info
            else:
                break


def patch_ase_config(config):
    def getHeavy(config):
        symbols = np.array(config.symbols)
        heavy = np.where(np.array(symbols != "H"))[0]
        return heavy, symbols[heavy], config.positions[heavy]

    config.getHeavy = types.MethodType(getHeavy, config)


def read_ase(config_file, index):
    if index is None:
        index = ":"
    configs = ase.io.read(config_file, index)
    metas = list(read_extxyz_meta_only(config_file))
    assert len(configs) == len(metas)  # File format error?
    for midx, meta in enumerate(metas):
        configs[midx].info = meta
        patch_ase_config(configs[midx])
    return configs


def read_xyz(config_file, index=":"):
    if ase.io is not None:
        return read_ase(config_file, index)
    configs = []
    with open(config_file, "r", encoding="utf-8") as ifs:
        while True:
            header = ifs.readline().split()
            if header:
                assert len(header) == 1
                n_atoms = int(header[0])
                config = ExtendedXyz()
                config.create(n_atoms, ifs)
                configs.append(config)
            else:
                break
    return configs


def write_xyz(config_file, configs, allow_json=True):
    if ase.io is not None:
        ase.io.write(config_file, configs)
    else:
        with open(config_file, "w", encoding="utf-8") as ofs:
            for c in configs:
                ofs.write(f"{len(c)}\n")
                if allow_json:
                    json_s = json.dumps(c.info, sort_keys=True)
                    ofs.write(json_s)
                else:
                    for k in sorted(c.info.keys()):
                        # int or float?
                        if not isinstance(c.info[k], str):
                            ofs.write(f"{k}={c.info[k]} ")
                        # String
                        else:
                            ofs.write(f'{k}="{c.info[k]}" ' '%s="%s" ')
                ofs.write("\n")
                for i in range(len(c)):
                    ofs.write(
                        "%s %+1.4f %+1.4f %+1.4f\n"
                        % (
                            c.get_chemical_symbols()[i],
                            c.positions[i][0],
                            c.positions[i][1],
                            c.positions[i][2],
                        )
                    )


def save(arch_file_path, obj, method=None, embed_git_hash=False, **kwargs):
    if method is None:
        if not isinstance(arch_file_path, str):
            assert isinstance(obj, str)  # Invalid function call to benchml.save
            arch_file_path, obj = obj, arch_file_path
        if embed_git_hash:
            obj = {"git_hash": utils.git_hash(), "bml_object": obj}
        with open(arch_file_path, "wb") as arch_file:
            arch_file.write(pickle.dumps(obj))
    else:
        if embed_git_hash:
            obj = {"git_hash": utils.git_hash(), "bml_object": obj}
        method.save(arch_file_path, obj, **kwargs)


def load(arch_file_path, method=None, **kwargs):
    if method is None:
        with open(arch_file_path, "rb") as model:
            res = pickle.load(model)
    else:
        res = method.load(arch_file_path, **kwargs)
    return res


read_formats = {".extt": read_extt, ".xyz": read_xyz}

write_formats = {".extt": write_extt, ".xyz": write_xyz}


def read(input_file, *args, **kwargs):
    _, ext = os.path.splitext(input_file)
    return read_formats[ext](input_file, *args, **kwargs)


def write(output_file, *args, **kwargs):
    _, ext = os.path.splitext(output_file)
    return write_formats[ext](output_file, *args, **kwargs)
