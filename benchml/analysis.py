import numpy as np
from .logger import log
from .accumulator import Accumulator

def read_split_props_single(split):
    props = { k: v for kv in split.split(";") for k, v in [ kv.split("=") ] }
    props["id"] = split
    props["train:test"] = list(map(int, props["train:test"].split(":")))
    return props

def read_split_props(splits):
    for split in splits:
        yield read_split_props_single(split)

def analyse_section(split_this, benchmark_section, return_ordered=False):
    models = []
    P = []
    P_std = []
    S = []
    metrics = None
    for record in benchmark_section:
        splits = filter(lambda s: s["id"] == split_this["id"], 
            read_split_props(record["splits"]))
        for split in splits:
            models.append(record["model"])
            perf = record["performance"][split["id"]]
            keys = sorted(perf.keys())
            metrics = list(filter(lambda k: not k.endswith('_std'), keys))
            std = list(filter(lambda k: k.endswith('_std'), keys))
            p = [ perf[key] for key in metrics ]
            p_std = [ perf[key] for key in std ]
            s = [ (+1 if Accumulator.select_best[key] == "smallest" else -1) \
                for key in metrics ]
            P.append(p)
            P_std.append(p_std)
            S.append(s)
    P = np.array(P)
    P_std = np.array(P_std)
    S = np.array(S)
    P = P*S
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            rank = np.searchsorted(np.sort(P[:,j]), P[i,j])
            R[i,j] = rank
    R = R + 1.
    rank = np.mean(R, axis=1)
    order = np.argsort(rank)
    P = P*S
    log << "    %-30s" % "Model" << log.flush
    log << "Rank " << log.flush
    for i in range(len(metrics)):
        log << "| %-18s" % metrics[i] << log.flush
    log << log.endl
    log << "   " << "-"*(20+5+21*len(metrics)) << log.endl
    for o in order:
        log << "    %-30s %5.2f" % (models[o], rank[o]) << log.flush
        for i in range(len(metrics)):
            log << "| %+1.4f +- %+1.4f" % (P[o, i], P_std[o,i]) << log.flush
        log << log.endl
    if return_ordered:
        return {
            "models": [models[o] for o in order],
            "ranks": [rank[o] for o in order],
            "metrics": metrics,
            "mmatrix": P[order],
            "mmatrix_std": P_std[order]
        }
    else:
        return {
            "models": models,
            "ranks": rank,
            "metrics": metrics,
            "mmatrix": P,
            "mmatrix_std": P_std
        }

def analyse(benchmark):
    sections = {}
    for record in benchmark:
        if not record["dataset"] in sections:
            sections[record["dataset"]] = []
        sections[record["dataset"]].append(record)
    for section_name, benchmark_section in sorted(sections.items()):
        log << log.mg << "Section:" << section_name << log.endl
        splits_section = sorted(list(set(
            [ s for r in benchmark_section for s in r["splits"] ])))
        splits_section = list(map(read_split_props_single, splits_section))
        splits_section = sorted(splits_section, key=lambda s: s["train:test"][0])
        for split_this in splits_section:
            if split_this["perf"] == "train": continue
            log << log.mg << "  Split:" << split_this << log.endl
            ranking = analyse_section(split_this, benchmark_section)
            yield {**split_this, **ranking}

