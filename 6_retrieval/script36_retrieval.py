"""
script36_retrieval.py — attribution-conditioned literature retrieval, bounded vs full update.

The arXiv paper's retrieval analysis (try3/script13Rag_BvsC.py), on the clean pipeline and
for both model families. Per case, each model's explanation becomes two PubMed queries:
  physiology query  = "<label> ICU" + top-5 physiology features by |attribution|
  treatment query   = "<label> ICU" + top-4 treatment features by |attribution|
(feature names cleaned with the arXiv normalise_feature_name, copied verbatim), encoded
with the MedCPT query encoder and matched (cosine) against the cached MedCPT article
embeddings of the broad 2009-2019 corpus (try3: pubmed_corpus_broad.json, 3,668 abstracts,
corpus_broad_emb.npy). Top-5 documents per query.
Stability = Jaccard of the documents retrieved for the updated model vs the source model,
per stream; bounded (B) vs full (C) over five seeds.
Also stores, for the first cases, the query built from what the UPDATE changed
(|attribution after - attribution source|) and its documents, for a qualitative table.

Cases (as arXiv select_cases): final era, incident evaluation cohort, up to 150 positive
and 150 negative per label, fixed seed. Families: trees (modxgb_models{TREE_SUFFIX}.pkl),
neural (nn_states{NN_SUFFIX}.pt). Needs the try3 venv (transformers); runs offline.
"""
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from script30_modular_xgboost_clean import Frame, inc_mask, unpack_model
from script32_explanation_stability import explain as tree_explain
from script33_nn_explanation_stability import explain as nn_explain
from script17_seed_floor import device
from models.architectures import TwoStreamModel
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset

TRY3 = Path(os.environ.get("CORPUS_DIR", Path(__file__).resolve().parent / "corpus"))
TREE_SUFFIX = os.environ.get("TREE_SUFFIX", "_eta0.03")
NN_SUFFIX = os.environ.get("NN_SUFFIX", "_lr3e-3")
N_CASES, SEED, TOPK_P, TOPK_T, TOPK_DOCS, N_EXAMPLES = 300, 42, 5, 4, 5, 5
SEEDS = [42, 123, 7, 2024, 99]
nP = len(SEQ_FEATURES)
NAMES = list(SEQ_FEATURES) + list(TREATMENT_FEATURES)


def normalise_feature_name(raw: str) -> str:            # verbatim from try3/script13Rag_BvsC.py:137
    name = raw.split(":")[-1]
    name = re.sub(r'^(last|mean|std|min|max)_', '', name)
    name = re.sub(r'_?roll(mean|std|min|max|sum)_?\d*', '', name)
    name = re.sub(r'^(has_|early_|high_|total_)', '', name)
    name = re.sub(r'_(obs|flag|mask|ml|hrs|first|last|invasive|'
                  r'noninvasive)$', '', name)
    name = name.replace("_", " ").strip()
    _ABBREV = {"map": "MAP", "spo2": "SpO2", "fio2": "FiO2",
               "peep": "PEEP", "wbc": "WBC", "hr": "heart rate",
               "rr": "respiratory rate", "sbp": "systolic blood pressure",
               "dbp": "diastolic blood pressure"}
    if name.lower() in _ABBREV:
        name = _ABBREV[name.lower()]
    return name


def build_query(attr_row, idx_range, label, top_k):     # as try3 build_query: anchor + top-k, deduped
    anchor = label.replace("label_", "").replace("_", " ") + " ICU"
    order = [idx_range[j] for j in np.argsort(-np.abs(attr_row[idx_range]))]
    terms, seen = [anchor], set()
    for j in order[:top_k]:
        t = normalise_feature_name(NAMES[j])
        if t and t not in seen:
            seen.add(t)
            terms.append(t)
    return " ".join(terms)


def select_cases(y, n=N_CASES):                          # as try3 select_cases
    rng = np.random.RandomState(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.r_[rng.choice(pos, min(len(pos), n // 2), replace=False),
                 rng.choice(neg, min(len(neg), n // 2), replace=False)]


class MedCPT:
    def __init__(self):
        corpus = json.loads((TRY3 / "pubmed_corpus_broad.json").read_text())
        self.E = np.load(TRY3 / "corpus_broad_emb.npy")
        assert self.E.shape[0] == len(corpus), "cached embeddings do not match the corpus"
        assert np.allclose(np.linalg.norm(self.E[:50], axis=1), 1, atol=1e-4), "embeddings not normalised"
        self.pmid = [c["pmid"] for c in corpus]
        self.title = [c["title"] for c in corpus]
        self.tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
        self.cache = {}

    def top(self, queries):
        new = [q for q in dict.fromkeys(queries) if q not in self.cache]
        for a in range(0, len(new), 256):
            b = new[a:a + 256]
            with torch.no_grad():
                e = self.tok(b, truncation=True, padding=True, return_tensors="pt", max_length=64)
                Q = self.qm(**e).last_hidden_state[:, 0, :].numpy()
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            for q, row in zip(b, np.argsort(-(Q @ self.E.T), axis=1)[:, :TOPK_DOCS]):
                self.cache[q] = [self.pmid[j] for j in row]
        return [self.cache[q] for q in queries]


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    SEQ = json.loads(Path("eval_split.json").read_text())["sequential"]
    era = SEQ["adapt_steps"][-1]
    ids = [int(x) for x in SEQ["era_eval_stays"][era]]
    tr = load_enriched_split(Path("./data"), "train", SEQ_FEATURES, TREATMENT_FEATURES)
    te = load_enriched_split(Path("./data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
    F = Frame(te.filter(pl.col("stay_id").is_in(ids)))
    ds = ICUDataset(normalize(te, calculate_train_stats(tr, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES))))
                    .filter(pl.col("stay_id").is_in(ids)), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    pos_in_ds = {sid: j for j, sid in enumerate(ds.stay_ids)}
    arc = pickle.loads(Path(f"modxgb_models{TREE_SUFFIX}.pkl").read_bytes())
    ck = torch.load(f"nn_states{NN_SUFFIX}.pt", map_location="cpu")

    def nn_build(state):
        m = TwoStreamModel(nP, len(TREATMENT_FEATURES), len(LABEL_COLS)).to(device)
        m.load_state_dict(state)
        return m

    R = MedCPT()
    print(f"[check] corpus {len(R.pmid)} abstracts, cached embeddings {R.E.shape}, MedCPT query encoder loaded offline")
    out, examples = [], []
    for i, lbl in enumerate(LABEL_COLS):
        inc = np.where(inc_mask(F, i))[0]
        cases = inc[select_cases(F.y[inc, i])]
        cm = np.zeros(len(F.ids), bool)
        cm[cases] = True
        case_ids = F.ids[cm]                                   # Frame order (sorted stay ids)
        dsi = torch.as_tensor([pos_in_ds[int(s)] for s in case_ids])
        xs, xt = torch.from_numpy(ds.seq_data)[dsi], torch.from_numpy(ds.treat_data)[dsi]
        for fam in ("trees", "neural"):
            if fam == "trees":
                src = tree_explain(unpack_model(arc["source"][lbl]), F, cm)
                arm_attr = {(a, s): tree_explain(unpack_model(arc["models"][(lbl, f"{a}_xgb", s)]), F, cm)
                            for a in ("B", "C") for s in SEEDS}
            else:
                src = nn_explain(nn_build(ck["source"]), xs, xt, i)[0]
                arm_attr = {(a, s): nn_explain(nn_build(ck["states"][f"{a}__seed{s}"][-1]), xs, xt, i)[0]
                            for a in ("B", "C") for s in SEEDS}
            q = lambda A, rng_, k: [build_query(A[r], rng_, lbl, k) for r in range(len(A))]
            P, T = list(range(nP)), list(range(nP, len(NAMES)))
            src_docs = {"physio": R.top(q(src, P, TOPK_P)), "treat": R.top(q(src, T, TOPK_T))}
            res = {"label": lbl, "family": fam, "n_cases": int(len(case_ids))}
            for stream, rng_, k in (("physio", P, TOPK_P), ("treat", T, TOPK_T)):
                J = {a: np.array([[jac(d0, d1) for d0, d1 in zip(src_docs[stream], R.top(q(arm_attr[(a, s)], rng_, k)))]
                                  for s in SEEDS]) for a in ("B", "C")}       # (seeds, cases)
                dseed = J["B"].mean(1) - J["C"].mean(1)
                res[stream] = {"B": float(J["B"].mean()), "C": float(J["C"].mean()),
                               "B_minus_C": float(dseed.mean()), "sd": float(dseed.std(ddof=1)),
                               "seeds_B_higher": int((dseed > 0).sum()),
                               "same_docs_B": float((J["B"] == 1).mean()), "same_docs_C": float((J["C"] == 1).mean())}
            out.append(res)
            print(f"  {lbl[6:12]:6s} {fam:6s} n={res['n_cases']}  Jaccard vs source  physio B {res['physio']['B']:.3f} "
                  f"C {res['physio']['C']:.3f} (B>C {res['physio']['seeds_B_higher']}/5) | treat B {res['treat']['B']:.3f} "
                  f"C {res['treat']['C']:.3f} (B>C {res['treat']['seeds_B_higher']}/5)", flush=True)
            # qualitative: what the bounded update changed, for the first cases (seed 42)
            upd = np.abs(arm_attr[("B", 42)] - src)
            for r in range(min(N_EXAMPLES, len(case_ids))):
                qq = build_query(upd[r], list(range(len(NAMES))), lbl, TOPK_P)
                examples.append({"label": lbl, "family": fam, "stay_id": int(case_ids[r]),
                                 "update_query": qq, "docs": [R.title[R.pmid.index(p)][:120] for p in R.top([qq])[0]]})
    Path(f"retrieval{TREE_SUFFIX}{NN_SUFFIX}.json").write_text(json.dumps({"results": out, "examples": examples}, indent=1))
    print(f"Saved → retrieval{TREE_SUFFIX}{NN_SUFFIX}.json")


if __name__ == "__main__":
    main()
