# module: utils/rag_retrieval.py
"""
rag_retrieval.py — corpus, encoder and MeSH oracle for script10.

Split out of the old monolithic RAG cell so the metric layer in script10 can be
imported and tested without a GPU, a network connection, or a 500-document
PubMed cache.

TWO CORPORA, DELIBERATELY
─────────────────────────
`era_restricted`  publications up to the year BEFORE detected drift onset. This
                  is the deployment-realistic corpus: at prediction time in 2020
                  you cannot cite a 2022 paper. It is also the corpus the earlier
                  submission used, and a reviewer objected that excluding
                  post-2019 literature while predicting 2020-2022 outcomes is a
                  limitation.

`unrestricted`    everything fetched, no date ceiling.

The objection is real but so is the constraint, and they pull in opposite
directions: the restricted corpus is honest about deployment and dishonest about
COVID-era practice; the unrestricted corpus is the reverse. Scoring BOTH turns an
argument into a measurement — script10 reports how much the retrieved set changes
when only the corpus ceiling moves. That number is the answer to the reviewer.

MESH ORACLE
───────────
A fixed, model-independent relevance standard: the canonical MeSH descriptors for
each prediction target. Fixed matters — Jaccard-against-source cannot separate
"the model moved" from "the model got worse", because it uses a model as the
reference. This can. The descriptors are the standard MeSH terms for the three
conditions and are not tuned to any result.
"""
import json
import os
from pathlib import Path

import numpy as np

ART_DIRS = [Path("."),
            Path("./data"), Path(".")]

# Canonical MeSH descriptors per target. Chosen from the MeSH tree before any
# retrieval was scored; not adjusted afterwards.
MESH_ORACLE = {
    "label_vasopressor": {
        "Vasoconstrictor Agents", "Norepinephrine", "Vasopressins", "Epinephrine",
        "Shock", "Hypotension", "Blood Pressure", "Hemodynamics",
        "Fluid Therapy", "Resuscitation",
    },
    "label_intubation": {
        "Intubation, Intratracheal", "Respiration, Artificial", "Ventilators, Mechanical",
        "Respiratory Insufficiency", "Airway Management", "Ventilator Weaning",
        "Respiratory Distress Syndrome", "Oxygen Inhalation Therapy",
    },
    "label_septic_shock": {
        "Shock, Septic", "Sepsis", "Lactic Acid", "Systemic Inflammatory Response Syndrome",
        "Anti-Bacterial Agents", "Organ Dysfunction Scores", "Bacteremia",
        "Vasoconstrictor Agents", "Fluid Therapy",
    },
}


def _find(*names):
    for d in ART_DIRS:
        for n in names:
            if (d / n).exists():
                return d / n
    return None


def load_corpus():
    """
    Returns {"era_restricted": [doc, ...], "unrestricted": [doc, ...]}.

    A doc is {"pmid", "title", "abstract", "year", "mesh": [...]}.

    The cache is expected to hold the UNRESTRICTED set with a `year` on every
    document; the restricted variant is derived by filtering, so the two corpora
    are guaranteed to be the same documents under a different ceiling. Deriving
    it rather than fetching twice is what makes the sensitivity analysis in
    script10 a clean single-variable comparison.
    """
    p = _find("pubmed_corpus.json")
    if p is None:
        raise SystemExit(
            "pubmed_corpus.json not found. Run the corpus fetch step first.\n"
            "It must store the UNRESTRICTED corpus with a 'year' field per document; "
            "the era-restricted variant is derived here so the two differ only by "
            "the date ceiling.")
    docs = json.loads(p.read_text())
    if isinstance(docs, dict):
        docs = docs.get("documents", [])
    docs = [d for d in docs if d.get("abstract")]

    # Ceiling derived from the DETECTED drift onset, not hardcoded. The paper
    # defines the corpus as the window ending the year before drift was detected,
    # so pinning it to a literal silently decouples the corpus from the detector
    # the moment the onset moves — which it did when the calibration was fixed.
    ceiling = None
    log = _find("drift_detection_log.json")
    if log is not None:
        try:
            d = json.loads(log.read_text())
            cl = [r for r in (d.get("closed_loop") or []) if r.get("adapted")]
            onset = (cl[0]["era"] if cl else d.get("drift_group"))
            if onset:
                ceiling = int(str(onset).split("-")[0].strip()) - 1
                print(f"[rag] corpus ceiling {ceiling} (drift onset {onset}, "
                      f"year before)")
        except Exception as e:
            print(f"[rag] could not derive ceiling from drift log ({e})")
    if ceiling is None:
        ceiling = int(os.environ.get("RAG_ERA_CEILING", "2019"))
        print(f"[rag] ceiling fell back to {ceiling} — state this, it is not "
              f"derived from the detector")
    restricted = [d for d in docs if int(d.get("year", 0) or 0) <= ceiling]
    if not restricted:
        raise SystemExit(
            f"no documents at or before {ceiling} — check the 'year' field in the "
            f"corpus cache, or set RAG_ERA_CEILING.")
    return {"era_restricted": restricted, "unrestricted": docs}


class Retriever:
    """
    Dense retrieval over a fixed corpus.

    MedCPT is asymmetric — separate query and article encoders — which suits the
    structural mismatch here: queries are short attribution-derived keyword
    strings, documents are full abstracts. A symmetric encoder would embed both
    with the same tower and systematically favour long queries.

    Embeddings are L2-normalised at build time, so ranking is a single matmul and
    the top-k is exact rather than approximate. At a few hundred to a few thousand
    documents there is no reason to introduce an ANN index and its recall error
    into a measurement whose whole point is small differences between rankings.
    """

    def __init__(self, docs, model_name="ncbi/MedCPT", device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.docs = docs
        self.q_tok = AutoTokenizer.from_pretrained(f"{model_name}-Query-Encoder")
        self.q_model = AutoModel.from_pretrained(
            f"{model_name}-Query-Encoder").to(self.device).eval()
        self.a_tok = AutoTokenizer.from_pretrained(f"{model_name}-Article-Encoder")
        self.a_model = AutoModel.from_pretrained(
            f"{model_name}-Article-Encoder").to(self.device).eval()
        self.emb = self._embed_corpus()

    def _encode(self, texts, tok, model, max_len, batch=32):
        import torch
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = model(**enc).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.vstack(out)

    def _embed_corpus(self):
        texts = [f"{d.get('title','')} {d.get('abstract','')}".strip() for d in self.docs]
        return self._encode(texts, self.a_tok, self.a_model, 512)

    def retrieve(self, query, k=10):
        q = self._encode([query], self.q_tok, self.q_model, 64)[0]
        sims = self.emb @ q
        # argpartition then sort the k survivors: the full sort is wasted work at
        # k << n, and the tie-break must be deterministic or two identical
        # queries can return differently ordered lists and inflate every
        # instability metric downstream.
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((np.array([self.docs[i]["pmid"] for i in top]), -sims[top]))]
        return [{"pmid": str(self.docs[i]["pmid"]),
                 "sim": float(sims[i]),
                 "year": self.docs[i].get("year"),
                 "mesh": self.docs[i].get("mesh", [])} for i in top]
