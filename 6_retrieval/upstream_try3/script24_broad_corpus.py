"""
script24_broad_corpus.py
════════════════════════
Re-fetch a BROADER PubMed corpus so that patient-level retrieval relevance
becomes measurable.

Why: the existing corpus (514 abstracts) was seeded on three interventions —
vasopressor, intubation, septic shock — so every abstract is about roughly the
same topic. Comorbidity coverage is 2-18 abstracts each out of 514, which means
no patient-level relevance metric can discriminate: every patient's retrieval
looks alike because there is nothing else to retrieve. That also explains the
flat MeSH P@5 across arms (0.696 / 0.703 / 0.702).

This adds organ-system and comorbidity seeds while keeping the era cut
(2009-2019, closing the year before the detected drift boundary) so the corpus
still predates the drift exactly as the original did.

Fetch logic (_fetch_batch, fetch_corpus) is copied verbatim from
script13Rag_BvsC.py:191-270 so the two corpora are built the same way.

Outputs  pubmed_corpus_broad.json
"""
import json, os, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

from Bio import Entrez

SAVE_PATH = Path(os.environ.get("CORPUS_SAVE_DIR", Path(__file__).resolve().parent / "corpus"))
Entrez.email = "research@example.com"
_KEY = os.environ.get("NCBI_API_KEY")
if _KEY:
    Entrez.api_key = _KEY
    _DELAY = 1.0 / 9
else:
    _DELAY = 1.0 / 3
    print("no NCBI_API_KEY - 3 req/sec")

MIN_DATE, MAX_DATE = "2009/01/01", "2019/12/31"

# original three interventions, kept so the broad corpus is a superset in spirit
SEED_ORIGINAL = [
    ("vasopressor AND intensive care", 200),
    ("intubation AND intensive care", 200),
    ("septic shock AND intensive care", 200),
]

# organ-system and comorbidity seeds - the axis the original corpus lacks
SEED_BROAD = [
    ("heart failure AND intensive care", 200),
    ("hypertension AND critical illness", 200),
    ("diabetes mellitus AND intensive care", 200),
    ("chronic kidney disease AND intensive care", 200),
    ("acute kidney injury AND intensive care", 200),
    ("pulmonary disease, chronic obstructive AND intensive care", 200),
    ("liver disease AND intensive care", 200),
    ("cirrhosis AND critical illness", 150),
    ("neoplasms AND intensive care", 200),
    ("respiratory failure AND mechanical ventilation", 200),
    ("acute respiratory distress syndrome", 200),
    ("lactate AND critical illness", 150),
    ("arterial blood pressure monitoring AND intensive care", 150),
    ("glasgow coma scale AND intensive care", 150),
    ("urine output AND acute kidney injury", 150),
    ("fluid resuscitation AND critical illness", 200),
    ("blood transfusion AND intensive care", 150),
    ("insulin infusion AND critical illness", 150),
    ("renal replacement therapy AND intensive care", 150),
    ("antibiotic timing AND sepsis", 150),
    ("corticosteroids AND septic shock", 150),
    ("delirium AND intensive care", 150),
    ("nutrition AND critical illness", 150),
    ("weaning AND mechanical ventilation", 150),
]


def _fetch_batch(pmids):
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    xml_data = handle.read(); handle.close()
    root = ET.fromstring(xml_data)
    out = []
    for art in root.findall(".//PubmedArticle"):
        try:
            medline = art.find(".//MedlineCitation")
            article = medline.find(".//Article")
            t_el = article.find(".//ArticleTitle")
            title = "".join(t_el.itertext()).strip() if t_el is not None else ""
            a_el = article.find(".//Abstract")
            if a_el is None:
                continue
            abstract = " ".join("".join(a.itertext()).strip()
                                for a in a_el.findall(".//AbstractText")).strip()
            if len(abstract) < 100:
                continue
            p_el = medline.find(".//PMID")
            y_el = article.find(".//Journal/JournalIssue/PubDate/Year")
            year = int(y_el.text) if y_el is not None and y_el.text and y_el.text.isdigit() else 2015
            au = article.find(".//AuthorList/Author")
            first_au = "Unknown"
            if au is not None:
                ln, ini = au.find("LastName"), au.find("Initials")
                if ln is not None:
                    first_au = ln.text + (f" {ini.text}" if ini is not None else "")
            j_el = article.find(".//Journal/Title")
            out.append({
                "pmid": p_el.text if p_el is not None else "",
                "title": title, "abstract": abstract, "year": year,
                "first_author": first_au,
                "journal": j_el.text if j_el is not None else "Unknown",
                "mesh": [d.text for d in medline.findall(".//MeshHeading/DescriptorName") if d.text],
            })
        except Exception:
            continue
    return out


def fetch_corpus(queries, min_date, max_date):
    raw = []
    for query, max_n in queries:
        try:
            h = Entrez.esearch(db="pubmed", term=query, retmax=max_n,
                               mindate=min_date, maxdate=max_date,
                               sort="relevance", retmode="json")
            res = json.loads(h.read()); h.close()
            pmids = res.get("esearchresult", {}).get("idlist", [])
            time.sleep(_DELAY)
            for i in range(0, len(pmids), 100):
                raw.extend(_fetch_batch(pmids[i:i + 100]))
                time.sleep(_DELAY)
            print(f"  {query[:55]:<57} -> {len(pmids)} pmids")
        except Exception as e:
            print(f"  ERROR {query[:40]}: {e}")
    seen, dedup = set(), []
    for a in raw:
        if a["pmid"] and a["pmid"] not in seen:
            seen.add(a["pmid"]); dedup.append(a)
    return dedup


if __name__ == "__main__":
    t0 = time.time()
    corpus = fetch_corpus(SEED_ORIGINAL + SEED_BROAD, MIN_DATE, MAX_DATE)
    corpus = [a for a in corpus if 2009 <= a["year"] <= 2019]
    print(f"\ncorpus: {len(corpus)} unique abstracts, {time.time()-t0:.0f}s")
    json.dump(corpus, open(SAVE_PATH / "pubmed_corpus_broad.json", "w"))

    terms = {"hypertension": ["hypertension"], "diabetes": ["diabetes", "diabetic"],
             "chf": ["heart failure", "cardiac failure"],
             "ckd": ["renal insufficiency", "kidney failure, chronic", "renal dialysis", "kidney disease"],
             "copd": ["pulmonary disease, chronic obstructive", "lung diseases, obstructive"],
             "liver_disease": ["liver disease", "hepatic", "cirrhosis"],
             "malignancy": ["neoplasm", "carcinoma", "cancer"]}
    old = json.load(open(SAVE_PATH / "pubmed_corpus.json"))
    print(f"\n{'comorbidity':<16}{'old (514)':>12}{'broad':>18}")
    for k, pats in terms.items():
        n_o = sum(1 for a in old if any(p in m.lower() for m in (a.get("mesh") or []) for p in pats))
        n_b = sum(1 for a in corpus if any(p in m.lower() for m in (a.get("mesh") or []) for p in pats))
        print(f"  {k:<14}{n_o:>6} ({100*n_o/len(old):>4.1f}%){n_b:>8} ({100*n_b/max(len(corpus),1):>4.1f}%)")
    print("\nsaved pubmed_corpus_broad.json")
