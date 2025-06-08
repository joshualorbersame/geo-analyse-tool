#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
litcrit_gui.py · Streamlit-Frontend für den ‘Literature Critic’
v2.1 – Bug-Fix: tmp-Datei-Handhabung
(2025-05-28)
"""

from __future__ import annotations

import os, sys, json, time, textwrap, tempfile, datetime as dt
from pathlib import Path
from typing import List, Dict, Iterable

import streamlit as st
import requests
from langdetect import detect
from deep_translator import GoogleTranslator
from pdfminer.high_level import extract_text

import importlib, logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

###############################################################################
# Konstanten
###############################################################################

DESKTOP = Path.home() / "Desktop"
TARGET_LANGS = ["en", "de", "es", "fr", "ru", "hi", "zh-CN", "ar"]
MAX_HITS = 3
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "your@mail.here")

###############################################################################
# spaCy Utility
###############################################################################

@st.cache_data(show_spinner=False)
def load_spacy_model(code: str):
    model_map = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "pt": "pt_core_news_sm",
    }
    name = model_map.get(code, "en_core_web_sm")
    try:
        return importlib.import_module(name).load()
    except (ModuleNotFoundError, IOError):
        from spacy.cli import download
        download(name)
        return importlib.import_module(name).load()

def extract_keywords(text: str, nlp, top_k: int = 25) -> List[str]:
    doc = nlp(text)
    stopwords = nlp.Defaults.stop_words
    counter: Dict[str, int] = {}
    for np in doc.noun_chunks:
        t = np.text.strip()
        if (
            len(t) > 2
            and all(tok.is_alpha for tok in np)
            and t.lower() not in stopwords
        ):
            counter[t] = counter.get(t, 0) + 1
    return [w for w, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]]

###############################################################################
# Übersetzen
###############################################################################

@st.cache_data(show_spinner=False)
def translate_terms(terms: Iterable[str], tgt: str) -> List[str]:
    trans = GoogleTranslator(source="auto", target=tgt)
    out = []
    for t in terms:
        try:
            out.append(trans.translate(t))
        except Exception:
            time.sleep(0.4)
            out.append(t)
    return list(dict.fromkeys(out))

###############################################################################
# Quellen-Abfragen (PubMed, Crossref, arXiv, S2-API)
###############################################################################

def search_pubmed(q: str) -> List[Dict]:
    try:
        ids = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": q, "retmax": MAX_HITS, "retmode": "json", "email": PUBMED_EMAIL},
            timeout=10,
        ).json()["esearchresult"]["idlist"]
        if not ids:
            return []
        summ = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=10,
        ).json()["result"]
        return [
            {
                "title": summ[i]["title"],
                "authors": ", ".join(a["name"] for a in summ[i].get("authors", [])[:3]),
                "year": summ[i].get("pubdate", "")[:4],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
                "source": "PubMed",
                "keyword": q,
            } for i in ids
        ]
    except Exception:
        return []

def search_crossref(q: str) -> List[Dict]:
    try:
        items = requests.get(
            "https://api.crossref.org/works",
            params={"query": q, "rows": MAX_HITS},
            timeout=10,
        ).json()["message"]["items"]
        return [
            {
                "title": it.get("title", [""])[0],
                "authors": ", ".join(a.get("family", "") for a in it.get("author", [])[:3]),
                "year": it.get("issued", {}).get("date-parts", [[None]])[0][0],
                "url": it.get("URL", ""),
                "source": "Crossref",
                "keyword": q,
            } for it in items
        ]
    except Exception:
        return []

def search_arxiv(q: str) -> List[Dict]:
    try:
        xml = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{q}", "start": 0, "max_results": MAX_HITS},
            headers={"User-Agent": "litcrit/0.1"},
            timeout=10,
        ).text
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        res = []
        for e in root.findall("a:entry", ns):
            res.append(
                {
                    "title": e.find("a:title", ns).text.strip(),
                    "authors": ", ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)[:3]),
                    "year": e.find("a:published", ns).text[:4],
                    "url": e.find("a:id", ns).text,
                    "source": "arXiv",
                    "keyword": q,
                }
            )
        return res
    except Exception:
        return []

def search_semantic(q: str) -> List[Dict]:
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": q, "limit": MAX_HITS, "fields": "title,year,url,authors"},
            timeout=10,
        ).json()
        return [
            {
                "title": p["title"],
                "authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]),
                "year": p.get("year"),
                "url": p.get("url"),
                "source": "Semantic Scholar",
                "keyword": q,
            } for p in resp.get("data", [])
        ]
    except Exception:
        return []

def do_full_search(term: str) -> List[Dict]:
    out: List[Dict] = []
    out.extend(search_pubmed(term))
    out.extend(search_crossref(term))
    out.extend(search_arxiv(term))
    out.extend(search_semantic(term))
    return out

###############################################################################
# Report
###############################################################################

def build_report(meta: Dict, keywords: List[str], results: List[Dict]) -> str:
    wrap = textwrap.TextWrapper(width=90, subsequent_indent="   ")
    lines = [
        "# Critique & Source Report",
        f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M}",
        "\n## Analysierte Datei",
        meta['filename'],
        "\n## Schlüsselwörter",
        ", ".join(keywords) or "-",
        "\n---",
    ]
    for kw in keywords:
        hits = [res for res in results if res["keyword"] == kw]
        if not hits:
            continue
        lines.append(f"\n### {kw}")
        for r in hits:
            title = r["title"][:120] + ("…" if len(r["title"]) > 120 else "")
            year  = f" ({r['year']})" if r.get("year") else ""
            auth  = f" – {r['authors']}" if r.get("authors") else ""
            lines.append(f"* **{title}**{year}{auth} – {r['source']}\n  {r['url']}")
    return "\n".join(lines)

def save_report(txt: str, stem: str) -> Path:
    DESKTOP.mkdir(exist_ok=True)
    fn = f"{stem}_report_{dt.datetime.now():%Y%m%d_%H%M}.txt"
    path = DESKTOP / fn
    path.write_text(txt, encoding="utf-8")
    return path

###############################################################################
# Streamlit GUI
###############################################################################

st.set_page_config(page_title="Literature Critic", layout="wide")
st.title("📚 Literature Critic – multilinguales Quellen-Radar")

uploaded = st.file_uploader("PDF hochladen", type=["pdf"])

if uploaded:
    # ---------- tmp-Datei sichern ----------
    tmp_path = st.session_state.get("tmp_pdf")
    if not tmp_path or not Path(tmp_path).is_file():
        tmp_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_fd.write(uploaded.read())
        tmp_fd.close()
        tmp_path = tmp_fd.name
        st.session_state["tmp_pdf"] = tmp_path

    meta = {"filename": uploaded.name, "path": tmp_path}

    # ---------- PDF → Text ----------
    try:
        with st.spinner("📖 PDF wird gelesen …"):
            fulltext = extract_text(tmp_path)
    except Exception as e:
        st.error(f"Fehler beim PDF-Parsing: {e}")
        st.stop()

    # ---------- Sprache & spaCy ----------
    lang = detect(fulltext[:5000]) if fulltext.strip() else "en"
    nlp = load_spacy_model(lang)
    st.info(f"Erkannte Sprache: **{lang}**")

    # ---------- Keywords ----------
    with st.spinner("🔑 Schlüsselwörter …"):
        keywords = extract_keywords(fulltext, nlp, 25)
    st.write("**Top-Keywords:**", ", ".join(keywords))

    # ---------- Übersetzungen ----------
    all_terms = set(keywords)
    with st.spinner("🌐 Übersetzungen …"):
        for tgt in TARGET_LANGS:
            if tgt == lang:
                continue
            all_terms.update(translate_terms(keywords, tgt))
    st.write(f"Suchbegriffe gesamt ({len(all_terms)}):", ", ".join(list(all_terms)[:40]), "…")

    # ---------- Recherche ----------
    results: List[Dict] = []
    bar = st.progress(0)
    for i, term in enumerate(all_terms):
        bar.progress((i+1)/len(all_terms), text=term)
        results.extend(do_full_search(term))
    bar.empty()
    st.success(f"Recherche abgeschlossen – {len(results)} Treffer.")

    # ---------- Report ----------
    report = build_report(meta, keywords, results)
    txt_path = save_report(report, Path(uploaded.name).stem)

    st.download_button("⬇️ Report herunterladen", report, file_name=txt_path.name, mime="text/markdown")
    st.subheader("Vorschau")
    st.markdown(report)
else:
    st.info("Bitte zuerst eine PDF hochladen.")