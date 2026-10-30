from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_citation_metadata_agrees_with_bibtex_and_readme():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    paper = citation["preferred-citation"]
    bibtex = (ROOT / "citation.bib").read_text()
    readme = (ROOT / "README.md").read_text()
    author = paper["authors"][0]

    assert citation["cff-version"] == "1.2.0"
    assert paper["type"] == "conference-paper"
    assert author["given-names"] == "Shih-Yao"
    assert author["family-names"] == "Lin"
    assert author["affiliation"] == "Independent Researcher"
    assert str(paper["year"]) == "2027"
    assert paper["collection-title"] in bibtex
    assert citation["authors"] == paper["authors"]
    assert paper["title"] in bibtex.replace("{QA}", "QA")
    assert f'{author["family-names"]}, {author["given-names"]}' in bibtex
    assert bibtex.strip() in readme
    assert "doi" not in paper
    assert "start" not in paper
    assert "end" not in paper
