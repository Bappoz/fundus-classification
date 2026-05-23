import pandas as pd
from pathlib import Path
from retino.data.loader import build_labels, verify_files

SAMPLE = Path("data/sample")
META = Path("data/meta")

def test_schema():
    df = build_labels(images_dir=SAMPLE, meta_dir=META)
    assert set(["image","patient_id","label","source","path"]).issubset(df.columns)
    assert df["label"].isin([0,1]).all()

def test_no_duplicate_patient_across_sources():
    df = build_labels(images_dir=SAMPLE, meta_dir=META)
    # patient_id deve ter prefixo de fonte — sem colisão
    brset_pats = set(df[df.source=="brset"]["patient_id"])
    odir_pats  = set(df[df.source=="odir"]["patient_id"])
    assert brset_pats.isdisjoint(odir_pats), "colisão de patient_id entre fontes"

def test_labels_match_folder():
    df = build_labels(images_dir=SAMPLE, meta_dir=META)
    # label 0 só pode vir do BRSET
    assert (df[df.label==0]["source"] == "brset").all()
    # label 1 vem do BRSET ou ODIR
    assert df[df.label==1]["source"].isin(["brset","odir"]).all()