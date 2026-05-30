import pytest
import pandas as pd
from pathlib import Path
from retino.data.loader import build_labels, verify_files, filter_quality
from retino.data.split import split_by_patient

SAMPLE = Path("data/sample")
META   = Path("data/meta")


def get_df():
    df = build_labels(images_dir=SAMPLE, meta_dir=META)
    df = verify_files(df)
    return filter_quality(df)


def test_no_leakage():
    """Nenhum patient_id deve aparecer em mais de um split."""
    train, val, test = split_by_patient(get_df())
    s = [set(d.patient_id) for d in (train, val, test)]
    assert s[0].isdisjoint(s[1]), "leakage treino/val"
    assert s[0].isdisjoint(s[2]), "leakage treino/teste"
    assert s[1].isdisjoint(s[2]), "leakage val/teste"


def test_covers_all_data():
    """União dos splits deve cobrir todo o dataset."""
    df = get_df()
    train, val, test = split_by_patient(df)
    total_split = len(train) + len(val) + len(test)
    assert total_split == len(df), \
        f"perdeu dados: {len(df)} → {total_split}"


def test_ratios_approximate():
    """Splits devem ter tamanhos aproximados ao esperado (±5%)."""
    df = get_df()
    train, val, test = split_by_patient(df, 0.70, 0.15, 0.15)
    n = len(df)
    assert abs(len(train)/n - 0.70) < 0.05
    assert abs(len(val)/n   - 0.15) < 0.05
    assert abs(len(test)/n  - 0.15) < 0.05


def test_reproducible():
    """Mesmo seed deve produzir os mesmos splits."""
    df = get_df()
    t1, v1, e1 = split_by_patient(df, seed=42)
    t2, v2, e2 = split_by_patient(df, seed=42)
    assert set(t1.patient_id) == set(t2.patient_id)
    assert set(v1.patient_id) == set(v2.patient_id)


def test_both_classes_in_all_splits():
    """Estratificação: todos os splits devem ter label 0 e label 1."""
    df = get_df()
    # com amostra pequena pode falhar — só roda se tiver positivos suficientes
    if (df.label == 1).sum() < 6:
        pytest.skip("amostra muito pequena para estratificação")
    train, val, test = split_by_patient(df)
    for name, d in [("treino", train), ("val", val), ("teste", test)]:
        assert (d.label == 0).sum() > 0, f"{name}: sem normais"
        assert (d.label == 1).sum() > 0, f"{name}: sem hipertensivas"