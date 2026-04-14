#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
confusion_metrics.py - Calcula matriz de confusion y metricas de clasificacion.

Uso rapido:
  py -3 confusion_metrics.py --csv resultados.csv --true-col y_true --pred-col y_pred

CSV esperado:
  y_true,y_pred,y_proba
  DNS,DNS,0.91
  DNS,SANO,0.42
  SANO,SANO,0.10

Campos:
- y_true: etiqueta real
- y_pred: etiqueta predicha
- y_proba (opcional): probabilidad de clase positiva en [0, 1] para AUC
"""

import argparse
import csv
import json
from typing import List, Tuple, Optional


def _to_binary(value: str, positive_label: str) -> int:
    return 1 if str(value).strip().lower() == str(positive_label).strip().lower() else 0


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den != 0 else 0.0


def confusion_counts(y_true_bin: List[int], y_pred_bin: List[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true_bin, y_pred_bin):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, fp, tn, fn


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    positive_label: str,
    y_proba: Optional[List[float]] = None,
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener el mismo tamaño")

    y_true_bin = [_to_binary(v, positive_label) for v in y_true]
    y_pred_bin = [_to_binary(v, positive_label) for v in y_pred]

    tp, fp, tn, fn = confusion_counts(y_true_bin, y_pred_bin)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "n": len(y_true),
        "positive_label": positive_label,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": None,
    }

    if y_proba is not None and len(y_proba) == len(y_true):
        try:
            from sklearn.metrics import roc_auc_score
            metrics["auc"] = float(roc_auc_score(y_true_bin, y_proba))
        except Exception:
            metrics["auc"] = None

    return metrics


def load_from_csv(
    csv_path: str,
    true_col: str,
    pred_col: str,
    proba_col: Optional[str] = None,
) -> Tuple[List[str], List[str], Optional[List[float]]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    y_proba: Optional[List[float]] = [] if proba_col else None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(str(row[true_col]).strip())
            y_pred.append(str(row[pred_col]).strip())
            if proba_col:
                raw = str(row.get(proba_col, "")).strip().replace(",", ".")
                if raw == "":
                    y_proba.append(0.0)
                else:
                    y_proba.append(float(raw))

    return y_true, y_pred, y_proba


def print_report(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 64)
    print("MATRIZ DE CONFUSION")
    print("=" * 64)
    print("                 Predicho")
    print("                Pos    Neg")
    print(f"Real Pos      {cm['TP']:4d}  {cm['FN']:4d}")
    print(f"Real Neg      {cm['FP']:4d}  {cm['TN']:4d}")

    print("\n" + "=" * 64)
    print("METRICAS")
    print("=" * 64)
    print(f"Muestras (n): {metrics['n']}")
    print(f"Accuracy    : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision   : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall      : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"Specificity : {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"F1 Score    : {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    if metrics.get("auc") is not None:
        print(f"AUC ROC     : {metrics['auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matriz de confusion y metricas para tesis")
    parser.add_argument("--csv", required=True, help="Ruta del CSV con resultados")
    parser.add_argument("--true-col", default="y_true", help="Nombre de columna real")
    parser.add_argument("--pred-col", default="y_pred", help="Nombre de columna predicha")
    parser.add_argument("--proba-col", default=None, help="Nombre de columna de probabilidad (opcional)")
    parser.add_argument("--positive-label", default="DNS", help="Etiqueta positiva (ej: DNS)")
    parser.add_argument("--json-out", default=None, help="Guardar metricas en JSON")

    args = parser.parse_args()

    y_true, y_pred, y_proba = load_from_csv(
        csv_path=args.csv,
        true_col=args.true_col,
        pred_col=args.pred_col,
        proba_col=args.proba_col,
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        positive_label=args.positive_label,
        y_proba=y_proba,
    )

    print_report(metrics)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetricas guardadas en: {args.json_out}")


if __name__ == "__main__":
    main()
