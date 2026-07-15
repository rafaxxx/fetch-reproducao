"""
inference.py — Script de inferência para reprodução do FETCH

Uso:
    TF_USE_LEGACY_KERAS=1 python inference.py \
        --data test_data.hdf5 \
        --model a \
        --output results/

Referência:
    Agarwal et al. (2020). FETCH: A deep-learning based classifier
    for fast transient classification. MNRAS, 497(2):1661-1674.
    DOI: 10.1093/mnras/staa1856
"""

import os
import sys
import argparse
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)

# Adicionar egg do FETCH ao path (necessário quando instalado via setup.py)
EGG = "/usr/local/lib/python3.12/dist-packages/fetch-0.2.0-py3.12.egg"
if os.path.exists(EGG) and EGG not in sys.path:
    sys.path.insert(0, EGG)

from fetch.utils import get_model  # noqa: E402


MODELOS_DISPONIVEIS = list("abcdefghijk")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferência com modelos FETCH sobre dataset de teste HDF5."
    )
    parser.add_argument(
        "--data", required=True,
        help="Caminho para test_data.hdf5"
    )
    parser.add_argument(
        "--model", default="a",
        choices=MODELOS_DISPONIVEIS + ["all"],
        help="Modelo a avaliar ('a'-'k') ou 'all' para todos os 11 (default: a)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size para inferência (default: 64; reduza para 32 se houver OOM)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Limiar de decisão (default: 0.5)"
    )
    parser.add_argument(
        "--output", default="results/",
        help="Diretório de saída para CSVs e métricas (default: results/)"
    )
    return parser.parse_args()


def carregar_labels(hdf5_path):
    """Carrega apenas os labels do HDF5 (pequeno — ~14 KB)."""
    with h5py.File(hdf5_path, "r") as f:
        labels = f["data_labels"][:].astype(int)
    return labels


def inferencia_batch(model, hdf5_path, n, batch_size):
    """
    Executa inferência em batches lendo o HDF5 sob demanda,
    sem carregar os arrays completos na RAM.
    Retorna array de probabilidades (classe positiva = FRB).
    """
    probs = np.zeros(n, dtype=np.float32)
    with h5py.File(hdf5_path, "r") as f:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ft = f["data_freq_time"][start:end]
            dm = f["data_dm_time"][start:end]
            # Modelo retorna shape (batch, 2) — coluna 1 = P(FRB)
            probs[start:end] = model.predict(
                [ft, dm], batch_size=batch_size, verbose=0
            )[:, 1]
            if start % 2000 == 0:
                print(f"  {start}/{n} candidatos processados...")
    return probs


def avaliar(labels, probs, threshold, modelo_id):
    """Calcula e imprime as métricas de avaliação."""
    preds = (probs >= threshold).astype(int)
    acc  = accuracy_score(labels, preds)
    rec  = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    print(f"\n  Acurácia:  {acc*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  Precisão:  {prec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  Matriz de confusão:\n{cm}")

    return {
        "modelo": modelo_id,
        "acuracia": acc,
        "recall": rec,
        "precisao": prec,
        "f1_score": f1,
        "threshold": threshold,
    }


def salvar_resultados(output_dir, modelo_id, labels, probs, metricas):
    """Salva CSV de probabilidades e CSV de métricas."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV de probabilidades por candidato
    pd.DataFrame({
        "candidate": [f"cand_{i:05d}" for i in range(len(labels))],
        "probability": probs,
        "label": labels,
        "prediction": (probs >= metricas["threshold"]).astype(int),
    }).to_csv(os.path.join(output_dir, f"results_{modelo_id}.csv"), index=True)

    # CSV de métricas resumidas
    pd.DataFrame([metricas]).to_csv(
        os.path.join(output_dir, f"metricas_{modelo_id}.csv"), index=False
    )
    print(f"\n  Resultados salvos em '{output_dir}'")


def main():
    args = parse_args()

    modelos = MODELOS_DISPONIVEIS if args.model == "all" else [args.model]

    print(f"Carregando labels de '{args.data}'...")
    labels = carregar_labels(args.data)
    n = len(labels)
    print(f"  Total: {n} candidatos | FRBs: {labels.sum()} | RFIs: {(labels==0).sum()}")

    resumo = []

    for m in modelos:
        print(f"\n{'='*40}")
        print(f"Avaliando modelo '{m}'...")
        t0 = time.time()

        model = get_model(m)
        probs = inferencia_batch(model, args.data, n, args.batch_size)
        metricas = avaliar(labels, probs, args.threshold, m)
        metricas["tempo_s"] = round(time.time() - t0, 1)
        salvar_resultados(args.output, m, labels, probs, metricas)
        resumo.append(metricas)

        # Liberar memória antes do próximo modelo
        import gc
        import tensorflow as tf
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    # Salvar resumo consolidado
    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(os.path.join(args.output, "resumo_metricas.csv"), index=False)

    print(f"\n{'='*40}")
    print("RESUMO FINAL:")
    print(df_resumo[["modelo", "acuracia", "recall", "precisao", "f1_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
