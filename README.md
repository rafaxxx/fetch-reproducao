# Reprodução do FETCH — Fast Extragalactic Transient Candidate Hunter

Repositório de reprodução do experimento descrito em:

> Agarwal, D. et al. **FETCH: A deep-learning based classifier for fast transient classification.** *Monthly Notices of the Royal Astronomical Society*, 497(2):1661–1674, 2020. DOI: [10.1093/mnras/staa1856](https://doi.org/10.1093/mnras/staa1856)

Este trabalho é parte de um projeto de pesquisa reproduzível para a disciplina de Metodologia Científica em Computação.

---

## Estrutura do repositório

```
.
├── src/                  # Código-fonte principal
│   └── inference.py      # Script de inferência com os 11 modelos do FETCH
├── data/                 # Instruções para obtenção dos dados
│   └── README_data.md
├── scripts/              # Notebook Colab para execução completa
│   └── FETCH_reproducao.ipynb
├── results/              # Resultados gerados pelo experimento
│   └── resumo_metricas.csv
├── docs/                 # Documentação adicional
│   └── desvios_protocolo.md
├── README.md
└── requirements.txt
```

---

## Requisitos

- Python 3.10+
- Google Colab (recomendado) com GPU T4 ou superior
- ~6 GB de espaço em disco (dataset de teste)

Instale as dependências:

```bash
pip install tf_keras scikit-learn h5py pandas numpy matplotlib seaborn
```

> **Importante:** defina a variável de ambiente `TF_USE_LEGACY_KERAS=1` **antes** de importar o TensorFlow. O código do FETCH (branch `tf2`) é incompatível com Keras 3, que é instalado por padrão a partir do TensorFlow 2.16.

---

## Como reproduzir

### Opção 1 — Google Colab (recomendado)

1. Abra o notebook `scripts/FETCH_reproducao.ipynb` no Google Colab
2. Selecione Runtime → Change runtime type → GPU
3. Execute as células em ordem

### Opção 2 — Script Python

1. Clone o repositório FETCH (branch `tf2`):
```bash
git clone -b tf2 https://github.com/devanshkv/fetch.git
cd fetch
python setup.py install
```

2. Baixe o dataset de teste (~5,6 GB):
```bash
wget http://astro.phys.wvu.edu/fetch/test_set/test_data.hdf5
```

3. Execute o script de inferência:
```bash
TF_USE_LEGACY_KERAS=1 python src/inference.py \
    --data test_data.hdf5 \
    --model a \
    --output results/
```

---

## Dados

Os dados utilizados neste experimento são de acesso público:

| Dataset | Fonte | Tamanho |
|---|---|---|
| Test set (13.983 candidatos) | [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/test_set/test_data.hdf5) | ~5,6 GB |
| Train set | [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/train_set/train_data.hdf5) | ~16,4 GB |

Veja `data/README_data.md` para mais detalhes.

---

## Resultados

Métricas obtidas para o modelo `a` sobre o conjunto de teste original (13.983 candidatos):

| Métrica | Reproduzido | Original (Agarwal et al., 2020) |
|---|---|---|
| Acurácia | 98,56% | 99,88% |
| Recall | 97,31% | 99,92% |
| Precisão | 99,65% | N/D |
| F1-Score | 98,47% | 99,87% |

---

## Desvios documentados do protocolo original

Ver `docs/desvios_protocolo.md` para descrição completa. Resumo:

1. Uso de `tf_keras` + `TF_USE_LEGACY_KERAS=1` em vez do TensorFlow 1.x original
2. Avaliação restrita ao modelo `a` por limitações de RAM do ambiente Colab gratuito
3. Filterbanks de exemplo inacessíveis a partir do Colab (servidor universitário bloqueando IPs de nuvem)

---

## Artefatos

| Artefato | Plataforma | Link |
|---|---|---|
| Código-fonte | GitHub | https://github.com/rafaxxx/fetch-reproducao |
| Notebook + resultados | Zenodo | https://zenodo.org/records/21385263 (DOI: 10.5281/zenodo.21385263 |
| Artigo | Zenodo | https://zenodo.org/records/21385263 (DOI: 10.5281/zenodo.21385263 |

---

## Citação

Se você usar este trabalho, cite o artigo original:

```bibtex
@article{agarwal2020fetch,
  author  = {Agarwal, Devansh and Aggarwal, Kshitij and Burke-Spolaor, Sarah
             and Lorimer, Duncan R. and Garver-Daniels, Nathaniel},
  title   = {{FETCH}: A deep-learning based classifier for fast transient classification},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume  = {497},
  number  = {2},
  pages   = {1661--1674},
  year    = {2020},
  doi     = {10.1093/mnras/staa1856}
}
```

---

## Licença

Este repositório de reprodução é disponibilizado sob licença MIT.
O código do FETCH original está sob licença GPL-3.0.
