# Desvios Documentados do Protocolo Original

Este documento registra todos os desvios em relação ao protocolo descrito em
Agarwal et al. (2020), conforme recomendado pelas diretrizes de estudos de reprodução.

---

## Desvio 1 — Versão do TensorFlow/Keras

**Original:** TensorFlow 1.x com Keras 2 nativo, GPU Tesla P100  
**Reproduzido:** TensorFlow 2.20 com pacote de compatibilidade `tf_keras` (`TF_USE_LEGACY_KERAS=1`), GPU Tesla T4 (Google Colab)

**Causa:** o `requirements.txt` do branch `tf2` do repositório instala Keras 3 por
padrão a partir do TensorFlow 2.16, quebrando a compatibilidade com o código do FETCH
(ver [issue #36](https://github.com/devanshkv/fetch/issues/36)).

**Solução aplicada:** instalação do pacote `tf_keras` e definição de
`TF_USE_LEGACY_KERAS=1` antes de qualquer `import tensorflow`.

**Impacto potencial:** possível causa da discrepância de ~1,3 pp em acurácia e
~2,6 pp em recall observada em relação aos valores originais.

---

## Desvio 2 — Escopo da avaliação (modelos avaliados)

**Original:** 11 modelos (`a` a `k`) avaliados e comparados  
**Reproduzido:** apenas modelo `a` avaliado

**Causa:** limitação de memória RAM do ambiente Google Colab gratuito (~12 GB).
O carregamento sequencial de modelos com pesos de ~110 MB cada, somado ao dataset
de teste (~3,5 GB descomprimido), excedeu a capacidade disponível.

**Impacto:** RQ2 (variação entre modelos) não foi respondida nesta etapa.

---

## Desvio 3 — Smoke test com filterbanks de exemplo

**Original:** disponível em `astro.phys.wvu.edu/files/askap_frb_180417.tgz`  
**Reproduzido:** não executado

**Causa:** o servidor da WVU bloqueou requisições provenientes de IPs do Google Cloud
durante a execução do experimento, impedindo o download dos filterbanks de exemplo
diretamente no Colab.

**Impacto:** a validação do pipeline com dados reais do ASKAP (RQ3) não foi realizada.

---

## Desvio 4 — Instalação do FETCH

**Original:** `pip install fetch` (conforme README)  
**Reproduzido:** `python setup.py install` (necessário para registrar os scripts
`predict.py` e `train.py` no PATH do sistema)

**Causa:** `pip install -e .` não registra os entry points corretamente
no ambiente Colab com Python 3.12.

---

## Resumo do impacto dos desvios

| Desvio | RQ afetada | Impacto |
|---|---|---|
| TensorFlow/Keras | RQ1 | Provável causa da discrepância de métricas |
| Escopo reduzido | RQ2 | Não respondida |
| Filterbanks inacessíveis | RQ3 | Não respondida |
| Instalação | Nenhuma | Contornado sem impacto nos resultados |
