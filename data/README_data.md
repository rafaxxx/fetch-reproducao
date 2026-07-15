# Dados do Experimento

Os dados utilizados neste experimento são de acesso público e disponibilizados pelos autores originais do FETCH.

## Dataset de teste (utilizado neste estudo)

```bash
wget http://astro.phys.wvu.edu/fetch/test_set/test_data.hdf5
```

- **Tamanho:** ~5,6 GB
- **Formato:** HDF5 com três datasets:
  - `data_freq_time`: array (13983, 256, 256, 1) float32 — imagens frequência × tempo
  - `data_dm_time`: array (13983, 256, 256, 1) float32 — imagens DM × tempo
  - `data_labels`: array (13983,) bool — rótulos (True = FRB, False = RFI)
- **Composição:** 6.664 FRBs simulados + 7.319 RFIs reais (Green Bank Observatory)

## Dataset de treinamento (não utilizado diretamente neste estudo)

```bash
wget http://astro.phys.wvu.edu/fetch/train_set/train_data.hdf5
```

- **Tamanho:** ~16,4 GB
- **Nota:** os pesos pré-treinados dos 11 modelos são baixados automaticamente
  do Zenodo (DOI: 10.5281/zenodo.5029590) pelo utilitário `get_model()` do FETCH,
  dispensando o re-treinamento para fins de reprodução da inferência.

## Estrutura do HDF5

```python
import h5py
with h5py.File('test_data.hdf5', 'r') as f:
    print(list(f.keys()))
    # ['data_dm_time', 'data_freq_time', 'data_labels']
    print(f['data_freq_time'].shape)  # (13983, 256, 256, 1)
    print(f['data_labels'][:10])      # [False False False ...]
```

## Nota sobre disponibilidade

O servidor `astro.phys.wvu.edu` é mantido pelo West Virginia University. Durante a
execução deste experimento, verificou-se que o servidor pode bloquear requisições
provenientes de IPs de provedores de nuvem (Google Cloud, AWS, etc.).
Recomenda-se baixar os dados localmente e fazer upload para o Colab via Google Drive
caso o `wget` direto não funcione.
