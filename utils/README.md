# 🧰 utils/ — Utility Toolkit for Data Projects

Este diretório contém **funções utilitárias** usadas pelos notebooks do template para **ingestão**, **limpeza**, **engenharia de atributos**, **codificação**, **escala**, **datas**, **texto**, **catálogo de DataFrames** e **exportação**.  
O módulo principal é **`utils_data.py`**.

> Dica: importar funções específicas conforme a etapa do notebook, por exemplo:
>
> ```python
> from utils.utils_data import (
>     load_table_simple, basic_overview, strip_whitespace, infer_numeric_like,
>     simple_impute_with_flags, detect_outliers_iqr, deduplicate_rows,
>     encode_categories_safe, scale_numeric_safe, apply_encoding_and_scaling,
>     parse_dates_with_report, expand_date_features, build_calendar_from,
>     extract_text_features, TableStore, save_table, save_named_interims,
>     list_directory_files
> )
> ```

---

## 📦 Ingestão & Exportação

### `load_csv(filepath, **read_kwargs) -> pd.DataFrame`
Carrega CSV com logging padronizado. Aceita os mesmos parâmetros do `pandas.read_csv` (ex.: `sep`, `encoding`, `low_memory`).

### `save_parquet(df, filepath) -> None`
Salva um DataFrame em **Parquet**. Cria diretórios pais automaticamente.

### `infer_format_from_suffix(path) -> str`
Deduz o formato do arquivo a partir da extensão (`csv`/`parquet`).

### `load_table_simple(path, fmt=None, read_opts=None) -> pd.DataFrame`
Leitura simples e consistente de CSV/Parquet. Se `fmt=None`, detecta pelo sufixo. Útil na etapa **📥 Ingestão & Visão Rápida**.

### `save_table(df, path) -> None`
Salva respeitando a **extensão do caminho**: se terminar em `.csv`, grava CSV; se `.parquet`, grava Parquet.

### `save_named_interims(named_frames, base_dir, fmt="parquet") -> None`
Salva **múltiplos** DataFrames nomeados em `data/interim/` usando a convenção `<nome>_interim.<fmt>`.

### `list_directory_files(dir_path, pattern="*", sort_by="name") -> pd.DataFrame`
Lista arquivos de um diretório (nome, extensão, tamanho e data de modificação). Útil para configurar rapidamente o bloco `SOURCES`.

---

## 🔎 Perfil & Otimização

### `basic_overview(df) -> dict`
Retorna shape, colunas, dtypes, memória e contagem de nulos (para log/manifest).

### `reduce_memory_usage(df) -> pd.DataFrame`
Faz **downcast** de inteiros e floats para reduzir uso de memória. Loga antes/depois.

### `strip_whitespace(df) -> pd.DataFrame`
Remove espaços excedentes em colunas textuais (`object`).

### `infer_numeric_like(df, columns=None, min_ratio=0.9, create_new_col_when_partial=True, blacklist=None, whitelist=None) -> (df, report_df)`
Converte strings “parecidas com número” em valores numéricos, com auditoria:
- Detecta porcentagens (`%`) e normaliza separadores (`1.234,56` ↔ `1,234.56`).
- Se **`ratio >= min_ratio`**, sobrescreve a coluna; caso parcial, cria `<col>_num` (se habilitado).
- Retorna um **relatório** com `column`, `action`, `ratio`, `converted`, `non_convertible`, `examples`.

---

## 🩹 Faltantes & Outliers & Duplicatas

### `missing_report(df) -> pd.DataFrame`
Tabela com `missing_rate` e `missing_count` por coluna.

### `simple_impute(df) -> pd.DataFrame`
Imputação “simples”: numéricos → mediana; categóricos → moda.

### `simple_impute_with_flags(df) -> pd.DataFrame`
Igual ao anterior, mas adiciona **flags booleanas** `was_imputed_<col>` marcando linhas preenchidas (rastreabilidade).

### `detect_outliers_iqr(df, cols=None) -> pd.DataFrame`
Cria colunas `<col>_is_outlier` usando método **IQR** (robusto a assimetrias).

### `detect_outliers_zscore(df, threshold=3.0, cols=None) -> pd.DataFrame`
Cria colunas `<col>_is_outlier` a partir de **Z-score** (assume distribuição ~normal).

### `deduplicate_rows(df, subset=None, keep="first", log_path=None, return_report=False)`
Remove duplicatas e, opcionalmente, **loga** as linhas duplicadas em CSV.
- `subset`: colunas que definem a chave; `None` = linha inteira.
- `keep`: `"first" | "last" | False` (remove todas as repetições).
- `return_report=True` retorna `(df_limpo, dups_df, resumo_df)`.

---

## 🔤 Categóricas & 🔢 Numéricas

> Use as versões **_safe_** abaixo para mais controle (exclusões e avisos).

### `encode_categories(df, encoding="onehot") -> (df, meta)`
Wrapper simples (usa scikit-learn). Converte todas as categóricas do DF passado.
- `encoding="onehot" | "ordinal"`
- `meta` inclui mapeamentos de categorias.

### `scale_numeric(df, method="standard") -> (df, meta)`
Padroniza/normaliza **todas** as colunas numéricas do DF passado.
- `method="standard" | "minmax"`

### `encode_categories_safe(df, method="onehot", exclude_cols=None, high_card_threshold=50) -> (df, meta)`
Codificação com **exclusão de colunas** (ex.: `["Churn","customerID"]`) e **aviso de alta cardinalidade**.
- `meta`: colunas categóricas tratadas, excluídas e aviso de cardinalidade.

### `scale_numeric_safe(df, method="standard", exclude_cols=None, only_continuous=True) -> (df, meta)`
Escala **apenas** as numéricas **contínuas** (ignora dummies/booleanas) e permite excluir colunas-alvo.

### `apply_encoding_and_scaling(df, encode_cfg=None, scale_cfg=None) -> (df, encoding_meta, scaling_meta)`
Orquestra **encode → scale** com configs:
- `encode_cfg = {enabled, type, exclude_cols, high_card_threshold}`
- `scale_cfg  = {enabled, method, exclude_cols, only_continuous}`

---

## 📅 Datas

### `detect_date_candidates(df, pattern) -> list[str]`
Encontra colunas candidatas a data por **regex** ou dtype datetime já existente.

### `_maybe_to_datetime(s, dayfirst, utc, formats) -> pd.Series` _(interno)_
Tenta converter com formatos explícitos; se falhar, usa fallback genérico.

### `parse_dates_with_report(df, cfg) -> (df, parse_report, parsed_cols)`
Converte colunas de data e retorna relatório com:
- `column`, `parsed_ratio`, `converted` (aceita se `parsed_ratio >= min_ratio`).
- `cfg`: `detect_regex`, `explicit_cols`, `dayfirst`, `utc`, `formats`, `min_ratio`.

### `expand_date_features(df, cols, features=None, prefix_mode="auto", fixed_prefix=None) -> list[str]`
Gera features como `*_year`, `*_month`, `*_day`, `*_dow`, `*_quarter`, `*_week`, `*_is_month_start`, `*_is_month_end`.

### `build_calendar_from(df, date_col, freq="D") -> pd.DataFrame`
Cria uma **dimensão calendário** (`dim_date`) com atributos de data entre o min/max observados.

---

## 📝 Texto

### `extract_text_features(df, lower=True, strip_collapse_ws=True, keywords=None, blacklist=None, export_summary=False, summary_dir=None) -> (df, summary_df)`
Extrai métricas simples de colunas textuais (`object`):
- Comprimento, contagem de palavras, contagem de **letras** e **dígitos**.
- Flags de **palavras-chave** `*_has_<kw>` (case-insensitive).
- `summary_df` lista colunas processadas e flags criadas; pode salvar CSV.

---

## 📚 Catálogo de DataFrames

### `TableStore`
Catálogo leve para gerenciar múltiplos DataFrames nomeados com um **“current”**.
- **Principais métodos**
  - `add(name, df, set_current=False)` — registra/atualiza.
  - `get(name=None)` — obtém o df de `name` (ou o atual).
  - `use(name)` — torna `name` atual e retorna o df.
  - `list()` — inventário (nome, linhas, colunas, memória).
  - `rename(old, new)` / `drop(name)`.
- **Acesso estilo dicionário:** `T["features_v1"]`.

---

## 🧪 Exemplos rápidos

### 1) Ingestão + visão geral
```python
df = load_table_simple(RAW_DIR / "dataset.csv", read_opts={"encoding":"utf-8", "sep": ","})
print(basic_overview(df))
```

### 2) Limpeza e tipagem
```python
df = strip_whitespace(df)
df, rep = infer_numeric_like(df, min_ratio=0.9, blacklist=["customerID"])
```

### 3) Faltantes, outliers e duplicatas
```python
df = simple_impute_with_flags(df)
df = detect_outliers_iqr(df)
df = deduplicate_rows(df, subset=["customerID"], keep="first", log_path=REPORTS_DIR/"duplicates.csv")
```

### 4) Datas + calendário
```python
cfg = {"detect_regex": r"(date|data|_dt$|_date$)", "min_ratio": 0.8}
df, parse_report, parsed = parse_dates_with_report(df, cfg)
created = expand_date_features(df, parsed)
dim_date = build_calendar_from(df, date_col="order_date", freq="D")
```

### 5) Texto
```python
df, text_sum = extract_text_features(df, keywords=["error", "cancel"], blacklist=["customerID"])
```

### 6) Encode & Scale (safe)
```python
ENCODE_CFG = {"enabled": True, "type": "onehot", "exclude_cols": ["Churn","customerID"], "high_card_threshold": 50}
SCALE_CFG  = {"enabled": True, "method": "standard", "exclude_cols": ["Churn"], "only_continuous": True}
df, enc_meta, scl_meta = apply_encoding_and_scaling(df, ENCODE_CFG, SCALE_CFG)
```

### 7) Catálogo de DataFrames
```python
T = TableStore(initial={"main": df}, current="main")
df = T.get()
T.add("features_v1", df, set_current=True)
display(T.list())
```

---

## ✅ Dependências
- `pandas`, `numpy`
- `scikit-learn` (para codificação/escala)
- Python ≥ 3.10 recomendado

---

## 🔖 Exportações (API do módulo)
As principais funções/classes expostas via `__all__` incluem:
- Ingestão/Exportação: `load_csv`, `save_parquet`, `save_table`, `load_table_simple`, `infer_format_from_suffix`, `save_named_interims`
- Perfil/Otimização: `basic_overview`, `reduce_memory_usage`, `strip_whitespace`, `infer_numeric_like`, `missing_report`, `simple_impute`, `simple_impute_with_flags`
- Qualidade: `detect_outliers_iqr`, `detect_outliers_zscore`, `deduplicate_rows`
- Categ/Num: `encode_categories`, `scale_numeric`, `encode_categories_safe`, `scale_numeric_safe`, `apply_encoding_and_scaling`
- Datas: `parse_dates_with_report`, `expand_date_features`, `build_calendar_from`, `detect_date_candidates`
- Texto: `extract_text_features`
- Catálogo: `TableStore`
- Utilidade: `list_directory_files`

---

## 🧭 Convenções
- **Sufixos de auditoria:** `_is_outlier`, `was_imputed_<col>`, `<col>_num`, `*_has_<kw>`.
- **Logs** são emitidos pelo `logger` do módulo (gravados em `reports/data_preparation.log` quando configurado no notebook).
- Funções “_safe_” priorizam **previsibilidade** e **controle explícito** ao custo de mais parâmetros.
