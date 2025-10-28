# 🧩 Data Project Template

Modelo-base para projetos de **análise** e **engenharia de dados**, com foco em **clareza**, **organização** e **reprodutibilidade**.  
Inclui pipeline de preparação, utilitários prontos em `utils/`, configuração declarativa via `config/` e exportação de artefatos.

---

## 📁 Estrutura do Repositório

```
data-project-template/
├── artifacts/        # saídas auxiliares do projeto (dim_date, métricas, etc.)
├── config/           # defaults.json (obrigatório) e local.json (opcional, overrides)
├── dashboards/       # arquivos de dashboards (Power BI, etc.)
├── data/
│   ├── raw/          # dados brutos 
│   ├── interim/      # camadas intermediárias após limpeza/padrões
│   └── processed/    # dataset final para modelagem/visualização
├── notebooks/        # Jupyter Notebooks do fluxo (N1…Nn)
├── prints/           # screenshots/figuras para documentação
├── reports/          # logs, relatórios CSV/HTML, PDF e manifest.json
├── utils/            # funções reutilizáveis (utils_data.py, etc.)
├── .gitignore
└── README.md
```


---

## 🚀 Fluxo de Trabalho no notebook 01_data_preparation_template

1. **Configuração do Projeto**  
   - Descoberta da raiz, carregamento de `config/defaults.json` (e `local.json` se existir).  
   - Definição de caminhos `data/`, `reports/`, `artifacts/` e logging em `reports/data_preparation.log`.

2. **Configuração de Fontes**  
   - Defina `SOURCES` (caminho, formato e opções) e `MAIN_SOURCE`.  
   - Opcional: `MERGE_STEPS` para encadear *joins* (com checagem básica de chaves).

3. **Ingestão & Visão Rápida**  
   - Leitura de cada fonte (`load_table_simple`) + `basic_overview` e `missing_report`.  
   - Execução de *merge chain* quando configurado.

4. **Catálogo de DataFrames**  
   - `TableStore` centraliza múltiplos `DataFrames` nomeados.  
   - Convenção: manter um `df` “ativo” via `T.get()` ou `T.use("nome")`.

5. **Qualidade & Tipagem**  
   - `strip_whitespace` → limpeza leve de texto.  
   - `infer_numeric_like` → converte strings numéricas respeitando *ratio* mínimo, com *report*.  
   - `reduce_memory_usage` → *downcast* numérico e relatório de memória.  
   - Exportação **interim** (quando habilitado).

6. **Padronização Categórica (pré-engenharia)**  
   - Normalizações explícitas e simples (ex.: `"No internet service" → "No"`).  
   - Mantida próxima da etapa anterior por depender da detecção de tipos/valores.

7. **Tratamento de Faltantes**  
   - `simple_impute_with_flags` (mediana/moda) + colunas `was_imputed_<col>`.  
   - Transparência e rastreabilidade dos preenchimentos.

8. **Detecção de Outliers (opcional)**  
   - **IQR** ou **Z-score** gerando apenas *flags* `*_is_outlier` (decisão de negócio fica fora).

9. **Duplicidades**  
   - `deduplicate_rows` com suporte a `subset`, política `keep` e *log* CSV opcional de duplicatas.  
   - Pode retornar relatório de chaves duplicadas.

10. **Tratamento de Datas**  
    - `parse_dates_with_report` detecta/força colunas de data e audita *parsed_ratio*.  
    - `expand_date_features` cria `*_year`, `*_month`, `*_week`, `*_is_month_*` etc.  
    - Criação de **dim_date** com `build_calendar_from` (opcional).

11. **Tratamento de Texto (opcional)**  
    - `extract_text_features`: tamanho, contagem de palavras/letras/dígitos e *keywords*.  
    - Opção de exportar *summary* em `reports/text_features/`.

12. **Codificação & Escalonamento (opcionais)**  
    - `apply_encoding_and_scaling`: *wrapper* que orquestra  
      `encode_categories_safe` (one-hot/ordinal com exclusões e alerta de cardinalidade)  
      e `scale_numeric_safe` (standard/minmax apenas em contínuas, se desejado).

13. **Exportação de Artefatos**  
    - `save_table` respeita extensão (`.csv`/`.parquet`) para **interim**/**processed**.  
    - Geração de `reports/manifest.json` com shape, colunas e metadados de encode/scale.

---

## ⚙️ Configuração via `config/`

- **`defaults.json`**: parâmetros padrão (obrigatório).  
- **`local.json`**: overrides por projeto/ambiente (opcional).  
- As *flags* mais usadas:  
  - `infer_types`, `cast_numeric_like`, `strip_whitespace`  
  - `handle_missing` + `missing_strategy`  
  - `detect_outliers` + `outlier_method`  
  - `deduplicate` (+ subset/keep/log)  
  - `normalize_categories`  
  - `date_features`, `text_features`, `feature_engineering`  
  - `encode_categoricals` + `encoding_type`  
  - `scale_numeric` + `scaler`  
  - `export_interim`, `export_processed`

> As configurações ativas são registradas no log e no `manifest.json`.

---

## 🧰 Principais Utilitários (`utils/utils_data.py`)

- **Ingestão**: `infer_format_from_suffix`, `load_table_simple`, `merge_chain`  
- **Qualidade/Tipos**: `strip_whitespace`, `infer_numeric_like`, `reduce_memory_usage`, `missing_report`, `simple_impute_with_flags`  
- **Outliers & Duplicidades**: `detect_outliers_iqr`, `detect_outliers_zscore`, `deduplicate_rows`  
- **Datas**: `parse_dates_with_report`, `expand_date_features`, `build_calendar_from`  
- **Texto**: `extract_text_features`  
- **Encode & Scale**: `encode_categories_safe`, `scale_numeric_safe`, `apply_encoding_and_scaling`  
- **Catálogo**: `TableStore`, `save_named_interims`  
- **Arquivos**: `save_table`, `save_parquet`, `list_directory_files`

Um README detalhado dos utilitários está em `utils/UTILS_README.md`.

---

## 🧪 Rodando o Template (resumo)

1. Coloque seus arquivos em `data/raw/`.  
2. (Opcional) Execute no notebook a listagem de arquivos: `list_directory_files(RAW_DIR)` para escolher *sources*.  
3. Configure `SOURCES`, `MAIN_SOURCE` e (se necessário) `MERGE_STEPS`.  
4. Siga as células do pipeline (N1 — Preparação de Dados).  
5. Exporte *interim*/*processed* + `manifest.json`.

---

## 🔒 Boas Práticas

- **Não comite dados sensíveis**. Prefira *placeholders* e `.gitignore`.  
- Documente normalizações e decisões de negócio no README do projeto.  
- Use `local.json` para ajustes de ambiente sem tocar o template.  
- Registre mudanças relevantes nos logs e no `manifest.json`.

---

## 📝 Licença & Créditos

- Licença: MIT (ajuste conforme sua necessidade).  
- Template montado para estudos/portfólio e rápido *bootstrap* de projetos de dados.

## 🚀 Getting Started

### 1) Ambiente
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (Powershell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt  # ou instale as libs do seu stack padrão
```

### 2) Estrutura mínima
Coloque seus arquivos de entrada em `data/raw/`. Exemplo:
```
data/raw/
├── input.csv
└── customers_2025-10-01.csv
```

### 3) Configurações
- O arquivo `config/defaults.json` contém as flags padrão do pipeline.
- Para ajustes locais (sem mexer nos defaults), crie `config/local.json`. Exemplo:
```json
{
  "text_features": true,
  "export_processed": true,
  "scale_numeric": true,
  "scaler": "minmax",
  "normalize_categories": true
}
```
> O projeto faz *merge* de `defaults.json` com `local.json` (local sobrepõe).

### 4) Execução do N1 (Preparação de Dados)
Abra e rode o notebook:
```
notebooks/01_data_preparation.ipynb
```
Saídas esperadas:
- Intermediários em `data/interim/` (se habilitado)
- Processados em `data/processed/` (se habilitado)
- Relatórios e logs em `reports/`

### 5) Dicas
- Mantenha apenas uma **fonte canônica** de dados brutos em `data/raw/`.
- Use nomes descritivos e com datas (`snake_case` + `YYYY-MM-DD`).

---

### 🧭 Filosofia de normalização categórica
Por padrão, mantemos os rótulos exatamente como estão nos dados brutos. A normalização só ocorre quando
`normalize_categories = true`, garantindo **controle explícito** e evitando perda de semântica (ex.: diferenças sutis
de grafia que carregam significado). Essa regra torna a transformação **previsível** e **auditável** — você decide quando
e como normalizar.
