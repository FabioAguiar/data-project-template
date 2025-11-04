# ⚙️ `config/` — Guia de Parâmetros

Esta pasta contém os arquivos de **configuração** utilizados pelos notebooks do projeto.  
O arquivo principal é o **`defaults.json`**, que define os parâmetros padrão para a fase de **preparação e padronização de dados (N1)**.  
Você pode (opcionalmente) criar um **`local.json`** para sobrescrever configurações do `defaults.json` **sem alterar** o template original.

---

## 📘 Estrutura dos Arquivos

- **`defaults.json`** → Configurações padrão aplicadas a todos os projetos.  
- **`local.json`** → Arquivo **opcional** para ajustes locais (sobrepõe os valores do `defaults.json`).

O sistema faz *merge* automático de ambos (prioridade para `local.json`).

---

## 🧩 Organização das Seções no `defaults.json`

O arquivo agora está **estruturado em blocos temáticos**, cada um controlando um estágio do pipeline.

| Seção | Descrição |
|-------|------------|
| **Outliers** | Configura a detecção de outliers (`iqr` ou `zscore`). Permite excluir colunas e definir limites. |
| **Deduplicate** | Controla remoção de duplicatas, política de retenção e log de duplicadas. |
| **Feature Engineering** | Regras para geração automática de novas features (log1p, proporções, partes de datas). |
| **Reporting** | Ativa ou desativa geração de manifestos e logs complementares. |
| **Target** | Define a variável-alvo e o mapeamento das classes (`Yes`/`No`). |
| **Dates** | Controla a detecção, parsing e criação de *features* de data. |

---

## 🔧 Parâmetros Globais (nível raiz)

| Parâmetro | Tipo | Valor padrão | Descrição |
|------------|------|---------------|------------|
| `infer_types` | bool | `true` | Otimiza tipos (ex.: *downcast* numérico) para reduzir memória. |
| `cast_numeric_like` | bool | `true` | Converte textos “parecidos com números” em numéricos (respeitando *ratio* mínimo). |
| `strip_whitespace` | bool | `true` | Remove espaços em branco excedentes nas colunas textuais. |
| `handle_missing` | bool | `true` | Ativa o tratamento de valores nulos/ausentes. |
| `missing_strategy` | str | `"simple"` | Estratégia de imputação: `"simple"` (mediana/moda). |
| `detect_outliers` | bool | `true` | Ativa a detecção de outliers nas colunas numéricas. |
| `outlier_method` | str | `"iqr"` | Método de detecção de outliers. |
| `normalize_categories` | bool | `true` | Padroniza rótulos categóricos equivalentes. |
| `encode_categoricals` | bool | `true` | Ativa codificação de variáveis categóricas. |
| `encoding_type` | str | `"onehot"` | Tipo de codificação: `"onehot"` (seguro) ou `"ordinal"`. |
| `scale_numeric` | bool | `false` | Ativa o escalonamento de colunas numéricas. |
| `scaler` | str | `"standard"` | Método de escala: `"standard"` ou `"minmax"`. |
| `date_features` | bool | `true` | Ativa criação de *features* de data. |
| `text_features` | bool | `true` | Ativa criação de *features* simples de texto. |
| `export_interim` | bool | `true` | Exporta dataset intermediário para `data/interim/`. |
| `export_processed` | bool | `true` | Exporta dataset final para `data/processed/`. |

---

## 🧮 Seção: `outliers`

```json
"outliers": {
  "cols": null,
  "exclude_cols": ["customerID"],
  "exclude_binaries": true,
  "iqr_factor": 1.5,
  "z_threshold": 3.0,
  "persist_summary": true,
  "persist_relpath": "outliers/summary.csv"
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `cols` | list/null | Colunas específicas para aplicar a detecção (ou `null` para todas numéricas). |
| `exclude_cols` | list | Colunas a ignorar. |
| `exclude_binaries` | bool | Evita analisar colunas 0/1 como outliers. |
| `iqr_factor` | float | Multiplicador do intervalo interquartil (IQR). |
| `z_threshold` | float | Limite do Z-score. |
| `persist_summary` | bool | Salva CSV com resumo de outliers detectados. |
| `persist_relpath` | str | Caminho relativo dentro de `reports/` para o resumo. |

---

## 🔁 Seção: `deduplicate`

```json
"deduplicate": {
  "subset": null,
  "keep": "first",
  "log_enabled": true,
  "log_relpath": "duplicates.csv"
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `subset` | list/null | Colunas que definem duplicidade (`null` = linha inteira). |
| `keep` | str/bool | Política de retenção: `"first"`, `"last"`, `false` (remove todas). |
| `log_enabled` | bool | Gera log CSV de duplicatas removidas. |
| `log_relpath` | str | Caminho relativo do log (dentro de `reports/`). |

---

## 🧠 Seção: `feature_engineering`

```json
"feature_engineering": {
  "enable_default_rules": true,
  "log1p_cols": [],
  "ratios": [],
  "binaries": [],
  "date_parts": []
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `enable_default_rules` | bool | Ativa regras básicas automáticas. |
| `log1p_cols` | list | Colunas para aplicar transformação log1p. |
| `ratios` | list | Lista de pares ou expressões de proporção entre colunas. |
| `binaries` | list | Criação de colunas binárias baseadas em condições simples. |
| `date_parts` | list | Extração de partes de data personalizadas. |

---

## 🗓️ Seção: `dates`

```json
"dates": {
  "detect_regex": "(date|data|dt_|_dt$|_date$|_at$|time|timestamp|created|updated)",
  "explicit_cols": [],
  "dayfirst": false,
  "utc": false,
  "formats": [],
  "min_ratio": 0.8,
  "report_path": "date_parse_report.csv"
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `detect_regex` | str | Regex para detectar automaticamente colunas de data. |
| `explicit_cols` | list | Lista de colunas a forçar como datetime. |
| `dayfirst` | bool | Interpreta datas como D/M/Y. |
| `utc` | bool | Converte para timezone UTC. |
| `formats` | list | Formatos explícitos aceitos. |
| `min_ratio` | float | Mínimo de sucesso no parsing para considerar válida. |
| `report_path` | str | Caminho do relatório de parsing salvo em `reports/`. |

---

## 🎯 Seção: `target`

```json
"target": {
  "name": "Churn",
  "source": "Churn",
  "positive": "Yes",
  "negative": "No"
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `name` | str | Nome da coluna de destino (após processamento). |
| `source` | str | Coluna de origem no dataset cru. |
| `positive` | str | Valor representando a classe positiva. |
| `negative` | str | Valor representando a classe negativa. |

> Essa configuração é utilizada na função `ensure_target_from_config()` e no `meta.json` exportado pelo N1.

---

## 🧾 Seção: `reporting`

```json
"reporting": {
  "manifest_enabled": true
}
```

| Parâmetro | Tipo | Descrição |
|------------|------|------------|
| `manifest_enabled` | bool | Controla se o `manifest.json` é gerado automaticamente após a execução. |

---

## ⚙️ Exemplos práticos

### 🧪 Exemplo 1 — desativar outliers e ativar escala MinMax
```json
{
  "detect_outliers": false,
  "scale_numeric": true,
  "scaler": "minmax"
}
```

### 🧩 Exemplo 2 — definir subset de deduplicação e ajustar target
```json
{
  "deduplicate": {
    "subset": ["customerID"],
    "keep": "last"
  },
  "target": {
    "name": "Exited",
    "source": "Exited",
    "positive": "1",
    "negative": "0"
  }
}
```

---

## ✅ Recomendações

1. **Mantenha o `defaults.json` estável** e use `local.json` para ajustes por projeto.  
2. **Versione** ambos os arquivos para manter rastreabilidade.  
3. **Valide** alterações com uma execução curta do N1 antes de usar em produção.  
4. Consulte o **`manifest.json`** e o **`meta.json`** para auditoria rápida dos parâmetros aplicados.

