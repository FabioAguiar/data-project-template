# ⚙️ `config/` — Guia de Parâmetros

Esta pasta contém os arquivos de **configuração** utilizados pelos notebooks do projeto.  
O arquivo principal é o **`defaults.json`**, que define os parâmetros de comportamento padrão para a fase de **preparação e padronização de dados**.  
Você pode (opcionalmente) criar um **`local.json`** para sobrescrever parâmetros do `defaults.json` **sem alterar** o template original.

---

## 📘 Estrutura dos Arquivos

- **`defaults.json`** → Configurações padrão aplicadas a todos os projetos.  
- **`local.json`** → Arquivo **opcional** para ajustes locais (sobrepõe os valores do `defaults.json`).

---

## 🔧 Parâmetros Disponíveis (com valores padrão)

> A tabela abaixo reflete **todos os parâmetros** presentes no `defaults.json` fornecido.  
> Tipo → `bool`, `str`, `int`, `null` (ausência/None).

| Parâmetro | Tipo | Valor padrão | Descrição |
|---|---|---|---|
| `infer_types` | bool | `true` | Otimiza tipos (ex.: *downcast* numérico) para reduzir memória. |
| `cast_numeric_like` | bool | `true` | Converte textos “parecidos com números” em numéricos (respeitando *ratio* mínimo). |
| `strip_whitespace` | bool | `true` | Remove espaços em branco excedentes nas colunas textuais. |
| `handle_missing` | bool | `true` | Ativa o tratamento de valores nulos/ausentes. |
| `missing_strategy` | str | `"simple"` | Estratégia de imputação: `"simple"` (mediana/moda). Espaço reservado para técnicas avançadas. |
| `detect_outliers` | bool | `true` | Ativa a detecção de outliers nas colunas numéricas. |
| `outlier_method` | str | `"iqr"` | Método para outliers: `"iqr"` (robusto) ou `"zscore"`. |
| `deduplicate` | bool | `true` | Remove duplicatas. Pode registrar linhas duplicadas e resumo. |
| `deduplicate_subset` | null/str/list | `null` | Subconjunto de colunas para definir duplicidade. `null` = linha inteira. Ex.: `["customerID"]`. |
| `deduplicate_keep` | str/bool | `"first"` | Política de remoção: `"first"`, `"last"` ou `false` (remove todas as repetições). |
| `deduplicate_log` | bool | `true` | Se `true`, salva um CSV com as duplicatas detectadas. |
| `deduplicate_log_filename` | str | `"duplicates.csv"` | Nome do arquivo de log (salvo em `reports/` por padrão no template). |
| `encode_categoricals` | bool | `true` | Ativa a codificação de variáveis categóricas para modelagem. |
| `encoding_type` | str | `"onehot"` | Tipo de codificação: `"onehot"` (seguro) ou `"ordinal"` (compacto). |
| `scale_numeric` | bool | `false` | Ativa o escalonamento de colunas numéricas. |
| `scaler` | str | `"standard"` | Método de escala: `"standard"` (z‑score) ou `"minmax"` (0–1). |
| `date_features` | bool | `true` | Ativa a conversão e a criação de *features* de data. |
| `text_features` | bool | `false` | Ativa geração de *features* simples de texto (tamanho, contagem de palavras etc.). |
| `feature_engineering` | bool | `true` | Habilita o bloco de Engenharia de Atributos (transformações manuais por projeto). |
| `export_interim` | bool | `true` | Exporta artefato **intermediário** (pós-limpeza) para `data/interim/`. |
| `normalize_categories` | bool | `true` | Padroniza rótulos categóricos equivalentes (ex.: `"No internet service"` → `"No"`). |
| `export_processed` | bool | `true` | Exporta artefato **final tratado** para `data/processed/`. |

---

## 🧭 Como os parâmetros afetam o pipeline

- **Qualidade & Tipagem** → `strip_whitespace`, `cast_numeric_like`, `infer_types`, `normalize_categories`  
- **Faltantes** → `handle_missing`, `missing_strategy`  
- **Outliers** → `detect_outliers`, `outlier_method`  
- **Duplicidades** → `deduplicate`, `deduplicate_subset`, `deduplicate_keep`, `deduplicate_log`, `deduplicate_log_filename`  
- **Datas** → `date_features`  
- **Texto** → `text_features`  
- **Engenharia de Atributos** → `feature_engineering`  
- **Codificação & Escala** → `encode_categoricals`, `encoding_type`, `scale_numeric`, `scaler`  
- **Exportação** → `export_interim`, `export_processed`

> Os valores efetivamente aplicados em cada execução são registrados no **log** (`reports/data_preparation.log`) e no **`artifacts/manifest.json`**.

---

## 🧪 Exemplo de `local.json` (sobrepondo o padrão)

```json
{
  "detect_outliers": false,
  "scale_numeric": true,
  "scaler": "minmax",
  "encoding_type": "ordinal",
  "deduplicate_subset": ["customerID"],
  "deduplicate_keep": "last",
  "text_features": true
}
```

---

## ✅ Recomendações

1. **Mantenha o `defaults.json` estável** e use `local.json` para ajustes por projeto.  
2. **Versione** ambos os arquivos para ter histórico de mudanças.  
3. Valide mudanças no `local.json` com uma execução curta do notebook (para evitar surpresas em produção).  
4. Consulte o **`manifest.json`** para auditoria rápida do que foi aplicado em cada rodada.  