# ⚙️ config

Esta pasta contém os arquivos de **configuração** utilizados pelos notebooks do projeto.  
O arquivo principal é o **`defaults.json`**, que define os parâmetros de comportamento padrão para a fase de **preparação e padronização de dados**.

Você pode criar um arquivo adicional chamado **`local.json`** para sobrescrever parâmetros específicos do `defaults.json` sem alterar o template original.

---

## 📘 Estrutura dos Arquivos

- **`defaults.json`** → Configurações padrão aplicadas a todos os projetos.  
- **`local.json`** → Arquivo opcional para ajustes locais (sobrepõe os valores do `defaults.json`).

---

## 🔧 Parâmetros Disponíveis

| Parâmetro | Tipo | Valor padrão | Descrição |
|------------|------|---------------|------------|
| **`infer_types`** | `bool` | `true` | Tenta inferir e ajustar automaticamente os tipos de dados (ex.: `int`, `float`, `datetime`). |
| **`cast_numeric_like`** | `bool` | `true` | Converte colunas numéricas armazenadas como texto (ex.: `"1.234,56"`) para valores numéricos. |
| **`strip_whitespace`** | `bool` | `true` | Remove espaços em branco extras nas colunas de texto (`object`). |
| **`handle_missing`** | `bool` | `true` | Ativa o tratamento de valores nulos ou ausentes. |
| **`missing_strategy`** | `str` | `"simple"` | Define a estratégia de imputação: `"simple"` (média/mediana/moda) ou `"advanced"` (técnicas personalizadas). |
| **`detect_outliers`** | `bool` | `true` | Ativa a detecção de outliers nas colunas numéricas. |
| **`outlier_method`** | `str` | `"iqr"` | Método para identificar outliers: `"iqr"` (intervalo interquartílico) ou `"zscore"`. |
| **`deduplicate`** | `bool` | `true` | Remove registros duplicados exatos e registra a quantidade removida. |
| **`encode_categoricals`** | `bool` | `true` | Ativa a codificação de variáveis categóricas para uso em modelos. |
| **`encoding_type`** | `str` | `"onehot"` | Define o tipo de codificação: `"onehot"` (mais seguro) ou `"ordinal"` (mais compacto). |
| **`scale_numeric`** | `bool` | `false` | Ativa a normalização ou padronização das colunas numéricas. |
| **`scaler`** | `str` | `"standard"` | Método de escalonamento: `"standard"` (Z-score) ou `"minmax"` (0–1). |
| **`date_features`** | `bool` | `true` | Ativa o processamento de colunas de data (extração de ano, mês, dia, etc.). |
| **`text_features`** | `bool` | `false` | Ativa o processamento de colunas de texto (comprimento, contagem de palavras, etc.). |
| **`feature_engineering`** | `bool` | `true` | Permite criar novas features derivadas (ex.: relações entre colunas). |
| **`export_interim`** | `bool` | `true` | Exporta o dataset intermediário após limpeza para `data/interim/`. |
| **`export_processed`** | `bool` | `true` | Exporta o dataset final tratado para `data/processed/`. |

---

## 🧠 Boas Práticas

- **Mantenha o `defaults.json` intacto**: ele define o comportamento base do template.  
- Use **`local.json`** para alterar configurações específicas de cada projeto.  
- Todos os parâmetros são carregados automaticamente no início dos notebooks via a função `load_config()`.  
- As configurações aplicadas são registradas no log (`reports/data_preparation.log`) e também no arquivo `manifest.json`.

---

### 💡 Exemplo de `local.json`

```json
{
  "detect_outliers": false,
  "scale_numeric": true,
  "scaler": "minmax",
  "encoding_type": "ordinal"
}
