# 📔 notebooks

Contém os **Notebooks** principais do projeto, organizados por etapas:

1. `01_data_preparation.ipynb` — limpeza e padronização de dados  
2. `02_model_training.ipynb` — modelagem e avaliação  
3. `03_visual_analysis.ipynb` — geração de insights e integração com dashboards  

---

## 🧩 Funções Utilitárias de Dados

O projeto disponibiliza um conjunto de **funções auxiliares** que podem ser usadas em qualquer notebook para acelerar o tratamento e a preparação dos dados.  
Essas funções foram criadas para complementar o uso direto do **pandas** e garantir **reprodutibilidade**, **padronização** e **melhor legibilidade** do código.

Abaixo estão as principais funções e seus propósitos:

---

### 🔹 Leitura e Salvamento
| Função | Descrição |
|--------|------------|
| `load_csv(filepath, **read_kwargs)` | Carrega um arquivo CSV com parâmetros configuráveis (`sep`, `encoding`, `dtype`, etc.). Registra a operação no log. |
| `save_parquet(df, filepath)` | Salva um DataFrame em formato Parquet, criando pastas automaticamente e registrando o caminho no log. |

---

### 🔹 Diagnóstico e Otimização
| Função | Descrição |
|--------|------------|
| `basic_overview(df)` | Retorna um resumo rápido do dataset (shape, colunas, tipos, memória e nulos). |
| `reduce_memory_usage(df)` | Reduz o uso de memória convertendo colunas numéricas para tipos menores (`int32`, `float32`, etc.). |
| `infer_numeric_like(df, columns=None)` | Converte automaticamente colunas com números armazenados como texto (ex.: `'1.234,56'`) para valores numéricos. |
| `strip_whitespace(df)` | Remove espaços extras em colunas de texto. |

---

### 🔹 Tratamento de Valores Ausentes
| Função | Descrição |
|--------|------------|
| `missing_report(df)` | Gera um relatório dos valores nulos, com contagem e percentual por coluna. |
| `simple_impute(df)` | Realiza imputação simples: preenche nulos de colunas numéricas com a mediana e categóricas com a moda. |

---

### 🔹 Outliers
| Função | Descrição |
|--------|------------|
| `detect_outliers_iqr(df, cols=None)` | Detecta outliers pelo método IQR (faixa interquartílica) e adiciona colunas booleanas `*_is_outlier`. |
| `detect_outliers_zscore(df, threshold=3.0, cols=None)` | Detecta outliers com base no Z-score, comparando desvios-padrão em relação à média. |

---

### 🔹 Duplicidades e Consistência
| Função | Descrição |
|--------|------------|
| `deduplicate_rows(df)` | Remove registros duplicados e registra a quantidade removida no log. |

---

### 🔹 Codificação e Escalonamento
| Função | Descrição |
|--------|------------|
| `encode_categories(df, encoding='onehot')` | Codifica variáveis categóricas usando One-Hot ou Ordinal Encoding. Retorna o DataFrame transformado e um dicionário com metadados. |
| `scale_numeric(df, method='standard')` | Escala colunas numéricas com `StandardScaler` ou `MinMaxScaler`. Retorna o DataFrame escalado e informações do método usado. |

---

## 💡 Boas práticas de uso

- Todas as funções registram suas ações no **arquivo de log** definido em `reports/data_preparation.log`.
- Prefira aplicar as funções em blocos lógicos (ex.: limpeza → imputação → codificação).
- O notebook `01_data_preparation.ipynb` demonstra o uso prático de cada uma delas.
- Se desejar criar novas funções utilitárias, siga o mesmo padrão de:
  - Nome claro e em inglês;
  - Docstring curta explicando propósito e parâmetros;
  - Log informativo (`logger.info()`).

