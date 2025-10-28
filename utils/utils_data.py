# utils_data.py
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(NullHandler())
logger.setLevel(logging.INFO)

from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional, Iterable
import logging
from logging import NullHandler
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import re

try:
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
except Exception as e:
    raise ImportError("scikit-learn é necessário para utils_data.py: pip install scikit-learn") from e
def load_csv(filepath, **read_kwargs) -> pd.DataFrame:
    logger.info(f'Loading CSV: {filepath}')
    df = pd.read_csv(filepath, **read_kwargs)
    return df

def save_parquet(df: pd.DataFrame, filepath) -> None:
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fp, index=False)
    logger.info(f'Saved parquet to: {fp}')

def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "na_counts": df.isna().sum().to_dict()
    }
    return info

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=['int64', 'int32', 'int16']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f'Memory reduced: {start_mem:.2f}MB -> {end_mem:.2f}MB')
    return df

def infer_numeric_like(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    min_ratio: float = 0.9,
    create_new_col_when_partial: bool = True,
    blacklist: Optional[List[str]] = None,
    whitelist: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converte colunas 'parecidas com numéricas' (strings com números) respeitando um limiar de conversão.
    - min_ratio: fração mínima (0..1) dos valores não nulos que precisam ser conversíveis para aplicar na coluna original.
    - create_new_col_when_partial: se True, cria <col>_num quando ratio estiver entre (0, min_ratio).
    - blacklist/whitelist: controle fino de colunas a considerar/ignorar.
    Retorna: (df_atualizado, report_df)
    """
    import numpy as np
    report_rows: List[Dict[str, Any]] = []

    # --- definir alvo ---
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if columns is not None:
        target_cols = [c for c in columns if c in df.columns]
    elif whitelist:
        target_cols = [c for c in whitelist if c in df.columns]
    else:
        target_cols = obj_cols
    if blacklist:
        target_cols = [c for c in target_cols if c not in set(blacklist)]

    for col in target_cols:
        s = df[col].astype("string")  # mantém NA como <NA>
        s_stripped = s.str.strip()

        # tratar vazios
        non_null_mask = s_stripped.notna() & (s_stripped != "")
        base_count = int(non_null_mask.sum())
        if base_count == 0:
            report_rows.append({
                "column": col, "action": "skip_empty", "ratio": 0.0,
                "converted": 0, "non_convertible": 0, "examples": []
            })
            continue

        # detectar porcentagem (qualquer %) e marcar flag
        has_percent = s_stripped.str.contains("%", regex=False, na=False).mean() > 0.0

        # limpeza de símbolos monetários/espaços (preserva sinal e separadores)
        cleaned = (
            s_stripped
            .str.replace(r"[ \t\r\n\$€£R\$]", "", regex=True)
            .str.replace("\u00A0", "", regex=False)  # espaço não-quebrável
        )

        # heurística de separadores:
        # casos principais: "1.234,56" (pt-BR) e "1,234.56" (en-US)
        # Estratégia: remover separador de milhar e padronizar decimal como ponto.
        # 1) se contém ponto e vírgula: assuma pt-BR ('.' milhar, ',' decimal)
        has_dot = cleaned.str.contains(r"\.", na=False).mean() > 0
        has_comma = cleaned.str.contains(r",", na=False).mean() > 0

        candidate = cleaned
        if has_dot and has_comma:
            # assume pt-BR
            candidate = candidate.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        elif has_comma and not has_dot:
            # pode ser decimal com vírgula; converte vírgula para ponto
            candidate = candidate.str.replace(",", ".", regex=False)
        else:
            # só ponto ou nenhum separador → segue
            pass

        # remove o símbolo de % no fim (se houver); conversão numérica
        candidate_num = candidate.str.replace("%", "", regex=False)
        numeric = pd.to_numeric(candidate_num, errors="coerce")

        # se for porcentagem, divide por 100
        if has_percent:
            numeric = numeric / 100.0

        convertible = int(numeric[non_null_mask].notna().sum())
        ratio = convertible / base_count if base_count else 0.0
        non_conv = base_count - convertible

        # exemplos de não conversão (até 3)
        examples = s_stripped[non_null_mask & numeric.isna()].dropna().unique().tolist()[:3]

        if convertible == 0:
            action = "skip_no_conversion"
        elif ratio >= min_ratio:
            # aplica na coluna original
            df[col] = numeric
            action = "apply_overwrite"
        else:
            if create_new_col_when_partial:
                new_col = f"{col}_num"
                df[new_col] = numeric
                action = f"partial_to_{new_col}"
            else:
                action = "partial_skip"

        report_rows.append({
            "column": col,
            "action": action,
            "ratio": round(ratio, 4),
            "converted": convertible,
            "non_convertible": non_conv,
            "examples": examples
        })

    report_df = pd.DataFrame(report_rows).sort_values(["action", "column"]).reset_index(drop=True)
    logging.getLogger(__name__).info("[infer_numeric_like] summary:\n%s", report_df.to_string(index=False))
    return df, report_df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = df.isna().mean().sort_values(ascending=False).rename('missing_rate').to_frame()
    rep['missing_count'] = df.isna().sum()
    return rep

def simple_impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df

def detect_outliers_iqr(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = cols or df.select_dtypes(include=['number']).columns.tolist()
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f'{c}_is_outlier'] = (df[c] < lower) | (df[c] > upper)
    return df

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = cols or df.select_dtypes(include=['number']).columns.tolist()
    for c in cols:
        mu, sigma = df[c].mean(), df[c].std(ddof=0)
        if sigma == 0:
            df[f'{c}_is_outlier'] = False
        else:
            z = (df[c] - mu) / sigma
            df[f'{c}_is_outlier'] = z.abs() > threshold
    return df


def deduplicate_rows(
    df: pd.DataFrame,
    subset: Optional[Iterable[str]] = None,
    keep: "str|bool" = "first",
    log_path: Optional[Path] = None,
    return_report: bool = False
) -> "pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """
    Remove duplicatas e (opcionalmente) registra as linhas duplicadas.
    - subset: colunas que definem duplicidade (None = todas as colunas)
    - keep: 'first' | 'last' | False (remove todas as repetições)
    - log_path: se informado e houver duplicatas, salva CSV com duplicatas detectadas
    - return_report: se True, retorna (df_sem_dup, duplicatas_df, resumo_df)

    resumo_df inclui contagens por chave (se subset for fornecido) ou por linha inteira (hash).
    """
    before = len(df)

    # máscara de duplicados (todas as ocorrências)
    dup_mask = df.duplicated(subset=subset, keep=False)
    dups_df = df.loc[dup_mask].copy()

    # resumo por chave
    if subset:
        grp = dups_df.groupby(list(subset), dropna=False, as_index=False).size().rename(columns={"size": "count"})
        # manter apenas grupos com count >= 2
        summary_df = grp[grp["count"] >= 2].sort_values("count", ascending=False)
    else:
        # linha inteira como chave (gera um hash estável por linha)
        key = pd.util.hash_pandas_object(dups_df, index=False)
        summary_df = (pd.DataFrame({"_row_hash": key})
                        .value_counts()
                        .reset_index(name="count")
                        .query("count >= 2")
                        .sort_values("count", ascending=False))

    # log em arquivo, se solicitado
    if log_path is not None and not dups_df.empty:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        dups_df.to_csv(log_path, index=False, encoding="utf-8")
        logger.info(f"[deduplicate] Duplicatas salvas em: {log_path} (linhas={len(dups_df)})")

    # remoção de duplicatas segundo a política 'keep'
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df_clean)
    logger.info(f"[deduplicate] Removed duplicates: {removed} (subset={subset}, keep={keep})")

    if return_report:
        return df_clean, dups_df, summary_df
    return df_clean


def _onehot_encoder_compat():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def encode_categories(df: pd.DataFrame, encoding: str = 'onehot') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    meta: Dict[str, Any] = {"categorical_columns": cat_cols, "encoding": encoding}
    if not cat_cols:
        return df, meta

    if encoding == 'onehot':
        encoder = _onehot_encoder_compat()
        arr = encoder.fit_transform(df[cat_cols])
        encoded = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), encoded], axis=1)
        meta['categories_'] = {c: list(map(str, cats)) for c, cats in zip(cat_cols, encoder.categories_)}
    elif encoding == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        meta['categories_'] = {c: list(map(str, cats)) for c, cats in zip(cat_cols, encoder.categories_)}
    else:
        raise ValueError("Unsupported encoding type.")
    return df, meta

def scale_numeric(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    meta: Dict[str, Any] = {"numeric_columns": num_cols, "scaler": method}
    if not num_cols:
        return df, meta

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('Unsupported scaler.')

    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, meta

__all__ = [
    'apply_encoding_and_scaling',
    'basic_overview',
    'build_calendar_from',
    'deduplicate_rows',
    'detect_date_candidates',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'encode_categories',
    'encode_categories_safe',
    'expand_date_features',
    'extract_text_features',
    'infer_format_from_suffix',
    'infer_numeric_like',
    'list_directory_files',
    'load_artifact',
    'load_csv',
    'load_manifest',
    'load_table_simple',
    'merge_chain',
    'missing_report',
    'parse_dates_with_report',
    'reduce_memory_usage',
    'save_artifact',
    'save_manifest',
    'save_named_interims',
    'save_parquet',
    'save_table',
    'scale_numeric',
    'scale_numeric_safe',
    'simple_impute',
    'simple_impute_with_flags',
    'strip_whitespace',
    'summarize_text_features',
    'update_manifest',
]


# ==============================================
# 📥 Ingestão flexível 
# ==============================================
def infer_format_from_suffix(path) -> str:
    """Deduz o formato pelo sufixo do arquivo. Suporta .csv e .parquet."""
    ext = str(path).lower().rsplit(".", 1)[-1]
    if ext in ("csv", "parquet"):
        return ext
    raise ValueError(f"Formato não suportado (use .csv ou .parquet): {path}")

def load_table_simple(path, fmt: Optional[str] = None, read_opts: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Leitura simples de tabela:
    - CSV: usa load_csv (para manter logging e opções padronizadas)
    - Parquet: usa pandas.read_parquet
    """
    fmt = (fmt or infer_format_from_suffix(path)).lower()
    read_opts = read_opts or {}
    logger.info(f"Loading table: path={path} | format={fmt} | opts={read_opts}")
    if fmt == "csv":
        return load_csv(path, **read_opts)
    elif fmt == "parquet":
        return pd.read_parquet(path, **read_opts)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")

def merge_chain(base_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], steps) -> pd.DataFrame:
    """
    Aplica merges em cadeia.
    steps: lista de tuplas (right_name, how, left_on, right_on).
    Faz checagem simples de chave não-única na tabela 'right' e loga avisos.
    """
    df = base_df.copy()

    def _is_unique(d: pd.DataFrame, key) -> Tuple[bool, int]:
        key = [key] if isinstance(key, str) else list(key)
        dups = d.duplicated(subset=key, keep=False).sum()
        return (dups == 0, int(dups))

    for (right_name, how, left_on, right_on) in steps:
        if right_name not in tables:
            raise KeyError(f"Tabela '{right_name}' não encontrada nas fontes.")
        right_df = tables[right_name]

        uniq_r, dups_r = _is_unique(right_df, right_on)
        if not uniq_r and how in {"left", "right"}:
            logger.warning(f"[merge] '{right_name}': chave {right_on} não é única (dups={dups_r}).")

        logger.info(f"[merge] how={how} | left_on={left_on} | right_on={right_on} | with={right_name}")
        df = df.merge(right_df, how=how, left_on=left_on, right_on=right_on,
                      suffixes=("", f"__{right_name}"))
        logger.info(f"[merge] shape atual: {df.shape}")

    return df

# Exporte também no __all__ se você o utiliza:
try:
    __all__.extend(["infer_format_from_suffix", "load_table_simple", "merge_chain"])
except NameError:
    __all__ = ["infer_format_from_suffix", "load_table_simple", "merge_chain"]
# -----------------------------------------------------------------------------

# ==============================================
# 📥 catálogo de DataFrames e salvamento
# ==============================================

@dataclass
class TableInfo:
    name: str
    rows: int
    cols: int
    memory_mb: float

class TableStore:
    """
    Catálogo de DataFrames nomeados com um 'current' (ativo).
    Uso:
      T = TableStore(initial={"main": df_main}, current="main")
      df = T.get()               # df atual
      T.add("features_v1", df2)  # adiciona
      df = T.use("features_v1")  # troca o atual e retorna o df
      T.list()                   # inventário
      T["features_v1"]           # acesso estilo dict
    """
    def __init__(self, initial: dict[str, pd.DataFrame] | None = None, current: str | None = None):
        self._store: dict[str, pd.DataFrame] = dict(initial or {})
        self.current: str | None = current if current in (initial or {}) else (next(iter(self._store)) if self._store else None)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self._store[name]

    def add(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("objeto precisa ser um pandas.DataFrame")
        self._store[name] = df
        logging.getLogger(__name__).info(f"[tables] add '{name}' shape={df.shape}")
        if set_current:
            self.current = name

    def get(self, name: str | None = None) -> pd.DataFrame:
        name = name or self.current
        if name is None:
            raise KeyError("Nenhuma tabela atual definida.")
        return self._store[name]

    def rename(self, old: str, new: str) -> None:
        if new in self._store:
            raise KeyError(f"Já existe tabela com nome '{new}'.")
        self._store[new] = self._store.pop(old)
        if self.current == old:
            self.current = new
        logging.getLogger(__name__).info(f"[tables] rename '{old}' → '{new}'")

    def drop(self, name: str) -> None:
        self._store.pop(name)
        logging.getLogger(__name__).info(f"[tables] drop '{name}'")
        if self.current == name:
            self.current = next(iter(self._store)) if self._store else None

    def list(self) -> pd.DataFrame:
        infos = []
        for n, d in self._store.items():
            mem = float(d.memory_usage(deep=True).sum() / (1024**2))
            infos.append(TableInfo(n, len(d), d.shape[1], mem))
        return pd.DataFrame([i.__dict__ for i in infos]).sort_values("name")

    def use(self, name: str) -> pd.DataFrame:
        if name not in self._store:
            raise KeyError(f"Tabela '{name}' não existe. Disponíveis: {list(self._store)}")
        self.current = name
        df = self._store[name]
        print(f"[using] current='{name}' shape={df.shape}")
        return df

def save_table(df: pd.DataFrame, path: Path) -> None:
    """Salva DataFrame respeitando a extensão do path (.csv ou .parquet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Extensão não suportada: {ext}")
    logging.getLogger(__name__).info(f"Saved: {path}")

def save_named_interims(named_frames: dict[str, pd.DataFrame], base_dir: Path, fmt: str = "parquet") -> None:
    """
    Salva vários DataFrames nomeados em data/interim com convenção <nome>_interim.<fmt>.
    Ex.: save_named_interims({"main": df, "dim": dim_df}, INTERIM_DIR)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, dfx in named_frames.items():
        out = base_dir / f"{name}_interim.{fmt}"
        if fmt == "csv":
            dfx.to_csv(out, index=False)
        elif fmt == "parquet":
            dfx.to_parquet(out, index=False)
        else:
            raise ValueError("fmt deve ser 'csv' ou 'parquet'")
        logging.getLogger(__name__).info(f"[interim] saved: {out} shape={dfx.shape}")

# Exporte no __all__
try:
    __all__.extend(["TableStore", "save_table", "save_named_interims"])
except NameError:
    __all__ = ["TableStore", "save_table", "save_named_interims"]
# -----------------------------------------------------------------------------


def simple_impute_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes de forma simples (numéricos → mediana, categóricos → moda)
    e cria colunas booleanas 'was_imputed_<col>' para indicar as linhas alteradas.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

    for c in num_cols:
        if df[c].isna().any():
            imputed_mask = df[c].isna()  # marca posições faltantes
            df[c] = df[c].fillna(df[c].median())
            df[f"was_imputed_{c}"] = imputed_mask
            logging.getLogger(__name__).info(f"[impute] coluna '{c}' → {imputed_mask.sum()} valores preenchidos (mediana).")

    for c in cat_cols:
        if df[c].isna().any():
            imputed_mask = df[c].isna()
            df[c] = df[c].fillna(df[c].mode().iloc[0])
            df[f"was_imputed_{c}"] = imputed_mask
            logging.getLogger(__name__).info(f"[impute] coluna '{c}' → {imputed_mask.sum()} valores preenchidos (moda).")

    return df


# ==============================================
# 📅 Tratamento com Datas
# ==============================================

def detect_date_candidates(df: pd.DataFrame, pattern: str) -> List[str]:
    """Retorna nomes de colunas candidatas a datas via regex + dtype datetime já existente."""
    rx = re.compile(pattern, re.IGNORECASE)
    return [
        c for c in df.columns
        if rx.search(c) or pd.api.types.is_datetime64_any_dtype(df[c])
    ]

def _maybe_to_datetime(s: pd.Series, *, dayfirst: bool, utc: bool, formats: List[str]) -> pd.Series:
    """Tenta converter com formatos explícitos; se falhar, usa fallback genérico."""
    for fmt in formats or []:
        try:
            out = pd.to_datetime(s, format=fmt, errors="coerce", dayfirst=dayfirst, utc=utc)
            if out.notna().mean() > 0:
                return out
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=utc)


def parse_dates_with_report(
    df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Converte colunas de data e retorna:
      - df (com colunas convertidas)
      - parse_report (coluna, parsed_ratio, converted)
      - parsed_cols (lista de colunas aceitas)
    """
    detect_regex = cfg.get("detect_regex", r"(date|data|dt_|_dt$|_date$)")
    dayfirst     = bool(cfg.get("dayfirst", False))
    utc          = bool(cfg.get("utc", False))
    formats      = cfg.get("formats", []) or []
    min_ratio    = float(cfg.get("min_ratio", 0.8))

    explicit = cfg.get("explicit_cols", []) or []
    candidates = list(dict.fromkeys(
        detect_date_candidates(df, detect_regex) + [c for c in explicit if c in df.columns]
    ))

    report_rows: List[Dict[str, Any]] = []
    parsed_cols: List[str] = []

    for c in candidates:
        s_raw = df[c]
        s_dt = s_raw if pd.api.types.is_datetime64_any_dtype(s_raw) else _maybe_to_datetime(
            s_raw, dayfirst=dayfirst, utc=utc, formats=formats
        )
        ratio = float(s_dt.notna().mean())
        converted = ratio >= min_ratio
        report_rows.append({"column": c, "parsed_ratio": ratio, "converted": converted})
        if converted:
            df[c] = s_dt
            parsed_cols.append(c)

    # --- relatório defensivo ---
    cols_order = ["column", "parsed_ratio", "converted"]
    if report_rows:
        parse_report = pd.DataFrame(report_rows)
        for col in cols_order:
            if col not in parse_report.columns:
                parse_report[col] = (False if col == "converted" else np.nan)
        parse_report = parse_report[cols_order].sort_values(
            ["converted", "parsed_ratio"], ascending=[False, False]
        )
    else:
        parse_report = pd.DataFrame(columns=cols_order)

    logger.info(f"[dates] candidates={candidates}")
    logger.info(f"[dates] parsed_ok={parsed_cols}")
    return df, parse_report, parsed_cols


def expand_date_features(
    df: pd.DataFrame,
    cols: List[str],
    *,
    features: List[str] = None,
    prefix_mode: str = "auto",
    fixed_prefix: str = None
) -> List[str]:
    """Cria features temporais para cada coluna datetime em cols. Retorna lista de colunas criadas."""
    features = features or ["year","month","day","dayofweek","quarter","week","is_month_start","is_month_end"]
    created = []
    for col in cols:
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            continue
        pfx = fixed_prefix if (prefix_mode == "fixed" and fixed_prefix) else col
        if "year" in features:
            df[f"{pfx}_year"] = s.dt.year; created.append(f"{pfx}_year")
        if "month" in features:
            df[f"{pfx}_month"] = s.dt.month; created.append(f"{pfx}_month")
        if "day" in features:
            df[f"{pfx}_day"] = s.dt.day; created.append(f"{pfx}_day")
        if "dayofweek" in features:
            df[f"{pfx}_dow"] = s.dt.dayofweek; created.append(f"{pfx}_dow")
        if "quarter" in features:
            df[f"{pfx}_quarter"] = s.dt.quarter; created.append(f"{pfx}_quarter")
        if "week" in features:
            df[f"{pfx}_week"] = s.dt.isocalendar().week.astype("Int64"); created.append(f"{pfx}_week")
        if "is_month_start" in features:
            df[f"{pfx}_is_month_start"] = s.dt.is_month_start; created.append(f"{pfx}_is_month_start")
        if "is_month_end" in features:
            df[f"{pfx}_is_month_end"] = s.dt.is_month_end; created.append(f"{pfx}_is_month_end")
    logger.info(f"[dates] created_features={len(created)}")
    return created

def build_calendar_from(df: pd.DataFrame, date_col: str, freq: str = "D") -> pd.DataFrame:
    """Gera uma tabela calendário (dim_date) a partir do range observado em df[date_col]."""
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Coluna inválida para calendário: {date_col}")
    start = df[date_col].min()
    end = df[date_col].max()
    idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq=freq)
    cal = pd.DataFrame({"date": idx})
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["day"] = cal["date"].dt.day
    cal["quarter"] = cal["date"].dt.quarter
    cal["week"] = cal["date"].dt.isocalendar().week.astype("Int64")
    cal["dow"] = cal["date"].dt.dayofweek
    cal["is_month_start"] = cal["date"].dt.is_month_start
    cal["is_month_end"] = cal["date"].dt.is_month_end
    cal["month_name"] = cal["date"].dt.month_name()
    cal["day_name"] = cal["date"].dt.day_name()
    return cal


# ==============================================
# 📝 Tratamento de Texto / Extração de Features
# ==============================================

def extract_text_features(
    df: pd.DataFrame,
    *,
    lower: bool = True,
    strip_collapse_ws: bool = True,
    keywords: Optional[List[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
    export_summary: bool = False,
    summary_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai métricas numéricas de colunas textuais.

    - Cria contadores de caracteres, palavras, letras e dígitos.
    - Detecta presença de palavras-chave.
    - Pode exportar um resumo CSV com colunas geradas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    lower : bool
        Se True, converte textos para minúsculas.
    strip_collapse_ws : bool
        Remove espaços duplicados e trim nos extremos.
    keywords : list[str], opcional
        Lista de palavras-chave para busca booleana (ex.: ["error", "cancel", "premium"])
    blacklist : list[str], opcional
        Lista de colunas a ignorar (ex.: ["customerID"])
    export_summary : bool
        Se True, salva CSV resumo das features geradas.
    summary_dir : Path, opcional
        Caminho para salvar o resumo (usado com export_summary=True).

    Retorna:
    --------
    (df, summary_df)
        df atualizado com novas colunas e resumo das colunas processadas.
    """
    df = df.copy()

    keywords = keywords or []
    blacklist = set(map(str, blacklist or []))
    text_cols = [c for c in df.columns if df[c].dtype == "object" and c not in blacklist]

    created_cols = []
    processed_cols = []

    for c in text_cols:
        s = df[c].astype("string").fillna("")
        if strip_collapse_ws:
            s = s.str.strip().str.replace(r"\s+", " ", regex=True)
        if lower:
            s = s.str.lower()

        df[f"{c}_len"] = s.str.len()
        df[f"{c}_word_count"] = s.str.split().map(len)
        df[f"{c}_alpha_count"] = s.str.count(r"[A-Za-z]")
        df[f"{c}_digit_count"] = s.str.count(r"\d")

        created_cols += [
            f"{c}_len", f"{c}_word_count", f"{c}_alpha_count", f"{c}_digit_count"
        ]

        for kw in keywords:
            safe_kw = str(kw).strip().replace(" ", "_")
            col_name = f"{c}_has_{safe_kw}"
            df[col_name] = s.str.contains(str(kw), case=False, na=False)
            created_cols.append(col_name)

        processed_cols.append(c)

    logger.info(f"[text] colunas processadas: {processed_cols}")
    logger.info(f"[text] features criadas: {len(created_cols)}")

    summary_df = pd.DataFrame([
        {
            "text_col": c,
            "keywords_cols": ", ".join([
                f"{c}_has_{str(kw).strip().replace(' ', '_')}" for kw in keywords
            ]),
        }
        for c in processed_cols
    ])

    if export_summary and summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        logger.info(f"[text] resumo salvo em: {summary_path}")

    return df, summary_df


# ================================
# 🔤 Categóricas & 🔢 Numéricas (safe)
# ================================
from typing import Set

def _warn_high_cardinality(df: pd.DataFrame, cols: list[str], threshold: int = 50) -> dict[str, int]:
    high = {c: int(df[c].nunique(dropna=False)) for c in cols if df[c].nunique(dropna=False) > threshold}
    if high:
        logger.warning(f"[encode] Alta cardinalidade (> {threshold}): {high}")
    return high

def encode_categories_safe(
    df: pd.DataFrame,
    *,
    method: str = "onehot",              # "onehot" | "ordinal"
    exclude_cols: list[str] | None = None,
    high_card_threshold: int = 50
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Codifica colunas categóricas com exclusões e aviso de cardinalidade.
    Retorna (df, meta).
    """
    exclude: Set[str] = set(exclude_cols or [])
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]

    # alerta de cardinalidade
    high = _warn_high_cardinality(df, cat_cols, threshold=high_card_threshold)

    # separa parte preservada (excluída) da parte a transformar
    keep_df = df[list(exclude & set(df.columns))].copy()
    work_df = df[[c for c in df.columns if c not in exclude]].copy()

    work_df, meta = encode_categories(work_df, encoding=method)  # usa seu utilitário já existente
    meta.update({
        "excluded": sorted(list(exclude)),
        "high_cardinality": high,
        "encoding": method,
    })

    # recompõe
    out = pd.concat([keep_df.reset_index(drop=True), work_df.reset_index(drop=True)], axis=1)
    return out, meta


def _is_dummy_or_boolean(s: pd.Series) -> bool:
    # considera dummy se o conjunto de valores não nulos ⊆ {0,1}
    vals = set(pd.unique(s.dropna()))
    return s.dtype == "bool" or vals.issubset({0, 1})

def scale_numeric_safe(
    df: pd.DataFrame,
    *,
    method: str = "standard",           # "standard" | "minmax"
    exclude_cols: list[str] | None = None,
    only_continuous: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Escalona colunas numéricas com exclusões e opção de ignorar dummies/booleanas.
    Retorna (df, meta).
    """
    exclude: Set[str] = set(exclude_cols or [])
    keep_df = df[list(exclude & set(df.columns))].copy()
    work_df = df[[c for c in df.columns if c not in exclude]].copy()

    # decide quais numéricas escalar
    all_num = work_df.select_dtypes(include=["number", "boolean"]).columns.tolist()
    if only_continuous:
        target_num = [c for c in all_num if not _is_dummy_or_boolean(work_df[c])]
    else:
        target_num = all_num

    if not target_num:
        logger.info("[scale] nenhuma coluna contínua elegível para escalonamento.")
        return df, {"numeric_columns": [], "scaler": method, "excluded": sorted(list(exclude))}

    # o scale_numeric original escala todas numéricas; vamos fazer um wrapper:
    # para manter compatibilidade simples, chamamos e depois restauramos as não-alvo.
    pre_vals = {c: work_df[c] for c in work_df.columns if c not in target_num}
    work_df_target = work_df[target_num].copy()

    # usa seu utilitário; ele escala todas numéricas do DF passado
    scaled_block, meta = scale_numeric(work_df_target, method=method)
    work_df[target_num] = scaled_block[target_num]

    # recompõe tudo (não precisava salvar/restaurar porque trabalhamos por seleção)
    out = pd.concat([keep_df.reset_index(drop=True), work_df.reset_index(drop=True)], axis=1)

    meta.update({
        "numeric_columns": target_num,
        "scaler": method,
        "excluded": sorted(list(exclude)),
        "only_continuous": only_continuous,
    })
    return out, meta


def apply_encoding_and_scaling(
    df: pd.DataFrame,
    *,
    encode_cfg: dict | None = None,
    scale_cfg: dict | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Aplica encode → scale de forma orquestrada com configs simples.
    encode_cfg: {enabled, type, exclude_cols, high_card_threshold}
    scale_cfg:  {enabled, method, exclude_cols, only_continuous}
    """
    encode_cfg = encode_cfg or {}
    scale_cfg  = scale_cfg  or {}
    enc_meta, scl_meta = {}, {}

    if encode_cfg.get("enabled", True):
        df, enc_meta = encode_categories_safe(
            df,
            method=encode_cfg.get("type", "onehot"),
            exclude_cols=encode_cfg.get("exclude_cols", []),
            high_card_threshold=int(encode_cfg.get("high_card_threshold", 50)),
        )
        logger.info(f"[encode] type={enc_meta.get('encoding')} | cols={enc_meta.get('categorical_columns', [])}")

    if scale_cfg.get("enabled", False):
        df, scl_meta = scale_numeric_safe(
            df,
            method=scale_cfg.get("method", "standard"),
            exclude_cols=scale_cfg.get("exclude_cols", []),
            only_continuous=bool(scale_cfg.get("only_continuous", True)),
        )
        logger.info(f"[scale] method={scl_meta.get('scaler')} | cols={scl_meta.get('numeric_columns', [])}")

    return df, enc_meta, scl_meta



try:
    __all__.extend([
        "encode_categories_safe", "scale_numeric_safe", "apply_encoding_and_scaling", "extract_text_features"
    ])
except NameError:
    __all__ = ["encode_categories_safe", "scale_numeric_safe", "apply_encoding_and_scaling"]


# ==============================================
# 📂 Utilitário: listar arquivos em um diretório
# ==============================================


def list_directory_files(dir_path: Path, pattern: str = "*", sort_by: str = "name") -> pd.DataFrame:
    """
    Lista arquivos em um diretório, mostrando nome, extensão, tamanho e data de modificação.
    
    Args:
        dir_path (Path): Caminho do diretório.
        pattern (str): Padrão glob (ex: "*.csv", "*.parquet"). Padrão "*" lista todos.
        sort_by (str): "name" ou "date" para ordenação.
    
    Returns:
        pd.DataFrame com as colunas: Arquivo, Extensão, Tamanho (KB), Modificado em.
    """
    if not dir_path.exists():
        print(f"⚠️ Diretório não encontrado: {dir_path}")
        return pd.DataFrame(columns=["Arquivo", "Extensão", "Tamanho (KB)", "Modificado em"])

    files = list(dir_path.glob(pattern))
    if not files:
        print(f"⚠️ Nenhum arquivo encontrado em: {dir_path}")
        return pd.DataFrame(columns=["Arquivo", "Extensão", "Tamanho (KB)", "Modificado em"])

    rows = []
    for f in files:
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({
                "Arquivo": f.name,
                "Extensão": f.suffix.lower(),
                "Tamanho (KB)": f"{size_kb:,.1f}",
                "Modificado em": mod_time
            })

    df_files = pd.DataFrame(rows)
    if sort_by == "date":
        df_files = df_files.sort_values("Modificado em", ascending=False)
    else:
        df_files = df_files.sort_values("Arquivo")

    return df_files.reset_index(drop=True)

__all__.extend(["list_directory_files"])

import json
from pathlib import Path
from datetime import datetime
import joblib

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

def _manifest_path() -> Path:
    return ARTIFACTS_DIR / "manifest.json"

def load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Falha ao ler manifest.json: {e}")
    return {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds")},
        "preprocessing": {},
        "reports": [],
        "artifacts": []
    }

def save_manifest(manifest: dict) -> None:
    try:
        _manifest_path().write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Falha ao salvar manifest.json: {e}")

def update_manifest(section: str, payload) -> None:
    m = load_manifest()
    if isinstance(payload, dict):
        base = m.get(section, {})
        if not isinstance(base, dict):
            base = {}
        base.update(payload)
        m[section] = base
    elif isinstance(payload, list):
        base = m.get(section, [])
        if not isinstance(base, list):
            base = []
        base.extend(payload)
        m[section] = base
    else:
        m[section] = payload
    save_manifest(m)

def save_artifact(obj, filename: str) -> Path:
    path = ARTIFACTS_DIR / filename
    try:
        joblib.dump(obj, path)
        logger.info(f"Artefato salvo: {path}")
        update_manifest("artifacts", [str(path)])
    except Exception as e:
        logger.warning(f"Falha ao salvar artefato {filename}: {e}")
    return path

def load_artifact(filename: str):
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

def encode_categories_safe(
    df: pd.DataFrame,
    categorical_cols: list[str],
    strategy: str = "onehot",
    handle_unknown: str = "ignore",
    encoder=None,
    persist: bool = False,
    artifact_name: str = "encoder.joblib",
) -> tuple[pd.DataFrame, object, dict]:
    if not categorical_cols:
        logger.info("encode_categories_safe: nenhuma coluna categórica informada.")
        return df, encoder, {"categories_": {}, "strategy": strategy, "used_cols": []}

    X = df.copy()
    info = {"strategy": strategy, "used_cols": categorical_cols}

    if strategy == "onehot":
        if encoder is None:
            encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
            enc_data = encoder.fit_transform(X[categorical_cols])
        else:
            enc_data = encoder.transform(X[categorical_cols])

        enc_df = pd.DataFrame(enc_data, index=X.index, columns=encoder.get_feature_names_out(categorical_cols))
        X = pd.concat([X.drop(columns=categorical_cols), enc_df], axis=1)
        info["categories_"] = {c: list(cat) for c, cat in zip(categorical_cols, encoder.categories_)}

    elif strategy == "ordinal":
        if encoder is None:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc_data = encoder.fit_transform(X[categorical_cols])
        else:
            enc_data = encoder.transform(X[categorical_cols])
        X[categorical_cols] = enc_data
        info["categories_"] = {c: list(cat) for c, cat in zip(categorical_cols, encoder.categories_)}

    else:
        logger.warning(f"Estratégia de encoding desconhecida: {strategy}")
        return df, None, {"error": "encoding_strategy_unsupported"}

    if persist:
        save_artifact(encoder, artifact_name)
        update_manifest("preprocessing", {
            "encoding": {
                "strategy": strategy,
                "categorical_cols": categorical_cols,
                "artifact": artifact_name,
                "handle_unknown": handle_unknown
            }
        })

    return X, encoder, info


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scale_numeric_safe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    scaler=None,
    strategy: str = "standard",
    persist: bool = False,
    artifact_name: str = "scaler.joblib",
) -> tuple[pd.DataFrame, object, dict]:
    if not numeric_cols:
        logger.info("scale_numeric_safe: nenhuma coluna numérica informada.")
        return df, scaler, {"used_cols": []}

    X = df.copy()
    info = {"used_cols": numeric_cols, "strategy": strategy}

    if scaler is None:
        if strategy == "standard":
            scaler = StandardScaler()
        elif strategy == "minmax":
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Estratégia de scaling desconhecida: {strategy}")
            return df, None, {"error": "scaling_strategy_unsupported"}
        arr = scaler.fit_transform(X[numeric_cols])
    else:
        arr = scaler.transform(X[numeric_cols])

    X[numeric_cols] = arr

    if persist:
        save_artifact(scaler, artifact_name)
        update_manifest("preprocessing", {
            "scaling": {
                "strategy": strategy,
                "numeric_cols": numeric_cols,
                "artifact": artifact_name
            }
        })

    return X, scaler, info


import pandas as pd
import numpy as np

def infer_numeric_like(series: pd.Series, name: str = None) -> tuple[pd.Series, pd.DataFrame]:
    col = name or series.name or "<unnamed>"
    before_dtype = str(series.dtype)

    s = series.astype(str).str.strip()
    cleaned = (
        s.str.replace(r"[R$%\s]", "", regex=True)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
    )

    converted = pd.to_numeric(cleaned, errors="coerce")
    n_converted = converted.notna().sum()
    n_total = len(converted)

    included = n_converted > max(2, 0.6 * n_total)
    reason = "suficiente conversão" if included else "baixa conversão"

    out = converted if included else series

    report = pd.DataFrame([{
        "column": col,
        "included": bool(included),
        "reason": reason,
        "before_dtype": before_dtype,
        "after_dtype": str(out.dtype),
        "n_nulls": int(pd.isna(out).sum() if hasattr(out, "isna") else 0),
        "converted_ratio": float(n_converted / max(1, n_total))
    }])

    # Append to CSV
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    csv_path = REPORTS_DIR / "parse_report.csv"
    if csv_path.exists():
        report.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8")
    else:
        report.to_csv(csv_path, index=False, encoding="utf-8")

    update_manifest("reports", [str(csv_path)])
    return out, report


import pandas as pd

def deduplicate_rows(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    before = len(df)
    if subset is None:
        logger.info("deduplicate_rows: subset=None → desduplicando linha inteira")
    out = df.drop_duplicates(subset=subset, keep=keep)
    after = len(out)
    logger.info(f"deduplicate_rows: removidas {before - after} duplicatas (de {before})")
    return out


import pandas as pd

def summarize_text_features(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in text_cols:
        len_col = f"{c}_len"
        wc_col  = f"{c}_word_count"
        if len_col in df.columns and wc_col in df.columns:
            rows.append({
                "column": c,
                "len_mean": float(df[len_col].mean()),
                "len_median": float(df[len_col].median()),
                "len_std": float(df[len_col].std() if hasattr(df[len_col], 'std') else 0.0),
                "wc_mean": float(df[wc_col].mean()),
                "wc_median": float(df[wc_col].median()),
                "wc_std": float(df[wc_col].std() if hasattr(df[wc_col], 'std') else 0.0),
                "null_ratio": float(df[c].isna().mean() if c in df.columns else 0.0)
            })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        path = REPORTS_DIR / "text_features_summary.csv"
        summary.to_csv(path, index=False, encoding="utf-8")
        update_manifest("reports", [str(path)])
        logger.info(f"text_features_summary salvo em {path}")
    return summary
