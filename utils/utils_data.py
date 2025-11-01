# -*- coding: utf-8 -*-
"""Utilitários do projeto (I/O, pré-processamento, export, manifest, relatórios)."""

from __future__ import annotations

# ============================================================================
# Imports básicos
# ============================================================================
import json
import logging
from logging import NullHandler
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# scikit-learn (com compat)
try:
    from sklearn.preprocessing import (
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
        MinMaxScaler,
    )
except Exception as e:
    raise ImportError(\"scikit-learn é necessário: pip install scikit-learn\") from e

# joblib para artefatos
import joblib

# ============================================================================
# Logger de módulo
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(NullHandler())
logger.setLevel(logging.INFO)

# ============================================================================
# Pastas padrão (relativas ao CWD do processo; sobrescritas em runtime via paths)
# ============================================================================
ARTIFACTS_DIR = Path(\"artifacts\")
REPORTS_DIR = Path(\"reports\")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Manifest: helpers básicos
# ============================================================================
def _manifest_path() -> Path:
    return ARTIFACTS_DIR / \"manifest.json\"

def load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding=\"utf-8\"))
        except Exception as e:
            logger.warning(f\"[manifest] falha ao ler: {e}\")
    return {
        \"run\": {\"started_at\": datetime.now().isoformat(timespec=\"seconds\")},
        \"preprocessing\": {},
        \"reports\": [],
        \"artifacts\": [],
        \"run_steps\": [],
    }

def save_manifest(manifest: dict) -> None:
    try:
        _manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding=\"utf-8\"
        )
    except Exception as e:
        logger.warning(f\"[manifest] falha ao salvar: {e}\")

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
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f\"[artifact] salvo: {path}\")
    update_manifest(\"artifacts\", [str(path)])
    return path

def load_artifact(filename: str):
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)

# ============================================================================
# Reporting + Decorators (manifest-aware)
# ============================================================================
def _cfg_reporting_enabled(config: dict | None) -> bool:
    if not config:
        return True
    return bool(config.get(\"reporting\", {}).get(\"manifest_enabled\", True))

def _log_manifest_list(section: str, items: Iterable[str], *, config: dict | None = None):
    if not _cfg_reporting_enabled(config):
        return
    try:
        update_manifest(section, list(map(str, items)))
    except Exception as e:
        logger.warning(f\"[manifest] falha ao atualizar '{section}': {e}\")

def log_report_path(path: Path | str, *, config: dict | None = None):
    _log_manifest_list(\"reports\", [str(path)], config=config)

def log_artifact_path(path: Path | str, *, config: dict | None = None):
    _log_manifest_list(\"artifacts\", [str(path)], config=config)

def record_step(step_name: str, *, payload: dict | None = None, config: dict | None = None):
    if not _cfg_reporting_enabled(config):
        return
    payload = {\"name\": step_name, \"ts\": datetime.now().isoformat(timespec=\"seconds\"), **(payload or {})}
    try:
        m = load_manifest()
        steps = m.get(\"run_steps\", [])
        steps.append(payload)
        m[\"run_steps\"] = steps
        save_manifest(m)
    except Exception as e:
        logger.warning(f\"[manifest] falha ao registrar step '{step_name}': {e}\")

def with_step(step_name: str) -> Callable:
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            config = kwargs.get(\"config\")
            record_step(f\"{step_name}:start\", config=config)
            try:
                result = fn(*args, **kwargs)
                record_step(f\"{step_name}:end\", config=config)
                return result
            except Exception as e:
                record_step(f\"{step_name}:error\", payload={\"error\": str(e)}, config=config)
                raise
        return wrapper
    return deco

def save_report_df(df: pd.DataFrame, path: Path, *, config: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=\"utf-8\")
    log_report_path(path, config=config)
    logger.info(f\"[report] salvo em: {path}\")
    return path

def save_text(text: str, path: Path, *, config: dict | None = None, section: str = \"reports\") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=\"utf-8\")
    _log_manifest_list(section, [str(path)], config=config)
    logger.info(f\"[{section}] salvo em: {path}\")
    return path

# ============================================================================
# Bootstrap helpers (N1)
# ============================================================================
def ensure_project_root(config_rel: str = \"config/defaults.json\", start: Path | None = None) -> Path:
    \"\"\"Sobe diretórios até encontrar o arquivo de config_rel; retorna o caminho absoluto dele.\"\"\"
    start = start or Path.cwd()
    rel = Path(config_rel)
    for base in (start, *start.parents):
        cand = base / rel
        if cand.exists():
            return cand
    raise FileNotFoundError(f\"Não encontrei '{config_rel}' em {start}…subindo pastas.\")

def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def set_display(max_cols: int = 200, width: int = 120) -> None:
    pd.set_option(\"display.max_columns\", max_cols)
    pd.set_option(\"display.width\", width)

@dataclass
class N1Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    prints_dir: Path
    dashboards_dir: Path
    output_interim: Path
    output_processed: Path

def resolve_n1_paths(config: dict, project_root: Path) -> N1Paths:
    data_dir = project_root / \"data\"
    raw_dir = data_dir / \"raw\"
    interim_dir = data_dir / \"interim\"
    processed_dir = data_dir / \"processed\"
    reports_dir = project_root / \"reports\"
    artifacts_dir = project_root / \"artifacts\"
    prints_dir = project_root / \"prints\"
    dashboards_dir = project_root / \"dashboards\"

    for d in [raw_dir, interim_dir, processed_dir, reports_dir, artifacts_dir, prints_dir, dashboards_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # nomes padrão; podem ser sobrescritos no notebook, se desejado
    output_interim = interim_dir / \"interim.parquet\"
    output_processed = processed_dir / \"processed.parquet\"

    # Atualiza globais locais para manifest default
    global ARTIFACTS_DIR, REPORTS_DIR
    ARTIFACTS_DIR = artifacts_dir
    REPORTS_DIR = reports_dir

    return N1Paths(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        prints_dir=prints_dir,
        dashboards_dir=dashboards_dir,
        output_interim=output_interim,
        output_processed=output_processed,
    )

def path_of(paths: N1Paths, key: str):
    return getattr(paths, key) if hasattr(paths, key) else paths[key]  # compat

def load_config(base_abs: Path, local_abs: Optional[Path] = None) -> Dict[str, Any]:
    if not base_abs.exists():
        raise FileNotFoundError(f\"Arquivo obrigatório não encontrado: {base_abs}\")
    cfg = json.loads(base_abs.read_text(encoding=\"utf-8\"))
    print(f\"[INFO] Config carregada de: {base_abs}\")
    if local_abs and local_abs.exists():
        local_cfg = json.loads(local_abs.read_text(encoding=\"utf-8\"))
        cfg.update(local_cfg)
        print(f\"[INFO] Overrides locais: {local_abs}\")
    return cfg

# ============================================================================
# Listagem de arquivos (para facilitar Fonte/RAW)
# ============================================================================
def list_directory_files(dir_path: Path, pattern: str = \"*\", sort_by: str = \"name\") -> pd.DataFrame:
    if not dir_path.exists():
        print(f\"⚠️ Diretório não encontrado: {dir_path}\")
        return pd.DataFrame(columns=[\"Arquivo\", \"Extensão\", \"Tamanho (KB)\", \"Modificado em\"])

    files = list(dir_path.glob(pattern))
    if not files:
        print(f\"⚠️ Nenhum arquivo encontrado em: {dir_path}\")
        return pd.DataFrame(columns=[\"Arquivo\", \"Extensão\", \"Tamanho (KB)\", \"Modificado em\"])

    rows = []
    for f in files:
        if f.is_file():
            stat = f.stat()
            rows.append({
                \"Arquivo\": f.name,
                \"Extensão\": f.suffix.lower(),
                \"Tamanho (KB)\": f\"{stat.st_size/1024:,.1f}\",
                \"Modificado em\": datetime.fromtimestamp(stat.st_mtime).strftime(\"%Y-%m-%d %H:%M:%S\"),
            })

    df_files = pd.DataFrame(rows)
    if sort_by == \"date\":
        df_files = df_files.sort_values(\"Modificado em\", ascending=False)
    else:
        df_files = df_files.sort_values(\"Arquivo\")
    return df_files.reset_index(drop=True)

def suggest_source_path(raw_dir: Path, pattern: str = \"*.csv\") -> None:
    print(f\"🔎 Arquivos em {raw_dir} (filtro: {pattern})\")
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        df = list_directory_files(raw_dir, pattern=pattern)
        display_dataframe_to_user(\"Arquivos em RAW\", df)
    except Exception:
        # fallback para ambientes sem o helper visual
        from IPython.display import display  # type: ignore
        display(list_directory_files(raw_dir, pattern=pattern))

# ============================================================================
# Ingestão flexível
# ============================================================================
def infer_format_from_suffix(path) -> str:
    ext = str(path).lower().rsplit(\".\", 1)[-1]
    if ext in (\"csv\", \"parquet\", \"xlsx\"):
        return ext
    raise ValueError(f\"Formato não suportado (use .csv/.parquet/.xlsx): {path}\")

def load_csv(filepath, **read_kwargs) -> pd.DataFrame:
    logger.info(f\"[load_csv] {filepath}\")
    return pd.read_csv(filepath, **read_kwargs)

def load_table_simple(path, fmt: Optional[str] = None, read_opts: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    fmt = (fmt or infer_format_from_suffix(path)).lower()
    read_opts = read_opts or {}
    logger.info(f\"[load] path={path} | format={fmt} | opts={read_opts}\")
    if fmt == \"csv\":
        return load_csv(path, **read_opts)
    elif fmt == \"parquet\":
        return pd.read_parquet(path, **read_opts)
    elif fmt == \"xlsx\":
        return pd.read_excel(path, **read_opts)
    else:
        raise ValueError(f\"Formato não suportado: {fmt}\")

def merge_chain(base_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], steps) -> pd.DataFrame:
    df = base_df.copy()

    def _is_unique(d: pd.DataFrame, key) -> Tuple[bool, int]:
        key = [key] if isinstance(key, str) else list(key)
        dups = d.duplicated(subset=key, keep=False).sum()
        return (dups == 0, int(dups))

    for (right_name, how, left_on, right_on) in steps:
        if right_name not in tables:
            raise KeyError(f\"Tabela '{right_name}' não encontrada.\")
        right_df = tables[right_name]
        uniq_r, dups_r = _is_unique(right_df, right_on)
        if not uniq_r and how in {\"left\", \"right\"}:
            logger.warning(f\"[merge] '{right_name}': chave {right_on} não é única (dups={dups_r}).\")
        logger.info(f\"[merge] how={how} | left_on={left_on} | right_on={right_on} | with={right_name}\")
        df = df.merge(right_df, how=how, left_on=left_on, right_on=right_on, suffixes=(\"\", f\"__{right_name}\"))
        logger.info(f\"[merge] shape atual: {df.shape}\")
    return df

# ============================================================================
# Overview / memória / limpeza básica
# ============================================================================
def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        \"shape\": df.shape,
        \"columns\": df.columns.tolist(),
        \"dtypes\": {c: str(t) for c, t in df.dtypes.items()},
        \"memory_mb\": float(df.memory_usage(deep=True).sum() / (1024**2)),
        \"na_counts\": df.isna().sum().to_dict(),
    }

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start_mem = out.memory_usage(deep=True).sum() / 1024**2
    for col in out.select_dtypes(include=[\"int64\", \"int32\", \"int16\"]).columns:
        out[col] = pd.to_numeric(out[col], downcast=\"integer\")
    for col in out.select_dtypes(include=[\"float64\", \"float32\"]).columns:
        out[col] = pd.to_numeric(out[col], downcast=\"float\")
    end_mem = out.memory_usage(deep=True).sum() / 1024**2
    logger.info(f\"Memory reduced: {start_mem:.2f}MB -> {end_mem:.2f}MB\")
    return out

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[\"object\", \"string\"]).columns:
        out[col] = out[col].astype(\"string\").str.strip()
    return out

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = df.isna().mean().sort_values(ascending=False).rename(\"missing_rate\").to_frame()
    rep[\"missing_count\"] = df.isna().sum()
    return rep

# ============================================================================
# Deduplicação (com opção de relatório)
# ============================================================================
def deduplicate_rows(
    df: pd.DataFrame,
    subset: Optional[Iterable[str]] = None,
    keep: \"str|bool\" = \"first\",
    log_path: Optional[Path] = None,
    return_report: bool = False
) -> \"pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]\":
    before = len(df)

    dup_mask = df.duplicated(subset=subset, keep=False)
    dups_df = df.loc[dup_mask].copy()

    if subset:
        grp = (
            dups_df.groupby(list(subset), dropna=False, as_index=False)
            .size()
            .rename(columns={\"size\": \"count\"})
        )
        summary_df = grp[grp[\"count\"] >= 2].sort_values(\"count\", ascending=False)
    else:
        key = pd.util.hash_pandas_object(dups_df, index=False)
        summary_df = (
            pd.DataFrame({\"_row_hash\": key})
            .value_counts()
            .reset_index(name=\"count\")
            .query(\"count >= 2\")
            .sort_values(\"count\", ascending=False)
        )

    if log_path is not None and not dups_df.empty:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        dups_df.to_csv(log_path, index=False, encoding=\"utf-8\")
        logger.info(f\"[deduplicate] Duplicatas salvas em: {log_path} (linhas={len(dups_df)})\")

    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df_clean)
    logger.info(f\"[deduplicate] Removed duplicates: {removed} (subset={subset}, keep={keep})\")

    if return_report:
        return df_clean, dups_df, summary_df
    return df_clean

# ============================================================================
# Inferência de numéricos a partir de texto (DataFrame inteiro)
# ============================================================================
def _onehot_encoder_compat():
    try:
        return OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=\"ignore\", sparse=False)

def infer_numeric_like(
    df: pd.DataFrame,
    *,
    columns: List[str] | None = None,
    min_ratio: float = 0.9,
    create_new_col_when_partial: bool = True,
    blacklist: List[str] | None = None,
    whitelist: List[str] | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"
    Converte colunas 'numérico-compatíveis' (armazenadas como texto) para tipo numérico.
    Retorna (df_out, report_df).
    \"\"\"
    out = df.copy()
    bl = set(blacklist or [])
    wl = set(whitelist or [])

    if columns is not None:
        candidates = [c for c in columns if c in out.columns]
    else:
        candidates = out.select_dtypes(include=[\"object\", \"string\", \"category\"]).columns.tolist()

    candidates = list(dict.fromkeys([*candidates, *wl]))
    candidates = [c for c in candidates if c not in bl]

    rows = []
    for col in candidates:
        s = out[col].astype(\"string\")
        s_stripped = s.str.strip()

        notna = s_stripped.notna() & (s_stripped != \"\")
        base_count = int(notna.sum())
        if base_count == 0:
            rows.append({\"column\": col, \"ratio\": 0.0, \"action\": \"skip_empty\", \"converted\": 0, \"total\": 0,
                         \"before_dtype\": str(out[col].dtype), \"after_dtype\": str(out[col].dtype)})
            continue

        s_clean = (
            s_stripped
            .str.replace(r\"[\\s\\t\\r\\n\\$\\€\\£R\\$]\", \"\", regex=True)
            .str.replace(\"\\u00A0\", \"\", regex=False)
        )
        has_dot = s_clean.str.contains(r\"\\.\", na=False).any()
        has_comma = s_clean.str.contains(r\",\", na=False).any()
        candidate = s_clean
        if has_dot and has_comma:
            candidate = candidate.str.replace(\".\", \"\", regex=False).str.replace(\",\", \".\", regex=False)
        elif has_comma and not has_dot:
            candidate = candidate.str.replace(\",\", \".\", regex=False)

        candidate_num = candidate.str.replace(\"%\", \"\", regex=False)
        numeric = pd.to_numeric(candidate_num, errors=\"coerce\")
        convertible = int(numeric[notna].notna().sum())
        ratio = convertible / base_count if base_count else 0.0

        before_dtype = str(out[col].dtype)
        if convertible == 0:
            action = \"skip_no_conversion\"
            after_dtype = before_dtype
        elif ratio >= min_ratio:
            # garante dtype mutável antes de setitem
            out[col] = out[col].astype(\"object\")
            out.loc[notna, col] = numeric[notna].to_numpy()
            out[col] = pd.to_numeric(out[col], errors=\"coerce\")
            action = \"inplace_convert\"
            after_dtype = str(out[col].dtype)
        else:
            if create_new_col_when_partial:
                new_col = f\"{col}_num\"
                out[new_col] = pd.NA
                out[new_col] = out[new_col].astype(\"object\")
                out.loc[notna, new_col] = numeric[notna].to_numpy()
                out[new_col] = pd.to_numeric(out[new_col], errors=\"coerce\")
                action = f\"partial_to_{new_col}\"
                after_dtype = str(out.get(new_col).dtype)
            else:
                out[col] = out[col].astype(\"object\")
                out.loc[notna, col] = numeric[notna].to_numpy()
                out[col] = pd.to_numeric(out[col], errors=\"coerce\")
                action = \"partial_inplace\"
                after_dtype = str(out[col].dtype)

        rows.append({
            \"column\": col,
            \"ratio\": float(ratio),
            \"action\": action,
            \"converted\": int(convertible),
            \"total\": int(base_count),
            \"before_dtype\": before_dtype,
            \"after_dtype\": after_dtype,
        })

    report = (
        pd.DataFrame(rows, columns=[\"column\", \"ratio\", \"action\", \"converted\", \"total\", \"before_dtype\", \"after_dtype\"])
        .sort_values([\"action\", \"column\"], ascending=[False, True])
        .reset_index(drop=True)
    )
    actions = dict(report[\"action\"].value_counts())
    logger.info(f\"[infer_numeric_like] {len(candidates)} colunas verificadas. Ações: {actions}\")
    return out, report

# ============================================================================
# Encoding / Scaling (safe + compat)
# ============================================================================
def encode_categories(df: pd.DataFrame, encoding: str = \"onehot\") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cat_cols = df.select_dtypes(include=[\"object\", \"category\"]).columns.tolist()
    meta: Dict[str, Any] = {\"categorical_columns\": cat_cols, \"encoding\": encoding}
    if not cat_cols:
        return df, meta

    if encoding == \"onehot\":
        encoder = _onehot_encoder_compat()
        arr = encoder.fit_transform(df[cat_cols])
        encoded = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), encoded], axis=1)
        meta[\"categories_\"] = {c: list(map(str, cats)) for c, cats in zip(cat_cols, encoder.categories_)}
    elif encoding == \"ordinal\":
        encoder = OrdinalEncoder(handle_unknown=\"use_encoded_value\", unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        meta[\"categories_\"] = {c: list(map(str, cats)) for c, cats in zip(cat_cols, encoder.categories_)}
    else:
        raise ValueError(\"Unsupported encoding type.\")
    return df, meta

def _warn_high_cardinality(df: pd.DataFrame, cols: list[str], threshold: int = 50) -> dict[str, int]:
    high = {c: int(df[c].nunique(dropna=False)) for c in cols if df[c].nunique(dropna=False) > threshold}
    if high:
        logger.warning(f\"[encode] Alta cardinalidade (> {threshold}): {high}\")
    return high

def encode_categories_safe(
    df: pd.DataFrame,
    *,
    method: str = \"onehot\",
    exclude_cols: list[str] | None = None,
    high_card_threshold: int = 50
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exclude: Set[str] = set(exclude_cols or [])
    cat_cols = [c for c in df.select_dtypes(include=[\"object\", \"category\"]).columns if c not in exclude]

    high = _warn_high_cardinality(df, cat_cols, threshold=high_card_threshold)
    keep_df = df[list(exclude & set(df.columns))].copy()
    work_df = df[[c for c in df.columns if c not in exclude]].copy()

    work_df, meta = encode_categories(work_df, encoding=method)
    meta.update({\"excluded\": sorted(list(exclude)), \"high_cardinality\": high, \"encoding\": method})

    out = pd.concat([keep_df.reset_index(drop=True), work_df.reset_index(drop=True)], axis=1)
    return out, meta

def _is_dummy_or_boolean(s: pd.Series) -> bool:
    vals = set(pd.unique(s.dropna()))
    return s.dtype == \"bool\" or vals.issubset({0, 1})

def scale_numeric(df: pd.DataFrame, method: str = \"standard\") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    num_cols = df.select_dtypes(include=[\"number\"]).columns.tolist()
    meta: Dict[str, Any] = {\"numeric_columns\": num_cols, \"scaler\": method}
    if not num_cols:
        return df, meta
    if method == \"standard\":
        scaler = StandardScaler()
    elif method == \"minmax\":
        scaler = MinMaxScaler()
    else:
        raise ValueError(\"Unsupported scaler.\")
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, meta

def scale_numeric_safe(
    df: pd.DataFrame,
    *,
    method: str = \"standard\",
    exclude_cols: list[str] | None = None,
    only_continuous: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exclude: Set[str] = set(exclude_cols or [])
    keep_df = df[list(exclude & set(df.columns))].copy()
    work_df = df[[c for c in df.columns if c not in exclude]].copy()

    all_num = work_df.select_dtypes(include=[\"number\", \"boolean\"]).columns.tolist()
    target_num = [c for c in all_num if not _is_dummy_or_boolean(work_df[c])] if only_continuous else all_num
    if not target_num:
        logger.info(\"[scale] nenhuma coluna contínua elegível para escalonamento.\")
        return df, {\"numeric_columns\": [], \"scaler\": method, \"excluded\": sorted(list(exclude))}

    # aplica scaler apenas no bloco alvo
    if method == \"standard\":
        scaler = StandardScaler()
    elif method == \"minmax\":
        scaler = MinMaxScaler()
    else:
        raise ValueError(\"Unsupported scaler.\")
    work_df[target_num] = scaler.fit_transform(work_df[target_num])

    out = pd.concat([keep_df.reset_index(drop=True), work_df.reset_index(drop=True)], axis=1)
    meta = {\"numeric_columns\": target_num, \"scaler\": method, \"excluded\": sorted(list(exclude)), \"only_continuous\": only_continuous}
    return out, meta

def apply_encoding_and_scaling(
    df: pd.DataFrame,
    *,
    encode_cfg: dict | None = None,
    scale_cfg: dict | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    encode_cfg = encode_cfg or {}
    scale_cfg  = scale_cfg  or {}
    enc_meta, scl_meta = {}, {}

    if encode_cfg.get(\"enabled\", True):
        df, enc_meta = encode_categories_safe(
            df,
            method=encode_cfg.get(\"type\", \"onehot\"),
            exclude_cols=encode_cfg.get(\"exclude_cols\", []),
            high_card_threshold=int(encode_cfg.get(\"high_card_threshold\", 50)),
        )
        logger.info(f\"[encode] type={enc_meta.get('encoding')} | cols={len(enc_meta.get('categorical_columns', []))}\")

    if scale_cfg.get(\"enabled\", False):
        df, scl_meta = scale_numeric_safe(
            df,
            method=scale_cfg.get(\"method\", \"standard\"),
            exclude_cols=scale_cfg.get(\"exclude_cols\", []),
            only_continuous=bool(scale_cfg.get(\"only_continuous\", True)),
        )
        logger.info(f\"[scale] method={scl_meta.get('scaler')} | cols={len(scl_meta.get('numeric_columns', []))}\")

    return df, enc_meta, scl_meta

# ============================================================================
# Datas
# ============================================================================
import re

def detect_date_candidates(df: pd.DataFrame, pattern: str) -> List[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    return [c for c in df.columns if rx.search(c) or pd.api.types.is_datetime64_any_dtype(df[c])]

def _maybe_to_datetime(s: pd.Series, *, dayfirst: bool, utc: bool, formats: List[str]) -> pd.Series:
    for fmt in formats or []:
        try:
            out = pd.to_datetime(s, format=fmt, errors=\"coerce\", dayfirst=dayfirst, utc=utc)
            if out.notna().mean() > 0:
                return out
        except Exception:
            pass
    return pd.to_datetime(s, errors=\"coerce\", dayfirst=dayfirst, utc=utc)

def parse_dates_with_report(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    detect_regex = cfg.get(\"detect_regex\", r\"(date|data|dt_|_dt$|_date$)\")
    dayfirst     = bool(cfg.get(\"dayfirst\", False))
    utc          = bool(cfg.get(\"utc\", False))
    formats      = cfg.get(\"formats\", []) or []
    min_ratio    = float(cfg.get(\"min_ratio\", 0.8))

    explicit = cfg.get(\"explicit_cols\", []) or []
    candidates = list(dict.fromkeys(detect_date_candidates(df, detect_regex) + [c for c in explicit if c in df.columns]))

    report_rows: List[Dict[str, Any]] = []
    parsed_cols: List[str] = []
    out = df.copy()

    for c in candidates:
        s_raw = out[c]
        s_dt = s_raw if pd.api.types.is_datetime64_any_dtype(s_raw) else _maybe_to_datetime(
            s_raw, dayfirst=dayfirst, utc=utc, formats=formats
        )
        ratio = float(s_dt.notna().mean())
        converted = ratio >= min_ratio
        report_rows.append({\"column\": c, \"parsed_ratio\": ratio, \"converted\": converted})
        if converted:
            out[c] = s_dt

    parse_report = pd.DataFrame(report_rows) if report_rows else pd.DataFrame(columns=[\"column\", \"parsed_ratio\", \"converted\"])
    if not parse_report.empty:
        parse_report = parse_report.sort_values([\"converted\", \"parsed_ratio\"], ascending=[False, False]).reset_index(drop=True)
        parsed_cols = parse_report.query(\"converted == True\")[\"column\"].tolist()

    logger.info(f\"[dates] candidates={candidates}\")
    logger.info(f\"[dates] parsed_ok={parsed_cols}\")
    return out, parse_report, parsed_cols

def expand_date_features(
    df: pd.DataFrame,
    cols: List[str],
    *,
    features: List[str] = None,
    prefix_mode: str = \"auto\",
    fixed_prefix: str = None
) -> List[str]:
    features = features or [\"year\",\"month\",\"day\",\"dayofweek\",\"quarter\",\"week\",\"is_month_start\",\"is_month_end\"]
    created = []
    out = df.copy()
    for col in cols:
        s = out[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            continue
        pfx = fixed_prefix if (prefix_mode == \"fixed\" and fixed_prefix) else col
        if \"year\" in features:
            out[f\"{pfx}_year\"] = s.dt.year; created.append(f\"{pfx}_year\")
        if \"month\" in features:
            out[f\"{pfx}_month\"] = s.dt.month; created.append(f\"{pfx}_month\")
        if \"day\" in features:
            out[f\"{pfx}_day\"] = s.dt.day; created.append(f\"{pfx}_day\")
        if \"dayofweek\" in features:
            out[f\"{pfx}_dow\"] = s.dt.dayofweek; created.append(f\"{pfx}_dow\")
        if \"quarter\" in features:
            out[f\"{pfx}_quarter\"] = s.dt.quarter; created.append(f\"{pfx}_quarter\")
        if \"week\" in features:
            out[f\"{pfx}_week\"] = s.dt.isocalendar().week.astype(\"Int64\"); created.append(f\"{pfx}_week\")
        if \"is_month_start\" in features:
            out[f\"{pfx}_is_month_start\"] = s.dt.is_month_start; created.append(f\"{pfx}_is_month_start\")
        if \"is_month_end\" in features:
            out[f\"{pfx}_is_month_end\"] = s.dt.is_month_end; created.append(f\"{pfx}_is_month_end\")
    df.loc[:, out.columns] = out
    logger.info(f\"[dates] created_features={len(created)}\")
    return created

def build_calendar_from(df: pd.DataFrame, date_col: str, freq: str = \"D\") -> pd.DataFrame:
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f\"Coluna inválida para calendário: {date_col}\")
    start = df[date_col].min()
    end = df[date_col].max()
    idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq=freq)
    cal = pd.DataFrame({\"date\": idx})
    cal[\"year\"] = cal[\"date\"].dt.year
    cal[\"month\"] = cal[\"date\"].dt.month
    cal[\"day\"] = cal[\"date\"].dt.day
    cal[\"quarter\"] = cal[\"date\"].dt.quarter
    cal[\"week\"] = cal[\"date\"].dt.isocalendar().week.astype(\"Int64\")
    cal[\"dow\"] = cal[\"date\"].dt.dayofweek
    cal[\"is_month_start\"] = cal[\"date\"].dt.is_month_start
    cal[\"is_month_end\"] = cal[\"date\"].dt.is_month_end
    cal[\"month_name\"] = cal[\"date\"].dt.month_name()
    cal[\"day_name\"] = cal[\"date\"].dt.day_name()
    return cal

# ============================================================================
# Texto
# ============================================================================
@with_step(\"extract_text_features\")
def extract_text_features(
    df: pd.DataFrame,
    *,
    lower: bool = True,
    strip_collapse_ws: bool = True,
    keywords: list[str] | None = None,
    blacklist: Iterable[str] | None = None,
    export_summary: bool = True,
    summary_dir: Path | None = None,
    config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    keywords = keywords or []
    blacklist = set(map(str, blacklist or []))
    text_cols = [c for c in out.columns if out[c].dtype == \"object\" and c not in blacklist]

    created_cols = []
    processed_cols = []

    for c in text_cols:
        s = out[c].astype(\"string\").fillna(\"\")
        if strip_collapse_ws:
            s = s.str.strip().str.replace(r\"\\s+\", \" \", regex=True)
        if lower:
            s = s.str.lower()

        out[f\"{c}_len\"] = s.str.len()
        out[f\"{c}_word_count\"] = s.str.split().map(len)
        out[f\"{c}_alpha_count\"] = s.str.count(r\"[A-Za-z]\")
        out[f\"{c}_digit_count\"] = s.str.count(r\"\\d\")
        created_cols += [f\"{c}_len\", f\"{c}_word_count\", f\"{c}_alpha_count\", f\"{c}_digit_count\"]

        for kw in keywords:
            safe_kw = str(kw).strip().replace(\" \", \"_\")
            col_name = f\"{c}_has_{safe_kw}\"
            out[col_name] = s.str.contains(str(kw), case=False, na=False)
            created_cols.append(col_name)

        processed_cols.append(c)

    logger.info(f\"[text] colunas processadas: {processed_cols}\")
    logger.info(f\"[text] features criadas: {len(created_cols)}\")

    summary_df = pd.DataFrame([
        {\"text_col\": c,
         \"keywords_cols\": \", \".join([f\"{c}_has_{str(kw).strip().replace(' ', '_')}\" for kw in keywords])}
        for c in processed_cols
    ])

    if export_summary:
        summary_dir = summary_dir or REPORTS_DIR / \"text_features\"
        summary_path = summary_dir / \"summary.csv\"
        save_report_df(summary_df, summary_path, config=config)

    return out, summary_df

# ============================================================================
# Normalização Categórica (com relatório)
# ============================================================================
# utils_data.py  — substitua a função normalize_categories inteira por esta versão

@with_step(\"normalize_categories\")
def normalize_categories(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
    replacements: dict | None = None,
    case: str = \"title\",          # \"lower\" | \"upper\" | \"title\" | \"none\"
    trim: bool = True,
    collapse_ws: bool = True,
    to_na: list[str] | None = None,
    exclude: list[str] | None = None,
    cast_to_category: bool = False,
    report_path: Path | None = None,
    config: dict | None = None,
):
    \"\"\"
    Padroniza colunas categóricas (strings/category) com:
      - trim, colapso de espaços
      - case normalization
      - mapa global de substituições (replacements)
      - per-column map opcional (cfg[\"per_column_map\"])
      - mapeamento para NA (to_na)
    Retorna: (df_out, report_df)
    \"\"\"
    out = df.copy()
    exclude = set(exclude or [])
    to_na = [str(x).strip().lower() for x in (to_na or [])]

    # Lê do cfg (se fornecido) valores padrão/ampliados
    if cfg:
        replacements = cfg.get(\"global_map\", replacements or {})
        case = cfg.get(\"case\", case)
        trim = bool(cfg.get(\"trim\", trim))
        collapse_ws = bool(cfg.get(\"collapse_ws\", collapse_ws))
        cast_to_category = bool(cfg.get(\"cast_to_category\", cast_to_category))
        exclude |= set(cfg.get(\"exclude\", []))
        # null values
        if cfg.get(\"null_values\"):
            to_na = [str(x).strip().lower() for x in cfg[\"null_values\"]]
        per_column_map: dict[str, dict] = cfg.get(\"per_column_map\", {}) or {}
    else:
        per_column_map = {}

    # Seleciona colunas categóricas elegíveis
    cat_cols = [c for c in out.select_dtypes(include=[\"object\", \"string\", \"category\"]).columns if c not in exclude]
    rows: list[dict[str, object]] = []

    # Preparar mapa global (case-insensitive de entrada)
    # Observação: só tornamos as CHAVES do mapa \"flexíveis\" na busca; o VALOR de saída é usado como está.
    def _apply_global_map(series: pd.Series) -> pd.Series:
        if not replacements:
            return series
        if not isinstance(replacements, dict):
            return series
        # aplicamos substituições por igualdade de string normalizada
        # construímos um dict de lookup de entrada normalizada -> saída
        norm_map = {str(k).strip().lower(): v for k, v in replacements.items()}
        s_norm = series.astype(\"string\").str.strip().str.lower()
        mask = s_norm.isin(norm_map.keys())
        # onde “bateu”, substitui pelo valor mapeado; senão mantém original
        return pd.Series(np.where(mask, s_norm.map(norm_map), series.astype(\"string\")), index=series.index, dtype=\"string\")

    for col in cat_cols:
        s0 = out[col].astype(\"string\")
        before = s0.copy()

        s = s0
        if trim:
            s = s.str.strip()
        if collapse_ws:
            s = s.str.replace(r\"\\s+\", \" \", regex=True)

        # global_map (case-insensitive para chave)
        s = _apply_global_map(s)

        # per-column map (aplicado literal na saída atual)
        if col in per_column_map and isinstance(per_column_map[col], dict):
            s = s.replace(per_column_map[col])

        # case normalization
        if case == \"lower\":
            s = s.str.lower()
        elif case == \"upper\":
            s = s.str.upper()
        elif case == \"title\":
            s = s.str.title()
        # \"none\" => não altera

        # to_na (valores que devem virar NA — comparação em lower/strip)
        if to_na:
            s_low = s.str.strip().str.lower()
            s = s.mask(s_low.isin(to_na), other=pd.NA)

        out[col] = s

        # métricas
        # compara com cuidado para NAs (preenche temp para comparar)
        bcmp = before.fillna(\"__NA__\")
        acmp = out[col].fillna(\"__NA__\")
        changed = int((bcmp != acmp).sum())
        unique_before = int(pd.Series(before).nunique(dropna=False))
        unique_after  = int(pd.Series(out[col]).nunique(dropna=False))

        # amostras (até 3 valores distintos para visual)
        def _sample_unique(s: pd.Series, k=3):
            vals = list(pd.Series(s).drop_duplicates().astype(str))[:k]
            return \", \".join(vals)

        rows.append({
            \"column\": col,
            \"before_sample\": _sample_unique(before),
            \"after_sample\":  _sample_unique(out[col]),
            \"changes\": changed,
            \"unique_before\": unique_before,
            \"unique_after\":  unique_after,
        })

        if cast_to_category:
            try:
                out[col] = out[col].astype(\"category\")
            except Exception:
                pass

    # Constrói relatório com colunas garantidas
    report_cols = [\"column\", \"before_sample\", \"after_sample\", \"changes\", \"unique_before\", \"unique_after\"]
    report_df = pd.DataFrame(rows, columns=report_cols)

    # Ordena de forma segura (só se houver linhas)
    if not report_df.empty and \"changes\" in report_df.columns:
        report_df = report_df.sort_values(\"changes\", ascending=False).reset_index(drop=True)

    # Persistência do relatório
    if report_path is None:
        report_path = REPORTS_DIR / \"cat_normalization.csv\"
    save_report_df(report_df, report_path, config=config)

    # Manifest
    record_step(\"normalize_categories:report\", payload={\"path\": str(report_path)}, config=config)
    update_manifest(\"reports\", [str(report_path)])

    # Logging resumo
    changed_cols = [r[\"column\"] for r in rows if r.get(\"changes\", 0) > 0]
    if changed_cols:
        logger.info(f\"[N1] Padronização categórica aplicada em: {changed_cols}\")
    else:
        logger.info(\"[N1] Padronização categórica: nenhuma alteração necessária.\")

    return out, report_df

# ============================================================================
# Imputação com flags
# ============================================================================
def simple_impute_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[\"number\"]).columns
    cat_cols = out.select_dtypes(exclude=[\"number\"]).columns

    for c in num_cols:
        if out[c].isna().any():
            mask = out[c].isna()
            out[c] = out[c].fillna(out[c].median())
            out[f\"was_imputed_{c}\"] = mask
            logger.info(f\"[impute] '{c}' → {mask.sum()} valores (mediana).\"

)
    for c in cat_cols:
        if out[c].isna().any():
            mask = out[c].isna()
            out[c] = out[c].fillna(out[c].mode().iloc[0])
            out[f\"was_imputed_{c}\"] = mask
            logger.info(f\"[impute] '{c}' → {mask.sum()} valores (moda).\")


    return out

# ============================================================================
# Target helpers
# ============================================================================
import unicodedata

def _strip_accents_txt(txt: str) -> str:
    if txt is None:
        return txt
    try:
        return \"\".join(ch for ch in unicodedata.normalize(\"NFKD\", str(txt)) if not unicodedata.combining(ch))
    except Exception:
        return str(txt)

def build_target(
    df: pd.DataFrame,
    *,
    source_col: str,
    name: str = \"target\",
    positive_values = (\"yes\",\"sim\",\"y\",\"true\",\"1\",1,True),
    negative_values = (\"no\",\"não\",\"nao\",\"n\",\"false\",\"0\",0,False),
    to_dtype: str = \"int\",           # \"int\" | \"bool\"
    drop_source: bool = False,
    positive_label: str = \"positive\",
    negative_label: str = \"negative\"
) -> tuple[pd.DataFrame, str, dict | None, pd.DataFrame]:
    if source_col not in df.columns:
        raise KeyError(f\"[build_target] Coluna fonte '{source_col}' não encontrada.\")

    s_raw = df[source_col].astype(\"string\")
    s_norm = s_raw.str.strip().str.lower().apply(_strip_accents_txt)

    pos_set = set(_strip_accents_txt(v).strip().lower() for v in positive_values)
    neg_set = set(_strip_accents_txt(v).strip().lower() for v in negative_values)

    is_pos = s_norm.isin(pos_set)
    is_neg = s_norm.isin(neg_set)

    mapped = pd.Series(np.where(is_pos, 1, np.where(is_neg, 0, np.nan)), index=df.index)

    if to_dtype == \"bool\":
        target_series = mapped.replace({1: True, 0: False}).astype(\"boolean\")
    else:
        target_series = mapped.astype(\"Int64\")

    out = df.copy()
    out[name] = target_series

    if drop_source and name != source_col:
        out.drop(columns=[source_col], inplace=True, errors=\"ignore\")

    total = len(out)
    na_count = int(out[name].isna().sum())
    if to_dtype == \"bool\":
        pos_count = int((out[name] == True).sum())
        neg_count = int((out[name] == False).sum())
    else:
        pos_count = int((out[name] == 1).sum())
        neg_count = int((out[name] == 0).sum())

    report = pd.DataFrame([{
        \"source_col\": source_col,
        \"target_col\": name,
        \"dtype\": to_dtype,
        \"positives\": pos_count,
        \"negatives\": neg_count,
        \"nulls\": na_count,
        \"total\": total
    }])

    class_map = None
    if to_dtype == \"int\":
        class_map = {negative_label: 0, positive_label: 1}
    elif to_dtype == \"bool\":
        class_map = {negative_label: False, positive_label: True}

    logger.info(f\"[target] '{name}' de '{source_col}' → pos={pos_count} neg={neg_count} nulls={na_count} total={total}\")
    return out, name, class_map, report

def ensure_target_from_config(
    df: pd.DataFrame,
    config: dict,
    *,
    verbose: bool = True
) -> tuple[pd.DataFrame, str, dict | None, pd.DataFrame]:
    tgt_cfg = dict(config.get(\"target_cfg\", {}))

    source_col     = tgt_cfg.get(\"source_col\", \"Churn\")
    target_name    = tgt_cfg.get(\"name\", \"target\")
    pos_values     = tgt_cfg.get(\"positive_values\", [\"yes\",\"sim\",\"y\",\"true\",\"1\",1,True])
    neg_values     = tgt_cfg.get(\"negative_values\", [\"no\",\"não\",\"nao\",\"n\",\"false\",\"0\",0,False])
    to_dtype       = tgt_cfg.get(\"to_dtype\", \"int\")
    drop_source    = bool(tgt_cfg.get(\"drop_source\", False))
    positive_label = tgt_cfg.get(\"positive_label\", \"positive\")
    negative_label = tgt_cfg.get(\"negative_label\", \"negative\")

    df_out, tgt_name, class_map, rep = build_target(
        df,
        source_col=source_col,
        name=target_name,
        positive_values=pos_values,
        negative_values=neg_values,
        to_dtype=to_dtype,
        drop_source=drop_source,
        positive_label=positive_label,
        negative_label=negative_label
    )

    config[\"target_column\"] = tgt_name
    if verbose:
        print(f\"[target] Definido target_column='{tgt_name}' (fonte='{source_col}')\")
    return df_out, tgt_name, class_map, rep

# ============================================================================
# Catálogo de DataFrames
# ============================================================================
@dataclass
class TableInfo:
    name: str
    rows: int
    cols: int
    memory_mb: float

class TableStore:
    def __init__(self, initial: dict[str, pd.DataFrame] | None = None, current: str | None = None):
        self._store: dict[str, pd.DataFrame] = dict(initial or {})
        self.current: str | None = current if current in (initial or {}) else (next(iter(self._store)) if self._store else None)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self._store[name]

    def add(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(\"objeto precisa ser um pandas.DataFrame\")
        self._store[name] = df
        logger.info(f\"[tables] add '{name}' shape={df.shape}\")
        if set_current:
            self.current = name

    def get(self, name: str | None = None) -> pd.DataFrame:
        name = name or self.current
        if name is None:
            raise KeyError(\"Nenhuma tabela atual definida.\")
        return self._store[name]

    def rename(self, old: str, new: str) -> None:
        if new in self._store:
            raise KeyError(f\"Já existe tabela com nome '{new}'.\")
        self._store[new] = self._store.pop(old)
        if self.current == old:
            self.current = new
        logger.info(f\"[tables] rename '{old}' → '{new}'\")

    def drop(self, name: str) -> None:
        self._store.pop(name)
        logger.info(f\"[tables] drop '{name}'\")
        if self.current == name:
            self.current = next(iter(self._store)) if self._store else None

    def list(self) -> pd.DataFrame:
        infos = []
        for n, d in self._store.items():
            mem = float(d.memory_usage(deep=True).sum() / (1024**2))
            infos.append(TableInfo(n, len(d), d.shape[1], mem))
        return pd.DataFrame([i.__dict__ for i in infos]).sort_values(\"name\")

    def use(self, name: str) -> pd.DataFrame:
        if name not in self._store:
            raise KeyError(f\"Tabela '{name}' não existe. Disponíveis: {list(self._store)}\")
        self.current = name
        df = self._store[name]
        print(f\"[using] current='{name}' shape={df.shape}\")
        return df

# ============================================================================
# N1: Orquestração “Qualidade & Tipagem” (com manifest)
# ============================================================================
@with_step(\"n1_quality_typing\")
def n1_quality_typing(df: pd.DataFrame, config: dict, reports_dir: Path) -> tuple[pd.DataFrame, dict]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, pd.DataFrame] = {}
    out = df.copy()

    if config.get(\"strip_whitespace\", True):
        out = strip_whitespace(out)
        logger.info(\"[N1] strip_whitespace aplicado.\")

    if config.get(\"cast_numeric_like\", True):
        out, cast_report = infer_numeric_like(
            out,
            columns=None,
            min_ratio=0.9,
            create_new_col_when_partial=True,
            blacklist=[\"customerID\"]
        )
        reports[\"cast_report\"] = cast_report
        save_report_df(cast_report, reports_dir / \"cast_report.csv\", config=config)

    if config.get(\"infer_types\", True):
        out = reduce_memory_usage(out)
        logger.info(\"[N1] reduce_memory_usage aplicado.\")

    if config.get(\"deduplicate\", True):
        subset = config.get(\"deduplicate_subset\") or None
        keep = config.get(\"deduplicate_keep\", \"first\")
        log_path = reports_dir / config.get(\"deduplicate_log_filename\", \"duplicates.csv\")
        out, dups_df, dup_summary = deduplicate_rows(
            out, subset=subset, keep=keep, log_path=log_path, return_report=True
        )
        if not dups_df.empty:
            reports[\"duplicates\"] = dups_df
            reports[\"duplicates_summary\"] = dup_summary
            log_report_path(log_path, config=config)  # já salvo no disco

    # snapshot de overview
    overview = basic_overview(out)
    save_text(json.dumps(overview, indent=2, ensure_ascii=False),
              reports_dir / \"overview_after_quality.json\",
              config=config, section=\"reports\")

    return out, reports

# ============================================================================
# Export utilitário
# ============================================================================
def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == \".csv\":
        df.to_csv(path, index=False, encoding=\"utf-8\")
    elif ext == \".parquet\":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f\"Extensão não suportada: {ext}\")
    logger.info(f\"[save_table] {path}\")

def save_named_interims(named_frames: dict[str, pd.DataFrame], base_dir: Path, fmt: str = \"parquet\") -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, dfx in named_frames.items():
        out = base_dir / f\"{name}_interim.{fmt}\"
        if fmt == \"csv\":
            dfx.to_csv(out, index=False, encoding=\"utf-8\")
        elif fmt == \"parquet\":
            dfx.to_parquet(out, index=False)
        else:
            raise ValueError(\"fmt deve ser 'csv' ou 'parquet'\")
        logger.info(f\"[interim] saved: {out} shape={dfx.shape}\")

# ============================================================================
# Relatório humano (MD + opcional PDF)
# ============================================================================
def generate_human_report_md(
    manifest_path: Path = ARTIFACTS_DIR / \"manifest.json\",
    out_md_path: Path = REPORTS_DIR / \"run_report.md\",
    *,
    extra_sections: dict[str, str] | None = None,
    config: dict | None = None
) -> Path:
    m = json.loads(manifest_path.read_text(encoding=\"utf-8\")) if manifest_path.exists() else {}
    lines = []
    lines.append(f\"# 📘 Data Project Run Report\\n\")
    lines.append(f\"- Generated at: {datetime.now().isoformat(timespec='seconds')}\")
    lines.append(f\"- Manifest file: `{manifest_path.as_posix()}`\\n\")

    lines.append(\"## 🏁 Pipeline Steps\")
    steps = m.get(\"run_steps\", [])
    if steps:
        for s in steps:
            lines.append(f\"- **{s.get('ts','')}** — {s.get('name','(step)')}\")
    else:
        lines.append(\"_No steps recorded._\")
    lines.append(\"\")

    lines.append(\"## 🧩 Artifacts\")
    arts = m.get(\"artifacts\", [])
    if arts:
        for a in arts:
            lines.append(f\"- `{a}`\")
    else:
        lines.append(\"_No artifacts recorded._\")
    lines.append(\"\")

    lines.append(\"## 📝 Reports\")
    reps = m.get(\"reports\", [])
    if reps:
        for r in reps:
            lines.append(f\"- `{r}`\")
    else:
        lines.append(\"_No reports recorded._\")
    lines.append(\"\")

    if extra_sections:
        for title, body in extra_sections.items():
            lines.append(f\"## {title}\\n{body}\\n\")

    md = \"\\n\".join(lines)
    save_text(md, out_md_path, config=config, section=\"reports\")
    return out_md_path

def md_to_pdf(md_path: Path, pdf_path: Path, *, engine: str = \"weasyprint\") -> Path:
    \"\"\"
    Converte Markdown → PDF.
    Requer: weasyprint (recomendado) + (pandoc para md->html, se disponível).
    \"\"\"
    import subprocess, shutil
    md_path = Path(md_path); pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == \"weasyprint\":
        html_path = pdf_path.with_suffix(\".html\")
        if shutil.which(\"pandoc\"):
            subprocess.run([\"pandoc\", md_path.as_posix(), \"-o\", html_path.as_posix()], check=True)
        else:
            html_path.write_text(f\"<pre>{md_path.read_text(encoding='utf-8')}</pre>\", encoding=\"utf-8\")
        subprocess.run([\"weasyprint\", html_path.as_posix(), pdf_path.as_posix()], check=True)
        return pdf_path

    raise ValueError(\"Engine não suportado. Use 'weasyprint'.\")

# =============================================================================
# 🚨 Detecção de Outliers 
# =============================================================================
import numpy as np
import pandas as pd

def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    iqr_factor: float = 1.5,
    persist: bool = True,
) -> pd.DataFrame:
    \"\"\"
    Marca outliers usando o método do IQR (Interquartile Range).
    Cria colunas booleanas <col>_is_outlier.
    \"\"\"
    df = df.copy()
    cols = cols or df.select_dtypes(include=[\"number\"]).columns.tolist()
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        df[f\"{c}_is_outlier\"] = (df[c] < lower) | (df[c] > upper)
    logger.info(f\"[outliers] IQR aplicado ({len(cols)} colunas).\"
)
    if persist:
        update_manifest(\"steps\", {\"outliers_iqr\": {\"cols\": cols, \"iqr_factor\": iqr_factor}})
    return df


def detect_outliers_zscore(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    threshold: float = 3.0,
    persist: bool = True,
) -> pd.DataFrame:
    \"\"\"
    Marca outliers usando o método do Z-score.
    Cria colunas booleanas <col>_is_outlier.
    \"\"\"
    df = df.copy()
    cols = cols or df.select_dtypes(include=[\"number\"]).columns.tolist()
    for c in cols:
        mu, sigma = df[c].mean(), df[c].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            df[f\"{c}_is_outlier\"] = False
        else:
            z = (df[c] - mu) / sigma
            df[f\"{c}_is_outlier\"] = z.abs() > threshold
    logger.info(f\"[outliers] Z-score aplicado ({len(cols)} colunas, threshold={threshold}).\")
    if persist:
        update_manifest(\"steps\", {\"outliers_zscore\": {\"cols\": cols, \"threshold\": threshold}})
    return df




# ============================================================================
# __all__
# ============================================================================
__all__ = [
    # bootstrap / config
    \"ensure_project_root\", \"set_random_seed\", \"set_display\",
    \"load_config\", \"resolve_n1_paths\", \"N1Paths\", \"path_of\",

    # ingestão / merges
    \"infer_format_from_suffix\", \"load_csv\", \"load_table_simple\", \"merge_chain\",

    # listagem RAW
    \"list_directory_files\", \"suggest_source_path\",

    # overview / memória / limpeza
    \"basic_overview\", \"reduce_memory_usage\", \"strip_whitespace\", \"missing_report\",

    # numérico-like + dedup
    \"infer_numeric_like\", \"deduplicate_rows\",

    # encoding / scaling
    \"encode_categories\", \"encode_categories_safe\",
    \"scale_numeric\", \"scale_numeric_safe\", \"apply_encoding_and_scaling\",

    # datas
    \"detect_date_candidates\", \"parse_dates_with_report\", \"expand_date_features\", \"build_calendar_from\",

    # texto
    \"extract_text_features\",

    # categóricas
    \"normalize_categories\",

    # imputação
    \"simple_impute_with_flags\",

    # target
    \"build_target\", \"ensure_target_from_config\",

    # catálogo
    \"TableStore\", \"TableInfo\",

    # export
    \"save_table\", \"save_named_interims\",

    # manifest base
    \"load_manifest\", \"save_manifest\", \"update_manifest\", \"save_artifact\",

    # reporting helpers
    \"save_report_df\", \"save_text\", \"log_report_path\", \"log_artifact_path\",
    \"record_step\", \"with_step\",

    # n1 orquestrador
    \"n1_quality_typing\",

    # human report
    \"generate_human_report_md\", \"md_to_pdf\",
]