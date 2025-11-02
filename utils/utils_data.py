# -*- coding: utf-8 -*-
"""utils_data.py — Utilitários centrais para projetos de dados (N1/N2/N3).

Versão "merge" que combina as funções antigas (compatibilidade retroativa)
com as melhorias da v1.1.3. Objetivos:
- Manter assinaturas antigas que seus notebooks já usam;
- Adicionar as funções novas e relatórios extras;
- Fornecer "wrappers" quando a assinatura/retorno mudou.

Principais compatibilidades:
- resolve_n1_paths aceita tanto (root) quanto (config, root);
- N1Paths tem aliases .raw_dir/.interim_dir/.processed_dir/.reports_dir/.artifacts_dir;
- load_table_simple aceita (path, fmt=None, **read_opts) e também (path, fmt, read_opts_dict);
- n1_quality_typing agora retorna (df, meta) para compatibilidade; a variante nova fica em n1_quality_typing_dict;
- normalize_categories aceita cfg= (modo avançado) e também parâmetros simples (case/trim/etc.).
- TableStore suporta __init__(initial=..., current=...) e mantém métodos get/use/list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import contextlib
import datetime as dt
import json
import logging
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore
    MinMaxScaler = None  # type: ignore

__all__ = [
    # raiz/config/manifest
    "ensure_project_root", "load_config",
    "load_manifest", "save_manifest", "update_manifest",
    "record_step", "with_step",
    # I/O de artefatos e relatórios
    "save_artifact", "load_artifact",
    "save_report_df", "save_text",
    # paths
    "N1Paths", "resolve_n1_paths", "path_of",
    # I/O tabelas
    "list_directory_files", "infer_format_from_suffix", "load_csv", "load_table_simple", "save_table",
    "suggest_source_path",
    # limpeza e tipagem
    "strip_whitespace", "infer_numeric_like",
    "n1_quality_typing", "n1_quality_typing_dict",
    # missing/duplicatas/outliers
    "simple_impute_with_flags", "deduplicate_rows",
    "detect_outliers_iqr", "detect_outliers_zscore",
    # categóricas/encoding/scaling
    "normalize_categories", "encode_categories", "encode_categories_safe",
    "scale_numeric", "scale_numeric_safe",
    # datas
    "detect_date_candidates", "parse_dates_with_report", "expand_date_features", "build_calendar_from",
    # texto
    "extract_text_features",
    # target e pipeline compacto
    "build_target", "ensure_target_from_config",
    "apply_encoding_and_scaling",
    # util de catálogo
    "TableStore",
    # visões rápidas e merge
    "basic_overview", "missing_report", "merge_chain",
    # relatórios humanos
    "generate_human_report_md", "md_to_pdf",
    # conveniências
    "set_random_seed", "set_display",
    # versão
    "UTILS_DATA_VERSION",
]

UTILS_DATA_VERSION = "1.2.2-merged"

logger = logging.getLogger("utils_data")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Raiz do projeto, Config e Manifest
# -----------------------------------------------------------------------------
def _find_up(relative_path: str, start: Optional[Path] = None) -> Optional[Path]:
    start = start or Path.cwd()
    rel = Path(relative_path)
    for base in (start, *start.parents):
        cand = base / rel
        if cand.exists():
            return cand
    return None

def ensure_project_root() -> Path:
    cfg = _find_up("config/defaults.json")
    if cfg is None:
        raise FileNotFoundError("config/defaults.json não encontrado ao subir a árvore de diretórios.")
    root = cfg.parent.parent
    logger.info(f"PROJECT_ROOT: {root}")
    utils_dir = root / "utils"
    if utils_dir.exists() and str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
        logger.info(f"sys.path ok. utils: {utils_dir}")
    return root

def load_config(base_abs: Optional[Path] = None, local_abs: Optional[Path] = None) -> Dict[str, Any]:
    root = ensure_project_root()
    base = base_abs or (root / "config" / "defaults.json")
    local = local_abs or (root / "config" / "local.json")
    with base.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if local.exists():
        with local.open("r", encoding="utf-8") as f:
            local_cfg = json.load(f)
        cfg = _deep_merge(cfg, local_cfg)
    return cfg

def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out

def _manifest_path(root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    return root / "reports" / "manifest.json"

def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    p = _manifest_path(root)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": []}

def save_manifest(manifest: Mapping[str, Any], root: Optional[Path] = None) -> None:
    p = _manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def update_manifest(update: Mapping[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    m = load_manifest(root)
    m = _deep_merge(m, update)
    save_manifest(m, root)
    return m

def record_step(name: str, details: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None) -> None:
    m = load_manifest(root)
    entry = {
        "step": name,
        "details": dict(details or {}),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    m.setdefault("runs", []).append(entry)
    save_manifest(m, root)
    logger.info(f"[manifest] step='{name}' registrado.")

@contextlib.contextmanager
def with_step(name: str, details: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None):
    record_step(f"{name}:start", details, root)
    try:
        yield
        record_step(f"{name}:end", details, root)
    except Exception as e:
        record_step(f"{name}:error", {"error": str(e)}, root)
        raise

def save_artifact(obj: Any, name: str, root: Optional[Path] = None) -> Path:
    if joblib is None:
        raise RuntimeError("joblib não está disponível. Instale com `pip install joblib`.")
    root = root or ensure_project_root()
    path = root / "artifacts" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)  # type: ignore
    record_step("save_artifact", {"name": name, "path": str(path)}, root)
    return path

def load_artifact(name: str, root: Optional[Path] = None) -> Any:
    if joblib is None:
        raise RuntimeError("joblib não está disponível. Instale com `pip install joblib`.")
    root = root or ensure_project_root()
    path = root / "artifacts" / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact não encontrado: {path}")
    return joblib.load(path)  # type: ignore

def save_report_df(df: pd.DataFrame, rel_path: Union[str, Path], root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    path = root / "reports" / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    record_step("save_report_df", {"path": str(path), "rows": len(df)})
    logger.info(f"[report] salvo: {path} ({len(df)} linhas)")
    return path

def save_text(text: str, rel_path: Union[str, Path], root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    path = root / "reports" / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    record_step("save_text", {"path": str(path), "size": len(text)})
    logger.info(f"[report] texto salvo: {path} ({len(text)} chars)")
    return path

# -----------------------------------------------------------------------------
# Paths N1 (com aliases de compatibilidade)
# -----------------------------------------------------------------------------
@dataclass
class N1Paths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    reports: Path
    artifacts: Path

# aliases legados
setattr(N1Paths, "raw_dir", property(lambda self: self.data_raw))
setattr(N1Paths, "interim_dir", property(lambda self: self.data_interim))
setattr(N1Paths, "processed_dir", property(lambda self: self.data_processed))
setattr(N1Paths, "reports_dir", property(lambda self: self.reports))
setattr(N1Paths, "artifacts_dir", property(lambda self: self.artifacts))

def _resolve_n1_paths_core(root: Optional[Path] = None) -> N1Paths:
    root = root or ensure_project_root()
    return N1Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        reports=root / "reports",
        artifacts=root / "artifacts",
    )

def resolve_n1_paths(*args) -> N1Paths:
    """Compatível com duas formas:
    - resolve_n1_paths() ou resolve_n1_paths(root)
    - resolve_n1_paths(config, root)  # notebooks antigos
    """
    if len(args) == 0:
        return _resolve_n1_paths_core(None)
    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, (str, Path)):
            return _resolve_n1_paths_core(Path(a0))
        else:
            return _resolve_n1_paths_core(None)
    if len(args) >= 2:
        root = args[1]
        return _resolve_n1_paths_core(Path(root))
    return _resolve_n1_paths_core(None)

def path_of(*parts: str, root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    return root.joinpath(*parts)

# -----------------------------------------------------------------------------
# I/O e inspeção
# -----------------------------------------------------------------------------
def list_directory_files(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
                "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return pd.DataFrame(rows)

def suggest_source_path(directory: Union[str, Path], pattern: str = "*.csv", max_rows: int = 50) -> pd.DataFrame:
    directory = Path(directory)
    rows = []
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    df = pd.DataFrame(rows)
    if len(df) > max_rows:
        df = df.head(max_rows)
    logger.info(f"[suggest_source_path] {len(rows)} arquivo(s) encontrados; exibindo {len(df)}.")
    return df

def infer_format_from_suffix(path: Union[str, Path]) -> str:
    s = Path(path).suffix.lower()
    if s == ".csv":
        return "csv"
    if s in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Não sei inferir formato a partir de '{s}'. Use csv/parquet.")

def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def load_table_simple(path: Union[str, Path], fmt: Optional[Union[str, dict]] = None, *args, **kwargs) -> pd.DataFrame:
    """Compatível com:
       - load_table_simple(path, fmt=None, **read_opts)
       - load_table_simple(path, fmt, read_opts_dict)
    """
    # caso antigo: terceiro arg é read_opts_dict posicional
    if args and isinstance(args[0], dict) and not kwargs:
        kwargs = args[0]
    # caso em que segundo arg veio como dict por engano
    if isinstance(fmt, dict):
        kwargs = {**fmt, **kwargs}
        fmt = None
    fmt = fmt or infer_format_from_suffix(path)
    if fmt == "csv":
        return pd.read_csv(path, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Formato não suportado: {fmt}")

def save_table(df: pd.DataFrame, path: Union[str, Path], fmt: Optional[str] = None, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or infer_format_from_suffix(path)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")
    logger.info(f"[save_table] {path} ({len(df)} linhas)")
    return path

# -----------------------------------------------------------------------------
# Visões rápidas e merge
# -----------------------------------------------------------------------------
def basic_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
    }

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = (df.isna().mean() * 100).round(2)
    rep = (
        pd.DataFrame({
            "column": df.columns,
            "missing_count": miss_cnt.values,
            "missing_pct": miss_pct.values,
        })
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    return rep

def merge_chain(base: pd.DataFrame, tables: dict, steps: list) -> pd.DataFrame:
    df = base.copy()
    for step in steps:
        src_name = step.get("from")
        if src_name not in tables:
            raise KeyError(f"[merge_chain] Tabela '{src_name}' não encontrada. Disponíveis: {list(tables.keys())}")
        right = tables[src_name]
        how = step.get("how", "left")
        suffixes = tuple(step.get("suffixes", ("", "_r")))
        kwargs = {"how": how, "suffixes": suffixes}
        if "on" in step:
            kwargs["on"] = step["on"]
        if "left_on" in step or "right_on" in step:
            kwargs["left_on"] = step.get("left_on")
            kwargs["right_on"] = step.get("right_on")
        if "validate" in step:
            kwargs["validate"] = step["validate"]
        df = pd.merge(df, right, **kwargs)
        for c in step.get("drop_cols", []) or []:
            if c in df.columns:
                df = df.drop(columns=c)
    return df

# -----------------------------------------------------------------------------
# Limpeza e tipagem
# -----------------------------------------------------------------------------
def strip_whitespace(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = list(cols) if cols is not None else list(df.columns)
    for c in cols:
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df

def infer_numeric_like(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
                       decimal: str = ".", thousands: Optional[str] = None,
                       report_path: Optional[Union[str, Path]] = "cast_report.csv",
                       root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    cols = list(cols) if cols is not None else [c for c in df.columns if df[c].dtype == "object"]
    rep_rows = []
    for c in cols:
        s = df[c].astype(str)
        before_nulls = s.isna().sum() if not isinstance(df[c].dtype, np.dtype) else df[c].isna().sum()
        s2 = s
        if thousands:
            s2 = s2.str.replace(thousands, "", regex=False)
        if decimal != ".":
            s2 = s2.str.replace(decimal, ".", regex=False)
        new = pd.to_numeric(s2, errors="coerce")
        introduced_nans = new.isna().sum() - before_nulls
        df[c] = new if not new.isna().all() else df[c]
        rep_rows.append({
            "column": c,
            "converted_non_null": int(new.notna().sum()),
            "introduced_nans": int(introduced_nans),
            "dtype_after": str(df[c].dtype),
        })
    report = pd.DataFrame(rep_rows).sort_values("converted_non_null", ascending=False)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

def n1_quality_typing_dict(df: pd.DataFrame, config: Mapping[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """Nova API: retorna dict com 'df', 'steps' e 'cast_report'."""
    out: Dict[str, Any] = {"steps": []}
    typing_cfg = config.get("typing", {})
    with with_step("n1_quality_typing", {"config": typing_cfg}, root):
        if typing_cfg.get("strip_whitespace", True):
            df = strip_whitespace(df)
            out["steps"].append("strip_whitespace")
        if typing_cfg.get("infer_numeric_like", True):
            df, rep = infer_numeric_like(
                df,
                decimal=typing_cfg.get("decimal", "."),
                thousands=typing_cfg.get("thousands"),
                report_path=typing_cfg.get("cast_report_path", "cast_report.csv"),
                root=root,
            )
            out["cast_report"] = rep
            out["steps"].append("infer_numeric_like")
    out["df"] = df
    return out

def n1_quality_typing(df: pd.DataFrame, config: Mapping[str, Any], root: Optional[Path] = None):
    """Compat: retorna (df, meta_dict)."""
    out = n1_quality_typing_dict(df, config, root=root)
    return out["df"], out

# -----------------------------------------------------------------------------
# Missing, duplicatas e outliers
# -----------------------------------------------------------------------------
def simple_impute_with_flags(df: pd.DataFrame, strategy: str = "median",
                             numeric_cols: Optional[Sequence[str]] = None,
                             categorical_cols: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {"strategy": strategy, "imputed": []}
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for c in numeric_cols:
        flag = f"{c}_was_missing"
        wasna = df[c].isna()
        df[flag] = wasna.astype(int)
        fill = df[c].median() if strategy == "median" else (df[c].mean() if strategy == "mean" else 0)
        df[c] = df[c].fillna(fill)
        meta["imputed"].append({"col": c, "fill": fill})

    for c in categorical_cols:
        flag = f"{c}_was_missing"
        wasna = df[c].isna()
        df[flag] = wasna.astype(int)
        df[c] = df[c].fillna("__MISSING__")
        meta["imputed"].append({"col": c, "fill": "__MISSING__"})

    return df, meta

# --- Helpers de robustez para N1 ------------------------------------------------
def coerce_df(obj) -> pd.DataFrame:
    """Garante um DataFrame. Se vier (df, meta), retorna o primeiro elemento."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, tuple) and len(obj) > 0 and isinstance(obj[0], pd.DataFrame):
        return obj[0]
    raise TypeError(f"[coerce_df] Esperado DataFrame/tupla(df,*), recebi: {type(obj)}")

def handle_missing_step(df: pd.DataFrame,
                        config: Mapping[str, Any],
                        save_reports: bool = True,
                        prefer: str = "auto") -> Dict[str, Any]:
    """
    Executa a etapa de 'faltantes' ponta-a-ponta:
      - Gera relatório 'antes' (reports/missing/before.csv)
      - Aplica estratégia (simple | knn | iterative). 'auto' lê do config com fallbacks
      - Gera relatório 'depois' (reports/missing/after.csv)
    Retorna dict: {'df','before','after','strategy','imputed_cols'}
    """
    df = coerce_df(df)

    # Lê config em dois formatos possíveis
    missing_cfg = dict(config.get("missing", {}))
    handle = bool(config.get("handle_missing", missing_cfg.get("enabled", True)))
    strategy = (config.get("missing_strategy",
                           missing_cfg.get("strategy", "simple")) or "simple").lower()

    # Parâmetros extras (com defaults)
    knn_k = int(missing_cfg.get("knn_k", 5))
    it_max = int(missing_cfg.get("iterative_max_iter", 10))
    it_seed = int(missing_cfg.get("iterative_random_state", 42))

    # Onde salvar relatórios
    before_rel = "missing/before.csv"
    after_rel  = "missing/after.csv"

    # Relatório antes
    rep_before = missing_report(df)
    if save_reports:
        save_report_df(rep_before, before_rel)

    out: Dict[str, Any] = {
        "before": rep_before,
        "strategy": strategy,
        "imputed_cols": []
    }

    if not handle:
        out["df"] = df
        out["after"] = rep_before.copy()
        return out

    # Estratégias
    def _simple(df_in: pd.DataFrame):
        df_out, meta = simple_impute_with_flags(df_in)
        cols = [m["col"] for m in meta.get("imputed", [])]
        return df_out, cols, "simple"

    def _knn(df_in: pd.DataFrame):
        try:
            from sklearn.impute import KNNImputer  # type: ignore
        except Exception:
            # fallback
            return _simple(df_in)
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)
        df_out = df_in.copy()
        for c in num_cols:
            df_out[f"{c}_was_missing"] = df_out[c].isna().astype(int)
        imputer = KNNImputer(n_neighbors=knn_k)
        df_out[num_cols] = imputer.fit_transform(df_out[num_cols])
        # cols imputadas: as que tinham NaN
        cols = [c for c in num_cols if df_in[c].isna().any()]
        return df_out, cols, "knn"

    def _iterative(df_in: pd.DataFrame):
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer  # type: ignore
        except Exception:
            return _simple(df_in)
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)
        df_out = df_in.copy()
        for c in num_cols:
            df_out[f"{c}_was_missing"] = df_out[c].isna().astype(int)
        imp = IterativeImputer(max_iter=it_max, random_state=it_seed, sample_posterior=False)
        df_out[num_cols] = imp.fit_transform(df_out[num_cols])
        cols = [c for c in num_cols if df_in[c].isna().any()]
        return df_out, cols, "iterative"

    # Seleção da estratégia com fallback
    chosen = strategy if prefer == "auto" else prefer
    try_chain = {
        "simple":    (_simple,   ["simple"]),
        "knn":       (_knn,      ["knn", "simple"]),
        "iterative": (_iterative,["iterative", "simple"]),
        "mice":      (_iterative,["iterative", "simple"]),
        "auto":      (None,      [strategy, "simple"]),
    }

    # resolve ordem de tentativa
    order = try_chain.get("auto" if prefer == "auto" else chosen, (None, ["simple"]))[1]

    df_work = df
    used = "simple"
    imputed_cols: List[str] = []

    for opt in order:
        if opt == "simple":
            df_work, imputed_cols, used = _simple(df_work)
            break
        elif opt == "knn":
            df_work, imputed_cols, used = _knn(df_work)
            if used == "knn":
                break
        elif opt in {"iterative", "mice"}:
            df_work, imputed_cols, used = _iterative(df_work)
            if used == "iterative":
                break

    out["df"] = df_work
    out["strategy"] = used
    out["imputed_cols"] = imputed_cols

    # Relatório depois
    rep_after = missing_report(df_work)
    out["after"] = rep_after
    if save_reports:
        save_report_df(rep_after, after_rel)

    return out


def deduplicate_rows(df: pd.DataFrame, subset: Optional[Sequence[str]] = None, keep: str = "first") -> Tuple[pd.DataFrame, pd.DataFrame]:
    before = len(df)
    dup_mask = df.duplicated(subset=subset, keep=keep)
    log = df.loc[dup_mask].copy()
    out = df.loc[~dup_mask].copy()
    logger.info(f"[deduplicate_rows] removidas {len(log)} duplicatas (de {before}).")
    return out, log

def detect_outliers_iqr(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, k: float = 1.5) -> pd.DataFrame:
    cols = list(cols) if cols is not None else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask[c] = (df[c] < lo) | (df[c] > hi)
    return mask

def detect_outliers_zscore(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, z: float = 3.0) -> pd.DataFrame:
    cols = list(cols) if cols is not None else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        s = df[c]
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0:
            mask[c] = False
        else:
            zscores = (s - mu) / sigma
            mask[c] = zscores.abs() > z
    return mask

# -----------------------------------------------------------------------------
# Categóricas, encoding e scaling
# -----------------------------------------------------------------------------
def _normalize_str(x: str) -> str:
    x = unicodedata.normalize("NFKD", x)
    x = "".join([ch for ch in x if not unicodedata.combining(ch)])
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_categories(df: pd.DataFrame,
                         cols: Optional[Sequence[str]] = None,
                         case: str = "lower",
                         trim: bool = True,
                         strip_accents: bool = True,
                         cfg: Optional[Mapping[str, Any]] = None,
                         report_path: Optional[Union[str, Path]] = None,
                         root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Modo compat + avançado.
       - Sem cfg: usa (case/trim/strip_accents) simples.
       - Com cfg: espera chaves como exclude, collapse_ws, null_values, global_map, per_column_map, cast_to_category.
       Retorna (df, report) e opcionalmente salva o CSV do report se report_path for informado.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]

    # defaults
    collapse_ws = True
    null_values = set()
    global_map = {}
    per_column_map = {}
    cast_to_category = False
    exclude = set()

    if cfg:
        case = cfg.get("case", case)
        trim = cfg.get("trim", trim)
        strip_accents = cfg.get("strip_accents", strip_accents)
        collapse_ws = cfg.get("collapse_ws", True)
        null_values = set(map(str, cfg.get("null_values", [])))
        global_map = {str(k): v for k, v in (cfg.get("global_map") or {}).items()}
        per_column_map = {k: {str(kk): vv for kk, vv in v.items()} for k, v in (cfg.get("per_column_map") or {}).items()}
        cast_to_category = bool(cfg.get("cast_to_category", False))
        exclude = set(cfg.get("exclude", []))
        cols = [c for c in cols if c not in exclude]

    changes_rows: List[Dict[str, Any]] = []

    for c in cols:
        s_orig = df[c].astype(str)
        s = s_orig
        if trim:
            s = s.str.strip()
        if collapse_ws:
            s = s.str.replace(r"\s+", " ", regex=True)
        if strip_accents:
            s = s.map(lambda v: _normalize_str(v))
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        if global_map:
            s = s.map(lambda v: global_map.get(v, v))
        if c in per_column_map:
            cmap = per_column_map[c]
            s = s.map(lambda v: cmap.get(v, v))
        if null_values:
            s = s.map(lambda v: (np.nan if str(v) in null_values else v))

        changed = (s_orig.astype(str) != s.astype(str)).sum()
        changes_rows.append({"column": c, "changed": int(changed)})
        df[c] = s
        if cast_to_category:
            df[c] = df[c].astype("category")

    report = pd.DataFrame(changes_rows).sort_values("changed", ascending=False).reset_index(drop=True)
    if report_path is not None:
        if isinstance(report_path, Path):
            out_path = report_path
        else:
            root = root or ensure_project_root()
            out_path = (root / "reports" / str(report_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False, encoding="utf-8")
        record_step("save_report_df", {"path": str(out_path), "rows": int(len(report))}, root)
    return df, report

def _top_k_categories(s: pd.Series, k: Optional[int]) -> Tuple[pd.Series, List[str]]:
    if not k or k <= 0:
        return s, []
    top = list(s.value_counts(dropna=False).head(k).index)
    s2 = s.where(s.isin(top), other="__OTHER__")
    return s2, top

def encode_categories(df: pd.DataFrame,
                      cols: Optional[Sequence[str]] = None,
                      drop_first: bool = False,
                      high_cardinality_threshold: int = 20,
                      top_k: Optional[int] = None,
                      other_label: str = "__OTHER__") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {"encoded": [], "high_cardinality": []}
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
    for c in cols:
        s = df[c].astype(str)
        card = s.nunique(dropna=True)
        use_top_k = (top_k is not None) and (card > high_cardinality_threshold)
        if use_top_k:
            s2, kept = _top_k_categories(s, top_k)
            meta["high_cardinality"].append({"col": c, "cardinality": int(card), "kept_top_k": int(len(kept))})
            s = s2.replace({"__OTHER__": other_label})
        else:
            kept = list(s.dropna().unique())
        dummies = pd.get_dummies(s, prefix=c, drop_first=drop_first, dummy_na=False)
        df = df.drop(columns=[c]).join(dummies)
        meta["encoded"].append({"col": c, "created_cols": list(dummies.columns), "kept": kept})
    return df, meta

def encode_categories_safe(df: pd.DataFrame,
                           exclude_cols: Optional[Sequence[str]] = None,
                           **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    exclude = set(exclude_cols or [])
    cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])) and c not in exclude]
    return encode_categories(df, cols=cols, **kwargs)

def scale_numeric(df: pd.DataFrame, method: str = "standard",
                  cols: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if StandardScaler is None or MinMaxScaler is None:
        raise RuntimeError("scikit-learn não está disponível. Instale com `pip install scikit-learn`.")
    df = df.copy()
    meta: Dict[str, Any] = {"method": method, "scaled": []}
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    scaler = StandardScaler() if method == "standard" else MinMaxScaler() if method == "minmax" else None
    if scaler is None:
        raise ValueError("method deve ser 'standard' ou 'minmax'.")

    df[cols] = scaler.fit_transform(df[cols])
    meta["scaled"] = list(cols)
    meta["scaler"] = scaler
    return df, meta

def scale_numeric_safe(df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    exclude = set(exclude_cols or [])
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return scale_numeric(df, cols=cols, **kwargs)

# -----------------------------------------------------------------------------
# Datas
# -----------------------------------------------------------------------------
_DEFAULT_DATE_REGEX = [r"(?:^|_)(date|data|dt)(?:$|_)", r"(?:_at$)", r"(?:_date$)"]

def detect_date_candidates(df: pd.DataFrame, regex_list: Optional[Sequence[str]] = None) -> List[str]:
    regex_list = list(regex_list) if regex_list else _DEFAULT_DATE_REGEX
    out: List[str] = []
    for c in df.columns:
        name = c.lower()
        if any(re.search(rx, name) for rx in regex_list):
            out.append(c)
    return out

def parse_dates_with_report(df: pd.DataFrame,
                            cols: Optional[Sequence[str]] = None,
                            dayfirst: bool = False,
                            utc: bool = False,
                            errors: str = "coerce",
                            min_ratio: float = 0.6,
                            report_path: Optional[Union[str, Path]] = "date_parse_report.csv",
                            max_fail_samples: int = 10,
                            root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if cols is None:
        cols = detect_date_candidates(df)

    rep_rows = []
    for c in cols:
        raw = df[c]
        parsed = pd.to_datetime(raw, dayfirst=dayfirst, utc=utc, errors=errors)
        success = parsed.notna().mean() if len(parsed) else 0.0
        fail_samples = raw[parsed.isna()].astype(str).head(max_fail_samples).tolist()
        rep_rows.append({
            "column": c,
            "success_ratio": float(round(success, 4)),
            "applied": bool(success >= min_ratio),
            "fail_samples": "; ".join(fail_samples),
        })
        if success >= min_ratio:
            df[c] = parsed

    report = pd.DataFrame(rep_rows).sort_values("success_ratio", ascending=False)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

def expand_date_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s = df[c].dt
        df[f"{c}_year"] = s.year
        df[f"{c}_month"] = s.month
        df[f"{c}_day"] = s.day
        df[f"{c}_dow"] = s.dayofweek
        df[f"{c}_week"] = s.isocalendar().week.astype(int)
        df[f"{c}_quarter"] = s.quarter
    return df

def build_calendar_from(df: pd.DataFrame, col: str, freq: str = "D") -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors="coerce")
    start, end = s.min(), s.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Coluna {col} não possui datas válidas.")
    idx = pd.date_range(start=start, end=end, freq=freq)
    cal = pd.DataFrame({"date": idx})
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["day"] = cal["date"].dt.day
    cal["dow"] = cal["date"].dt.dayofweek
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["quarter"] = cal["date"].dt.quarter
    return cal

# -----------------------------------------------------------------------------
# Texto
# -----------------------------------------------------------------------------
def extract_text_features(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
                          report_path: Optional[Union[str, Path]] = "text_features/summary.csv",
                          root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    rep = []
    for c in cols:
        s = df[c].astype(str)
        df[f"{c}_len"] = s.str.len()
        df[f"{c}_word_count"] = s.str.split().map(len)
        rep.append({"column": c, "len_col": f"{c}_len", "word_count_col": f"{c}_word_count"})
    report = pd.DataFrame(rep)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

# -----------------------------------------------------------------------------
# Target e pipeline compacto
# -----------------------------------------------------------------------------
def build_target(df: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    tcfg = config.get("target", {})
    name = tcfg.get("name", "target")
    rule = tcfg.get("rule", {})
    col, op, value = rule.get("col"), rule.get("op"), rule.get("value")
    if col is None or op is None:
        raise ValueError("Config de target inválida: especifique 'col' e 'op'.")
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    if op not in ops:
        raise ValueError(f"Operador não suportado: {op}")
    df[name] = ops[op](df[col], value).astype(int)
    meta = {"name": name, "rule": rule}
    return df, meta

def ensure_target_from_config(df: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Optional[str]]:
    tcfg = config.get("target", {})
    name = tcfg.get("name", "target")
    if name in df.columns:
        return df, name
    df2, meta = build_target(df, config)
    return df2, meta.get("name")  # type: ignore

def apply_encoding_and_scaling(df: pd.DataFrame,
                               config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {}

    enc_cfg = config.get("encoding", {})
    df, enc_meta = encode_categories_safe(
        df,
        exclude_cols=enc_cfg.get("exclude_cols"),
        drop_first=enc_cfg.get("drop_first", False),
        high_cardinality_threshold=enc_cfg.get("high_cardinality_threshold", 20),
        top_k=enc_cfg.get("top_k", None),
    )
    meta["encoding"] = enc_meta

    sc_cfg = config.get("scaling", {})
    df, sc_meta = scale_numeric_safe(
        df,
        exclude_cols=sc_cfg.get("exclude_cols"),
        method=sc_cfg.get("method", "standard"),
    )
    meta["scaling"] = sc_meta

    return df, meta

# -----------------------------------------------------------------------------
# Catálogo simples de DataFrames
# -----------------------------------------------------------------------------
class TableStore:
    """
    Catálogo simples de DataFrames nomeados.
    Compatível com __init__(initial=..., current=...) e APIs legadas.
    """
    def __init__(self, initial: Optional[dict] = None, current: Optional[str] = None):
        self._tables: Dict[str, pd.DataFrame] = {}
        self.current: Optional[str] = None
        if initial:
            for name, df in initial.items():
                self.put(name, df)
        if current is not None:
            if current not in self._tables:
                raise KeyError(f"'{current}' não encontrada em initial: {list(self._tables.keys())}")
            self.current = current
        elif self._tables:
            self.current = sorted(self._tables.keys())[0]

    # aliases legados
    def add(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        self.put(name, df, set_current=set_current)

    def put(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        self._tables[name] = df.copy()
        if set_current or self.current is None:
            self.current = name

    def get(self, name: Optional[str] = None) -> pd.DataFrame:
        key = name or self.current
        if key is None or key not in self._tables:
            raise KeyError(f"Tabela '{key}' não encontrada. Disponíveis: {list(self._tables.keys())}")
        return self._tables[key].copy()

    def use(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"Tabela '{name}' não encontrada. Disponíveis: {list(self._tables.keys())}")
        self.current = name
        return self._tables[name].copy()

    def names(self) -> List[str]:
        return sorted(self._tables.keys())

    def list(self) -> pd.DataFrame:
        rows = []
        for name, df in self._tables.items():
            rows.append({
                "name": name,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
                "current": bool(name == self.current),
            })
        return pd.DataFrame(rows).sort_values(["current", "name"], ascending=[False, True]).reset_index(drop=True)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.get(name)

    def __setitem__(self, name: str, df: pd.DataFrame) -> None:
        self.put(name, df)

    def __len__(self) -> int:
        return len(self._tables)

# -----------------------------------------------------------------------------
# Helpers e relatórios
# -----------------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"[set_random_seed] seed={seed}")

def set_display(max_rows: int = 200, max_cols: int = 120) -> None:
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_cols)
    logger.info(f"[set_display] rows={max_rows} cols={max_cols}")

def generate_human_report_md(df: pd.DataFrame, title: str = "Relatório de Dados") -> str:
    lines = [f"# {title}", "", f"- Linhas: {len(df)}", f"- Colunas: {len(df.columns)}", ""]
    lines.append("## Dtypes")
    for c, t in df.dtypes.astype(str).to_dict().items():
        lines.append(f"- **{c}**: `{t}`")
    lines.append("")
    lines.append("## Missing (%)")
    miss = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    for c, v in miss.items():
        lines.append(f"- {c}: {v}%")
    return "\n".join(lines)

def md_to_pdf(md_text: str, out_path: Union[str, Path], engine: str = "weasyprint") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if engine == "weasyprint":
        try:
            import weasyprint  # type: ignore
        except Exception as e:
            raise RuntimeError("weasyprint não está instalado. Use `pip install weasyprint` ou engine='pandoc'.") from e
        html = f"<pre>{md_text}</pre>"
        weasyprint.HTML(string=html).write_pdf(str(out_path))
        return out_path
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc não encontrado no PATH. Instale o binário ou use engine='weasyprint'.")
    tmp_md = out_path.with_suffix(".tmp.md")
    tmp_md.write_text(md_text, encoding="utf-8")
    cp = subprocess.run(["pandoc", str(tmp_md), "-o", str(out_path)], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Falha no pandoc: {cp.stderr}")
    tmp_md.unlink(missing_ok=True)
    return out_path
