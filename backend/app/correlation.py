from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class EffortCorrelationRequest(BaseModel):
    words: List[str]
    effort: List[float]
    word_column: Optional[str] = None
    group: Optional[Dict[str, str]] = None


class EffortCorrelationResponse(BaseModel):
    matched_count: int
    total_words: int
    metric_correlations: Dict[str, float]
    used_columns: List[str]
    token_column: Optional[str] = None


_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dataset() -> pd.DataFrame:
    global _DATA_CACHE
    key = "onestop_fixations"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    data_path = _repo_root() / "data" / "fixations_Paragraph_ordinary 2.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    _DATA_CACHE[key] = df
    return df


_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize_word(w: str) -> str:
    if not isinstance(w, str):
        w = str(w)
    s = w.strip().lower()
    s = _punct_re.sub("", s)
    return s


def _object_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == object]


def _preferred_word_column(df: pd.DataFrame) -> Optional[str]:
    # Always prefer the 'paragraph' column (case-insensitive) if present
    for c in df.columns:
        if str(c).strip().lower() == "paragraph":
            return c
    return None


def _guess_word_column(df: pd.DataFrame) -> str:
    # Hard preference: 'paragraph' column (case-insensitive)
    pref = _preferred_word_column(df)
    if pref is not None:
        return pref

    # Prefer likely word-bearing columns by name first
    name_priority = [
        "Word", "word", "WORD", "Stimulus", "stimulus", "Token", "token", "FixWord", "fixword",
    ]
    for c in name_priority:
        if c in df.columns:
            return c

    # Otherwise, score object columns by lexical characteristics
    obj_cols = _object_columns(df)
    if not obj_cols:
        return df.columns[0]

    def score_col(col: str) -> float:
        series = df[col].astype(str).head(1000)
        total = 0
        letters = 0
        avg_len = 0.0
        code_like = 0
        for s in series:
            total += 1
            norm = _normalize_word(s)
            avg_len += len(norm)
            if re.search(r"[a-zA-Z]", norm):
                letters += 1
            # penalize ID-like strings (e.g., l59_485, w12-34)
            if re.search(r"^[A-Za-z]+\d+(?:[_-]\d+)?$", s):
                code_like += 1
        if total == 0:
            return -1.0
        avg_len /= max(1, total)
        frac_letters = letters / total
        frac_code = code_like / total
        # Prefer columns with many alphabetic tokens and reasonable length; penalize code-like
        return frac_letters + 0.1 * max(0.0, min(1.0, (avg_len - 3) / 5)) - 0.7 * frac_code

    scores = {c: score_col(c) for c in obj_cols}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best


def _get_group_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    primary_candidates = [
        "TextID", "text_id", "Article", "article", "ItemID", "item", "Story", "story", "Doc", "doc",
    ]
    secondary_candidates = ["Paragraph", "paragraph", "Para", "para"]
    group_cols: List[str] = []
    for c in primary_candidates:
        if c in df.columns:
            group_cols.append(c)
            break
    for c in secondary_candidates:
        if c in df.columns:
            group_cols.append(c)
            break
    # If nothing found, no grouping (single group)
    return group_cols, secondary_candidates


def _get_order_columns(df: pd.DataFrame) -> List[str]:
    order_sets = [
        ["SentenceID", "WordIndex"],
        ["sentence_id", "word_index"],
        ["Line", "WordIndex"],
        ["line", "word_index"],
        ["Index"],
        ["WordIndex"],
    ]
    for cols in order_sets:
        if all(c in df.columns for c in cols):
            return cols
    return []


def _get_word_position_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["WordIndex", "word_index", "Index"]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _candidate_position_columns(df: pd.DataFrame, para_len: Optional[int] = None) -> List[str]:
    """Heuristically propose numeric columns that likely encode word positions in a paragraph.
    Prefer names containing word/index/position/interest_area/ia.
    Optionally, score closeness of unique value count to para_len.
    """
    name_re = re.compile(r"(word(_)?(index|id|number|no))|(position|pos)|(interest[_ ]?area|ia)(_|-)?(id|index|nr|no|num)?",
                         flags=re.IGNORECASE)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cands = [c for c in numeric_cols if name_re.search(str(c))]
    # If none match by name, fall back to all numeric columns
    if not cands:
        cands = numeric_cols

    def score(c: str) -> float:
        s = df[c].dropna()
        if s.empty:
            return -1.0
        uniq = int(s.nunique())
        # basic plausibility: integer-like, smallish range
        int_like = float((s % 1 == 0).mean()) if pd.api.types.is_float_dtype(s) else 1.0
        # closeness to paragraph length if provided
        closeness = 0.0
        if para_len and para_len > 0:
            closeness = 1.0 - min(1.0, abs(uniq - para_len) / max(para_len, 1))
        # prefer broader coverage
        coverage = uniq / max(len(s), 1)
        return 0.5 * int_like + 0.4 * closeness + 0.1 * coverage

    cands = sorted(set(cands), key=lambda c: score(c), reverse=True)
    return cands


def _split_tokens_from_paragraph(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    # Split on whitespace; normalization is applied during alignment
    return [t for t in re.split(r"\s+", text.strip()) if t]


class ParagraphGroup(BaseModel):
    group: Dict[str, str]
    size: int


class ParagraphListResponse(BaseModel):
    groups: List[ParagraphGroup]
    word_column: str
    word_candidates: List[str]
    grouping_columns: List[str]
    order_columns: List[str]


class ParagraphRequest(BaseModel):
    group: Dict[str, str]
    word_column: Optional[str] = None
    limit: Optional[int] = None


class ParagraphResponse(BaseModel):
    words: List[str]
    text: str
    group: Dict[str, str]
    limit: Optional[int] = None


class MetricsListResponse(BaseModel):
    metrics: List[str]


class AlignedSeriesRequest(BaseModel):
    words: List[str]
    metrics: List[str]
    word_column: Optional[str] = None
    group: Optional[Dict[str, str]] = None


class AlignedSeriesResponse(BaseModel):
    matched_count: int
    total_words: int
    series: Dict[str, List[Optional[float]]]
    used_columns: List[str]
    token_column: Optional[str] = None


@router.get("/onestop/paragraphs", response_model=ParagraphListResponse)
def list_paragraphs(word_column: Optional[str] = None):
    df = _load_dataset()
    # Always prefer 'paragraph' column if present
    word_col = _preferred_word_column(df) or (word_column if (word_column and word_column in df.columns) else _guess_word_column(df))
    group_cols, _ = _get_group_columns(df)
    order_cols = _get_order_columns(df)

    # Ensure the chosen word column is not used as a grouping key
    if word_col in group_cols:
        group_cols = [c for c in group_cols if c != word_col]

    groups: List[ParagraphGroup] = []
    if not group_cols:
        # Single group: entire dataset
        dfo = df
        if order_cols:
            dfo = dfo.sort_values(order_cols)
        words = dfo[word_col].astype(str).tolist()
        groups.append(ParagraphGroup(group={}, size=len(words)))
    else:
        # Group by the detected columns
        for key_vals, dfg in df.groupby(group_cols):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            dfo = dfg
            if order_cols:
                dfo = dfo.sort_values(order_cols)
            words = dfo[word_col].astype(str).tolist()
            group_dict = {col: str(val) for col, val in zip(group_cols, key_vals)}
            groups.append(ParagraphGroup(group=group_dict, size=len(words)))

    # Limit to first 20 groups for brevity
    groups = groups[:20]
    return ParagraphListResponse(
        groups=groups,
        word_column=word_col,
        word_candidates=_object_columns(df),
        grouping_columns=group_cols,
        order_columns=order_cols,
    )


@router.post("/onestop/paragraph", response_model=ParagraphResponse)
def get_paragraph(req: ParagraphRequest):
    df = _load_dataset()
    # Always prefer 'paragraph' column if present
    word_col = _preferred_word_column(df) or (req.word_column if (req.word_column and req.word_column in df.columns) else _guess_word_column(df))
    group_cols, _ = _get_group_columns(df)
    order_cols = _get_order_columns(df)

    # Ensure the chosen word column is not used as a grouping key
    if word_col in group_cols:
        group_cols = [c for c in group_cols if c != word_col]

    sel = df
    if group_cols:
        for col in group_cols:
            if col in req.group:
                sel = sel[sel[col].astype(str) == str(req.group[col])]
    if order_cols:
        sel = sel.sort_values(order_cols)
    words = sel[word_col].astype(str).tolist()
    if req.limit is not None and isinstance(req.limit, int) and req.limit > 0:
        words = words[: req.limit]
    text = " ".join(words)
    return ParagraphResponse(words=words, text=text, group={k: str(v) for k, v in req.group.items()}, limit=req.limit)


@router.get("/onestop/metrics", response_model=MetricsListResponse)
def list_metrics():
    df = _load_dataset()
    mets = _preferred_metric_columns(df)
    return MetricsListResponse(metrics=mets)


@router.post("/onestop/aligned-series", response_model=AlignedSeriesResponse)
def aligned_series(req: AlignedSeriesRequest):
    words = req.words
    df = _load_dataset()

    # Restrict to preferred metrics and those requested that exist
    available = {str(c).strip().lower(): c for c in df.columns}
    metrics_req = []
    for m in req.metrics:
        key = str(m).strip().lower()
        if key in available:
            metrics_req.append(available[key])
    metrics = metrics_req if metrics_req else _preferred_metric_columns(df)
    if not metrics:
        return AlignedSeriesResponse(matched_count=0, total_words=len(words), series={}, used_columns=[], token_column=None)

    # Coerce selected metrics to numeric where possible
    df = df.copy()
    for c in metrics:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass

    # Filter to group if provided
    if req.group:
        group_cols, _ = _get_group_columns(df)
        if group_cols:
            sel = df
            for col in group_cols:
                if col in req.group:
                    sel = sel[sel[col].astype(str) == str(req.group[col])]
            if not sel.empty:
                order_cols = _get_order_columns(df)
                df = sel.sort_values(order_cols) if order_cols else sel

    # Alignment: paragraph-based preferred
    para_col = _preferred_word_column(df)
    pos_col = _get_word_position_column(df)
    used_column: Optional[str] = None
    if para_col is not None:
        para_series = df[para_col].dropna()
        para_text = str(para_series.iloc[0]) if not para_series.empty else ""
        dataset_words = _split_tokens_from_paragraph(para_text)
        idx_map = _align_indices(words, dataset_words)
        used_column = para_col
        if pos_col is None:
            pos_candidates: List[str] = _candidate_position_columns(df, para_len=len(dataset_words))
        else:
            pos_candidates = [pos_col] + [c for c in _candidate_position_columns(df, para_len=len(dataset_words)) if c != pos_col]
    else:
        token_col = req.word_column if (req.word_column and req.word_column in df.columns and str(req.word_column).strip().lower() != "paragraph") else _get_token_column(df)
        dataset_words = df[token_col].astype(str).tolist()
        idx_map = _align_indices(words, dataset_words)
        used_column = token_col
        pos_candidates = []

    # If using token column, try alternative candidates for better coverage
    if used_column is not None and str(used_column).strip().lower() != "paragraph":
        best_map = idx_map
        best_count = sum(1 for j in best_map if j >= 0)
        for cand in _candidate_token_columns(df):
            if cand == used_column:
                continue
            m = _align_indices(words, df[cand].astype(str).tolist())
            cnt = sum(1 for j in m if j >= 0)
            if cnt > best_count:
                best_map, best_count, used_column = m, cnt, cand
        idx_map = best_map

    # Build per-word series
    series: Dict[str, List[Optional[float]]] = {m: [None] * len(words) for m in metrics}
    matched_count = 0
    if para_col is not None and pos_candidates:
        # choose best position column by max non-null sum across metrics
        best_series: Optional[Dict[str, List[Optional[float]]]] = None
        best_score = -1
        for pcol in pos_candidates:
            try:
                grouped = df.groupby(pcol)[metrics].mean(numeric_only=True)
            except Exception:
                continue
            if grouped.empty:
                continue
            try:
                base = int(grouped.index.min())
            except Exception:
                base = 0
            tmp_series = {m: [None] * len(words) for m in metrics}
            tmp_score = 0
            for i, j in enumerate(idx_map):
                if j < 0:
                    continue
                pos = j + base
                if pos in grouped.index:
                    tmp_score += 1
                    row = grouped.loc[pos]
                    for m in metrics:
                        val = row.get(m)
                        tmp_series[m][i] = float(val) if pd.notna(val) else None
            if tmp_score > best_score:
                best_series, best_score = tmp_series, tmp_score
        if best_series is not None:
            series = best_series
            matched_count = best_score
        else:
            matched_count = sum(1 for j in idx_map if j >= 0)
    else:
        # row-based fallback
        for i, j in enumerate(idx_map):
            if j < 0:
                continue
            row = df.iloc[j]
            matched_count += 1
            for m in metrics:
                try:
                    val = row[m]
                    series[m][i] = float(val) if pd.notna(val) else None
                except Exception:
                    series[m][i] = None

    return AlignedSeriesResponse(
        matched_count=matched_count,
        total_words=len(words),
        series=series,
        used_columns=metrics,
        token_column=str(used_column) if used_column is not None else None,
    )


def _get_word_column(df: pd.DataFrame) -> str:
    candidates = [
        "Word", "word", "WORD", "Token", "token", "Stimulus", "stimulus",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        return obj_cols[0]
    return df.columns[0]


def _get_token_column(df: pd.DataFrame) -> str:
    """Pick a per-word token column for alignment, excluding any column named 'paragraph'.
    Prefer known names; otherwise score object columns to avoid code-like IDs.
    """
    preferred = ["Word", "word", "WORD", "Token", "token", "Stimulus", "stimulus", "FixWord", "fixword"]
    for c in preferred:
        if c in df.columns and str(c).strip().lower() != "paragraph":
            return c

    # Score object columns excluding 'paragraph'
    cand = [c for c in df.columns if df[c].dtype == object and str(c).strip().lower() != "paragraph"]
    if not cand:
        return df.columns[0]

    def score_col(col: str) -> float:
        series = df[col].astype(str).head(1000)
        total = 0
        letters = 0
        avg_len = 0.0
        code_like = 0
        for s in series:
            total += 1
            norm = _normalize_word(s)
            avg_len += len(norm)
            if re.search(r"[a-zA-Z]", norm):
                letters += 1
            if re.search(r"^[A-Za-z]+\d+(?:[_-]\d+)?$", s):
                code_like += 1
        if total == 0:
            return -1.0
        avg_len /= max(1, total)
        frac_letters = letters / total
        frac_code = code_like / total
        return frac_letters + 0.1 * max(0.0, min(1.0, (avg_len - 3) / 5)) - 0.7 * frac_code

    scores = {c: score_col(c) for c in cand}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best


def _candidate_token_columns(df: pd.DataFrame) -> List[str]:
    names = ["Word", "word", "WORD", "Token", "token", "Stimulus", "stimulus", "FixWord", "fixword"]
    out: List[str] = []
    for c in names:
        if c in df.columns and str(c).strip().lower() != "paragraph":
            out.append(c)
    # add other object columns excluding 'paragraph'
    for c in df.columns:
        if df[c].dtype == object and str(c).strip().lower() != "paragraph" and c not in out:
            out.append(c)
    # fallback to first column if none
    if not out and len(df.columns) > 0:
        out = [df.columns[0]]
    return out


def _numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    # Exclude obvious non-metric or structural columns
    exclude_exact = {"Sentence", "SentenceID", "TextID", "ItemID", "WordIndex", "Index", "Trial", "line"}
    # Exclude by name pattern: indices, labels, coordinates, raw timestamps
    exclude_re = re.compile(r"(index|label|start|end|time|timestamp|x$|_x$|y$|_y$|nearest|interest[_ ]?area)", re.IGNORECASE)
    out: List[str] = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if exclude_re.search(str(c)):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _preferred_metric_columns(df: pd.DataFrame) -> List[str]:
    """Return only the user-specified metrics that exist in the dataframe.
    Matching is case-insensitive and returns the actual column names from df.
    """
    desired = [
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_INTEREST_AREA_DWELL_TIME",
        "CURRENT_FIX_INTEREST_AREA_FIX_COUNT",
        "CURRENT_FIX_REFIX_INTEREST_AREA",
        "TRIAL_FIXATION_TOTAL",
        "CURRENT_FIX_PUPIL",
        "PARAGRAPH_RT",
        "QUESTION_RT",
        "ANSWER_RT",
        "is_correct",
    ]
    # Build a lookup from lowercase -> actual column
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    present: List[str] = []
    for name in desired:
        key = name.strip().lower()
        if key in lower_map:
            present.append(lower_map[key])
    return present


def _align_indices(words: List[str], dataset_words: List[str]) -> List[int]:
    # More robust windowed greedy aligner with limited skips and jumps
    norm_inp = [_normalize_word(w) for w in words]
    norm_ds = [_normalize_word(d) for d in dataset_words]

    # Build index of dataset positions per token
    positions: Dict[str, List[int]] = {}
    for j, tok in enumerate(norm_ds):
        if not tok:
            continue
        positions.setdefault(tok, []).append(j)

    # Find candidate starting positions using the first non-empty input token
    first_idx = 0
    while first_idx < len(norm_inp) and not norm_inp[first_idx]:
        first_idx += 1
    cand_positions = positions.get(norm_inp[first_idx], []) if first_idx < len(norm_inp) else []
    # limit candidates for performance
    cand_positions = cand_positions[:200]

    def simulate(start_j: int, max_jump: int = 8, max_skip_in: int = 2, budget: int = 2000) -> Tuple[int, List[int]]:
        i = 0
        j = max(0, start_j)
        matched = 0
        idx_map_local: List[int] = [-1] * len(words)
        steps = 0
        while i < len(norm_inp) and j < len(norm_ds) and steps < budget:
            steps += 1
            wi = norm_inp[i]
            if not wi:
                i += 1
                continue
            if wi == norm_ds[j]:
                idx_map_local[i] = j
                matched += 1
                i += 1
                j += 1
                continue
            # try small forward jumps in dataset
            found = False
            for k in range(1, max_jump + 1):
                if j + k >= len(norm_ds):
                    break
                if norm_ds[j + k] == wi:
                    idx_map_local[i] = j + k
                    matched += 1
                    i += 1
                    j = j + k + 1
                    found = True
                    break
            if found:
                continue
            # optionally skip input tokens to resync
            skipped = 0
            while skipped < max_skip_in and i < len(norm_inp) and norm_inp[i] and norm_inp[i] not in positions:
                i += 1
                skipped += 1
            if skipped == 0:
                # advance dataset by one if no better option
                j += 1
        return matched, idx_map_local

    best_score = -1
    best_map: List[int] = [-1] * len(words)
    # If no candidates (rare), fall back to start of dataset
    if not cand_positions:
        cand_positions = [0]
    for start in cand_positions:
        score, mapping = simulate(start)
        if score > best_score:
            best_score = score
            best_map = mapping
    return best_map


@router.post("/effort-correlation", response_model=EffortCorrelationResponse)
def correlate_effort(req: EffortCorrelationRequest):
    words = req.words
    effort = np.asarray(req.effort, dtype=float)
    if len(words) != len(effort):
        n = min(len(words), len(effort))
        words = words[:n]
        effort = effort[:n]

    df = _load_dataset()
    # If a group is provided, filter dataset to that group for better alignment
    if req.group:
        group_cols, _ = _get_group_columns(df)
        if group_cols:
            sel = df
            for col in group_cols:
                if col in req.group:
                    sel = sel[sel[col].astype(str) == str(req.group[col])]
            if not sel.empty:
                # sort by order columns if available for consistent sequence
                order_cols = _get_order_columns(df)
                df = sel.sort_values(order_cols) if order_cols else sel
    # Try paragraph-based alignment if a paragraph column exists in this group
    para_col = _preferred_word_column(df)
    pos_col = _get_word_position_column(df)
    used_column = None
    if para_col is not None:
        para_series = df[para_col].dropna()
        para_text = str(para_series.iloc[0]) if not para_series.empty else ""
        dataset_words = _split_tokens_from_paragraph(para_text)
        idx_map = _align_indices(words, dataset_words)
        used_column = para_col
        # Try multiple candidate position columns for aggregation if the default isn't ideal
        if pos_col is None:
            pos_candidates: List[str] = _candidate_position_columns(df, para_len=len(dataset_words))
        else:
            pos_candidates = [pos_col] + [c for c in _candidate_position_columns(df, para_len=len(dataset_words)) if c != pos_col]
    else:
        # Fallback to token column alignment
        if req.word_column and req.word_column in df.columns and str(req.word_column).strip().lower() != "paragraph":
            token_col = req.word_column
        else:
            token_col = _get_token_column(df)
        dataset_words = df[token_col].astype(str).tolist()
        idx_map = _align_indices(words, dataset_words)
        used_column = token_col

    # If we used token-column alignment, try alternative candidate token columns and keep the best
    if used_column is not None and str(used_column).strip().lower() != "paragraph":
        matched_rows: List[int] = [j for j in idx_map if j >= 0]
        best_match_count = len(matched_rows)
        token_col = str(used_column)
        best_token_col = token_col
        best_map = idx_map
        if best_match_count < max(5, int(0.1 * len(words))):
            for cand in _candidate_token_columns(df):
                if cand == token_col:
                    continue
                dw = df[cand].astype(str).tolist()
                m = _align_indices(words, dw)
                cnt = sum(1 for j in m if j >= 0)
                if cnt > best_match_count:
                    best_match_count = cnt
                    best_token_col = cand
                    best_map = m
        used_column = best_token_col
        idx_map = best_map

    # Build per-position aggregation if we used paragraph alignment with a word position column
    # Prefer the explicitly requested metrics; fall back to heuristic if unavailable
    metrics = _preferred_metric_columns(df)
    if not metrics:
        metrics = _numeric_metric_columns(df)
    else:
        # Work on a numeric-coerced copy for these columns to ensure proper correlation
        df = df.copy()
        for c in metrics:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
    eff_vec: List[float] = []
    y_by_metric: Dict[str, List[float]] = {m: [] for m in metrics}
    matched_positions: List[int] = []
    if para_col is not None and metrics:
        # Try each candidate position column until we get the best coverage
        best_eff: List[float] = []
        best_y_by: Dict[str, List[float]] = {}
        best_count = -1
        for pcol in (pos_candidates if 'pos_candidates' in locals() else []):
            try:
                grouped = df.groupby(pcol)[metrics].mean(numeric_only=True)
            except Exception:
                continue
            if grouped.empty:
                continue
            try:
                base = int(grouped.index.min())
            except Exception:
                base = 0
            tmp_eff: List[float] = []
            tmp_y_by: Dict[str, List[float]] = {m: [] for m in metrics}
            tmp_count = 0
            for i, j in enumerate(idx_map):
                if j < 0:
                    continue
                pos = j + base
                if pos in grouped.index:
                    tmp_count += 1
                    tmp_eff.append(float(effort[i]))
                    row = grouped.loc[pos]
                    for m in metrics:
                        try:
                            tmp_y_by[m].append(float(row[m]))
                        except Exception:
                            tmp_y_by[m].append(np.nan)
            if tmp_count > best_count:
                best_count = tmp_count
                best_eff = tmp_eff
                best_y_by = tmp_y_by
        eff_vec = best_eff
        y_by_metric = best_y_by
        matched_count = len(eff_vec)
    else:
        # Fallback: original row-based matching
        matched_rows = [j for j in idx_map if j >= 0]
        matched_count = len(matched_rows)
        eff_vec = [effort[i] for i, j in enumerate(idx_map) if j >= 0]
    if matched_count == 0:
        return EffortCorrelationResponse(
            matched_count=0,
            total_words=len(words),
            metric_correlations={},
            used_columns=[],
            token_column=str(used_column) if used_column is not None else None,
        )

    corrs: Dict[str, float] = {}
    used_cols: List[str] = []
    eff_arr = np.asarray(eff_vec, dtype=float)
    if para_col is not None and pos_col is not None and metrics and len(eff_arr) > 1:
        for c in metrics:
            y_list = [v for v in y_by_metric.get(c, []) if np.isfinite(v)]
            # Ensure lengths match by trimming
            n = min(len(y_list), len(eff_arr))
            if n >= 3:
                y = np.asarray(y_list[:n], dtype=float)
                x = eff_arr[:n]
                if np.nanstd(y) > 0 and np.nanstd(x) > 0:
                    r = np.corrcoef(x, y)[0, 1]
                    if np.isfinite(r):
                        corrs[c] = float(r)
                        used_cols.append(c)
    else:
        # Fallback: row-aligned correlation
        for c in metrics:
            try:
                y = df.iloc[[j for j in idx_map if j >= 0]][c].astype(float).to_numpy()
                n = min(len(y), len(eff_arr))
                if n >= 3:
                    y = y[:n]
                    x = eff_arr[:n]
                    if np.nanstd(y) > 0 and np.nanstd(x) > 0:
                        r = np.corrcoef(x, y)[0, 1]
                        if np.isfinite(r):
                            corrs[c] = float(r)
                            used_cols.append(c)
            except Exception:
                continue

    return EffortCorrelationResponse(
        matched_count=matched_count,
        total_words=len(words),
        metric_correlations=corrs,
        used_columns=used_cols,
    token_column=str(used_column) if used_column is not None else None,
    )
