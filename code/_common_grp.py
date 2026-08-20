"""
_common.py — 본 분석과 견고성 검증 스크립트가 공유하는 유틸리티
================================================================
제공 함수:
  - load_data(use_reclassified=True): 데이터 로드 + 재분류 라벨 merge
                                       + 기간 필터 + 자식 댓글 추출
                                       + 파생 변수 계산
  - make_panel(ch, entity): PanelOLS 입력 패널 생성
  - fit_spec(...): 회귀 명세 적합
  - format_table(...): 결과 표 포맷팅
  - section(title): 섹션 헤더 출력

import 방법:
  from _common import load_data, make_panel, fit_spec, format_table, section
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
DATA_PATH = Path("final_analysis_data_clean_85w.csv")
# v3 dual reclassification (Gemini Flash-Lite + Flash escalation, far-right Tier 1+2 universe)
RECLASSIFIED_PATH = Path("results/dual_reclassified.csv")
# v2 legacy (참고용 — robustness check 시 사용): "results/intolerance_reclassified.csv"

USE_COLS = [
    "thread_id", "video_id", "author_id",
    "is_reply",
    "incivility_pred", "intolerance_pred",
    "thread_child_count", "parent_like_count", "text_length",
    "event_aftermath",
    "view_count", "comment_count", "days_elapsed",
    "published_at_x",
]

# 윤석열 전 대통령 관련 주요 사건 (KST 기준)
EVENTS = [
    ("martial_law",  "2024-12-03"),  # 비상계엄 선포
    ("impeachment",  "2024-12-14"),  # 탄핵 소추 가결
    ("arrest",       "2025-01-15"),  # 구속
    ("release",      "2025-03-08"),  # 석방
    ("removal",      "2025-04-04"),  # 파면 확정
]

WINDOW_DAYS = 2  # 문헌(Brady 2017, Frischlich 2021, Wojcieszak 2022) + 자체 sensitivity 결과
                  # 사건 직후 1–2일이 SNS 정치 댓글 반응 peak window.
                  # 5/13 sensitivity log: δ_1(Intol) 1일 p=0.016, 2일 p=0.018, 3일+ dilution.
ANALYSIS_START = pd.Timestamp("2024-12-03")

CONTROLS = [
    "log_text_length",
    "log_parent_like",
    "log_thread_size",
    "days_elapsed",
    "event_cross",
    "event_long",
    "comment_participation_rate",
]


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────
def load_data(use_reclassified: bool = True, verbose: bool = True) -> pd.DataFrame:
    """전체 파이프라인: 로드 → 재분류 → 기간 필터 → 자식 추출 → 파생 변수."""
    if verbose:
        section("데이터 로드")
    t0 = time.time()

    df = pd.read_csv(DATA_PATH, usecols=USE_COLS)
    # 안정적 comment_id 추가 (classify_dual_batch.py 와 동일 인덱싱: 원본 CSV row order)
    df = df.reset_index().rename(columns={"index": "comment_id"})
    if verbose:
        print(f"원시 행 수: {len(df):,}")

    # 시간 파싱: UTC → KST → tz-naive
    df["published_at_x"] = pd.to_datetime(
        df["published_at_x"], errors="coerce", utc=True,
    )
    df["published_at_x"] = (
        df["published_at_x"]
        .dt.tz_convert("Asia/Seoul")
        .dt.tz_localize(None)
    )
    before = len(df)
    df = df.dropna(subset=["published_at_x"])
    df = df[df["published_at_x"] >= ANALYSIS_START].copy()
    df = df.reset_index(drop=True)
    if verbose:
        print(f"기간 필터 (≥ {ANALYSIS_START.date()} KST): "
              f"{before:,} → {len(df):,}")

    # 재분류 라벨 merge
    df = _merge_reclassified(df, use_reclassified, verbose)

    # ── 채널 그룹 한정 (KCI 변환, dual 층화) ──
    import json as _json, csv as _csv
    _grp = globals().get("GROUP_VERDICTS", {"FR_extreme","FR_radical"})
    _fr=set()
    for _r in _csv.DictReader(open("reviews/재분류_매핑_85to_p2_FR47_2026-07-01.csv", encoding="utf-8-sig")):
        if _r["keep_in_analysis"]=="Y" and _r["verdict_final"] in _grp: _fr.add(_r["channel_id"].strip())
    _v2c=_json.load(open("v2c_video_to_channel.json"))
    _frv={v for v,c in _v2c.items() if c in _fr}
    _b=len(df)
    df=df[df["video_id"].isin(_frv)].reset_index(drop=True)
    if verbose: print(f"  채널그룹 필터 {_grp}: {_b:,} -> {len(df):,} ({len(_fr)}채널)")

    # 부모-자식 분리
    # ※ 2026-07-27 수정: parent_like_count 는 원자료에서 부모 행에만 채워져 있다.
    #   기존에는 부모 병합이 라벨 2개만 가져와 자식 행에서 전 행 NaN → log1p(0)=0 인
    #   빈 열이 되었고, log_parent_like 통제가 침묵 상태로 모든 모형에서 빠져 있었다
    #   (drop_absorbed=True 가 조용히 제거). thread_id 로 부모의 좋아요 수를 함께 병합한다.
    #   표본 수는 변하지 않으며, 복구 후 유의한 통제변수다.
    parents = (
        df[df["is_reply"] == 0][
            ["thread_id", "incivility_pred", "intolerance_pred", "parent_like_count"]
        ].rename(columns={"incivility_pred": "parent_inciv",
                          "intolerance_pred": "parent_intol",
                          "parent_like_count": "_parent_like"})
    )
    ch = (
        df[df["is_reply"] == 1]
        .drop(columns=["parent_like_count"])
        .merge(parents, on="thread_id", how="left")
        .rename(columns={"_parent_like": "parent_like_count"})
    )
    ch = ch.dropna(subset=["parent_inciv", "parent_intol"]).copy()
    ch["parent_inciv"] = ch["parent_inciv"].astype(int)
    ch["parent_intol"] = ch["parent_intol"].astype(int)
    ch = ch.reset_index(drop=True)
    if verbose:
        print(f"분석 대상 자식 댓글: {len(ch):,}")
        print(f"부모 비시민성 라벨율: {ch['parent_inciv'].mean()*100:.2f}%")
        print(f"부모 불관용 라벨율:   {ch['parent_intol'].mean()*100:.2f}%")

    # 파생 변수
    ch = _build_features(ch, verbose)

    if verbose:
        print(f"로드 완료 ({time.time()-t0:.1f}초)")
    return ch


def _merge_reclassified(df: pd.DataFrame, use_reclassified: bool,
                        verbose: bool) -> pd.DataFrame:
    """
    v3 dual reclassification (Gemini Flash-Lite + Flash) 적용.
    부모·자식 양 차원 (incivility, intolerance) 모두 v3 라벨로 교체.
    Universe 는 v3 매칭 가능한 row 만 유지 (= far-right Tier 1+2).
    """
    if not use_reclassified:
        if verbose:
            print("  재분류 비활성화 → KcELECTRA v2 사용 (전체 universe)")
        return df

    if not RECLASSIFIED_PATH.exists():
        if verbose:
            print(f"  [경고] {RECLASSIFIED_PATH} 없음 → v2 사용")
        return df

    v3 = pd.read_csv(
        RECLASSIFIED_PATH,
        usecols=["comment_id", "inciv_new", "intol_new"],
    )
    # 유효 v3 라벨만
    v3 = v3.dropna(subset=["inciv_new", "intol_new"])
    v3 = v3[(v3["inciv_new"] != -1) & (v3["intol_new"] != -1)]
    v3 = v3.drop_duplicates(subset="comment_id", keep="first")
    v3["inciv_new"] = v3["inciv_new"].astype(int)
    v3["intol_new"] = v3["intol_new"].astype(int)

    before = len(df)
    inciv_v2_rate = df["incivility_pred"].mean()
    intol_v2_rate = df["intolerance_pred"].mean()

    # inner merge → universe 를 Tier 1+2 ∩ 기간필터로 축소
    df = df.merge(v3, on="comment_id", how="inner")

    # 양 차원 모두 v3 라벨로 교체 (부모+자식 전부)
    df["incivility_pred"] = df["inciv_new"]
    df["intolerance_pred"] = df["intol_new"]
    df = df.drop(columns=["inciv_new", "intol_new"])

    if verbose:
        print(f"  v3 dual 적용 (Tier 1+2 universe): {before:,} → {len(df):,}")
        print(f"    inciv  KcELECTRA {inciv_v2_rate*100:.2f}% → "
              f"Gemini v3 {df['incivility_pred'].mean()*100:.2f}%")
        print(f"    intol  KcELECTRA {intol_v2_rate*100:.2f}% → "
              f"Gemini v3 {df['intolerance_pred'].mean()*100:.2f}%")
    return df


def _build_features(ch: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """파생 변수: 참여율, 사건, Mundlak 분해, 로그 변환."""
    # 댓글 참여율
    ch["comment_participation_rate"] = (
        ch["comment_count"] / ch["view_count"].replace(0, np.nan)
    ).fillna(0)

    # 기존 event_aftermath 기반 변수 (Mundlak 통제용으로 유지)
    video_event_mean = ch.groupby("video_id")["event_aftermath"].transform("mean")
    ch["event_cross"] = video_event_mean
    ch["event_long"]  = ch["event_aftermath"] - video_event_mean

    # 사건 변수 (in_event_window, days_into_window)
    ch = _add_event_variables(ch, verbose)

    # 작성자 단위 Mundlak 분해
    auth_pinciv_mean = ch.groupby("author_id")["parent_inciv"].transform("mean")
    auth_pintol_mean = ch.groupby("author_id")["parent_intol"].transform("mean")
    ch["PIncivB"] = auth_pinciv_mean
    ch["PIncivW"] = ch["parent_inciv"] - auth_pinciv_mean
    ch["PIntolB"] = auth_pintol_mean
    ch["PIntolW"] = ch["parent_intol"] - auth_pintol_mean

    # 로그 변환
    ch["log_text_length"] = np.log1p(ch["text_length"].clip(lower=0))
    ch["log_parent_like"] = np.log1p(
        ch["parent_like_count"].fillna(0).clip(lower=0)
    )
    ch["log_thread_size"] = np.log1p(
        ch["thread_child_count"].fillna(0).clip(lower=0)
    )

    # 결측 제거
    required = (
        ["incivility_pred", "intolerance_pred",
         "parent_inciv", "parent_intol",
         "PIncivW", "PIncivB", "PIntolW", "PIntolB",
         "video_id", "thread_id", "author_id",
         "in_event_window", "days_into_window"]
        + CONTROLS
    )
    before = len(ch)
    ch = ch.dropna(subset=required)
    if verbose:
        print(f"결측 제거: {before - len(ch):,}행 → {len(ch):,}행 잔류")
    return ch.reset_index(drop=True)


def _add_event_variables(ch: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    방식 D: in_event_window + days_into_window
      - in_event_window: 해당 댓글이 사건 후 WINDOW_DAYS 일 이내면 1
      - days_into_window: 사건 후 경과일 (0 ~ WINDOW_DAYS-1)
      - 여러 사건 윈도우 겹치면 가장 가까운 과거 사건 기준
    """
    comment_date = ch["published_at_x"].dt.normalize()

    days_since_list = []
    for _, date_str in EVENTS:
        ev_date = pd.Timestamp(date_str)
        days = (comment_date - ev_date).dt.days
        days_since_list.append(days)

    days_matrix = np.column_stack(days_since_list)  # (N, len(EVENTS))

    # 사건 이후(>=0) 이고 윈도우 내(<WINDOW_DAYS)
    in_window_per_event = (days_matrix >= 0) & (days_matrix < WINDOW_DAYS)
    in_event_window = in_window_per_event.any(axis=1).astype(int)

    # 윈도우 내에서 가장 가까운 과거 사건까지의 경과일
    days_valid = np.where(in_window_per_event, days_matrix, np.inf)
    days_into_window = np.where(
        in_event_window == 1,
        days_valid.min(axis=1),
        0,
    ).astype(int)

    ch["in_event_window"] = in_event_window
    ch["days_into_window"] = days_into_window

    if verbose:
        print(f"\n사건 윈도우 구성 (WINDOW_DAYS={WINDOW_DAYS}):")
        for name, date_str in EVENTS:
            print(f"    {name}: {date_str}")
        print(f"  in_event_window=1 비율: "
              f"{ch['in_event_window'].mean()*100:.2f}% "
              f"(n={ch['in_event_window'].sum():,})")
    return ch


# ─────────────────────────────────────────────────────────────
# 2. PanelOLS 입력 생성
# ─────────────────────────────────────────────────────────────
def make_panel(ch: pd.DataFrame, entity: str = "video_id"):
    """
    PanelOLS용 패널 인덱스 설정.
    entity: 'video_id' (기본) 또는 'author_id' (user-FE용)
    반환: (panel_df, thread_cluster)
    """
    ch = ch.copy()
    ch["obs_id"] = np.arange(len(ch))
    ch_panel = ch.set_index([entity, "obs_id"])
    thread_cluster = ch_panel["thread_id"]
    return ch_panel, thread_cluster


# ─────────────────────────────────────────────────────────────
# 3. 회귀 명세 적합
# ─────────────────────────────────────────────────────────────
def fit_spec(
    ch_panel: pd.DataFrame,
    thread_cluster: pd.Series,
    outcome: str,
    spec: str,
    extra_rhs: list = None,
    entity_effects: bool = True,
    other_effects: pd.Series = None,
):
    """
    회귀 명세 적합.

    outcome: 'incivility_pred' or 'intolerance_pred'
    spec:    'pooled' | 'cross' | 'userfe' | 'mundlak'
    extra_rhs: 추가 우변 변수
    entity_effects: entity 차원 FE 포함 여부
    other_effects: 두 번째 FE (선택)
    """
    if spec == "pooled":
        own = "parent_inciv" if outcome == "incivility_pred" else "parent_intol"
        rhs_main = [own]
    elif spec == "cross":
        rhs_main = ["parent_inciv", "parent_intol"]
    elif spec == "userfe":
        rhs_main = ["parent_inciv", "parent_intol"]
    elif spec == "mundlak":
        rhs_main = ["PIncivW", "PIncivB", "PIntolW", "PIntolB"]
    else:
        raise ValueError(f"Unknown spec: {spec}")

    if extra_rhs:
        rhs = rhs_main + extra_rhs + CONTROLS
    else:
        rhs = rhs_main + CONTROLS

    # 중복 제거 (순서 유지)
    seen = set()
    rhs = [x for x in rhs if not (x in seen or seen.add(x))]

    y = ch_panel[outcome].astype(float)
    X = ch_panel[rhs].astype(float)

    kwargs = dict(
        entity_effects=entity_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    if other_effects is not None:
        kwargs["other_effects"] = other_effects

    mod = PanelOLS(y, X, **kwargs)
    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        clusters=thread_cluster,
    )
    return res


# ─────────────────────────────────────────────────────────────
# 4. 결과 포맷팅
# ─────────────────────────────────────────────────────────────
def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def fmt_cell(res, var):
    if var not in res.params.index:
        return "—", ""
    coef = res.params[var]
    se = res.std_errors[var]
    p = res.pvalues[var]
    return f"{coef:+.4f}{stars(p)}", f"({se:.4f})"


def format_table(results: dict, display_vars: list, col_labels: dict,
                 footer_fns: list = None):
    """
    results: {key: PanelResults}
    display_vars: 표에 출력할 변수 리스트
    col_labels: {key: column label string} — 출력 순서 결정
    footer_fns: [(label, lambda res: str), ...] — 하단 통계
    """
    keys = list(col_labels.keys())
    rows = []

    for var in display_vars:
        coef_row = {"variable": var}
        se_row = {"variable": ""}
        for key in keys:
            coef, se = fmt_cell(results[key], var)
            coef_row[col_labels[key]] = coef
            se_row[col_labels[key]] = se
        rows.append(coef_row)
        rows.append(se_row)

    if footer_fns is None:
        footer_fns = [
            ("N",            lambda r: f"{int(r.nobs):,}"),
            ("R² (within)",  lambda r: f"{r.rsquared_within:.4f}"),
            ("R² (overall)", lambda r: f"{r.rsquared_overall:.4f}"),
        ]
    for label, fn in footer_fns:
        row = {"variable": label}
        for key in keys:
            try:
                row[col_labels[key]] = fn(results[key])
            except Exception:
                row[col_labels[key]] = "—"
        rows.append(row)

    return pd.DataFrame(rows)
