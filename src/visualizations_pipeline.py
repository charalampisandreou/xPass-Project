from __future__ import annotations

import inspect
import warnings
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from mplsoccer import Pitch

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False


# --------------------------------------------------------------------------- #
# Shared validation helpers
# --------------------------------------------------------------------------- #

RATE_COLUMNS = ("actual_completion_rate", "expected_completion_rate")
PASS_OUTCOME_CONTRACT = (
    "pass_outcome must be binary: 1 = completed pass, 0 = incomplete/failed pass."
)


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"`{name}` is missing required columns: {missing}")


def _to_numeric_strict(df: pd.DataFrame, cols, name: str, hint: str = "") -> pd.DataFrame:
    """Return a copy of `df` with `cols` coerced to float.

    Missing values (NaN/None) are preserved so the caller can decide whether to drop or
    reject them. Non-numeric text and +/-inf are rejected instead of being silently coerced.
    """
    df = df.copy()
    for col in cols:
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        unparseable = raw.notna() & num.isna()
        if unparseable.any():
            examples = raw[unparseable].unique()[:3].tolist()
            raise ValueError(
                f"`{name}['{col}']` contains non-numeric values (e.g. {examples}). {hint}".rstrip()
            )
        num = num.astype(float)
        if np.isinf(num).any():
            raise ValueError(f"`{name}['{col}']` contains infinite values.")
        df[col] = num
    return df


def _require_no_missing(df: pd.DataFrame, cols, name: str, hint: str = "") -> None:
    for col in cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise ValueError(f"`{name}['{col}']` contains {n_missing} missing value(s). {hint}".rstrip())


def _require_non_negative_passes(df: pd.DataFrame, name: str) -> None:
    negative = df["total_passes"] < 0
    if negative.any():
        raise ValueError(
            f"`{name}['total_passes']` must be non-negative; found {int(negative.sum())} "
            f"negative value(s) (min={df['total_passes'].min():g})."
        )


def _validate_rate_columns(df: pd.DataFrame, cols, name: str, missing_hint: str = "") -> None:
    """Require `cols` to be fully populated proportions in [0, 1] (0.85 means 85%). Never clips."""
    _require_no_missing(df, cols, name, hint=missing_hint)
    for col in cols:
        s = df[col]
        out_of_range = (s < 0) | (s > 1)
        if out_of_range.any():
            raise ValueError(
                f"`{name}['{col}']` must be a proportion between 0 and 1 (e.g. 0.85 for 85%), but "
                f"{int(out_of_range.sum())} value(s) fall outside that range "
                f"(min={s.min():.4g}, max={s.max():.4g}). Percent-scale data (0-100) must be "
                f"divided by 100 first."
            )


def _warn_dropped(n_dropped: int, name: str) -> None:
    if n_dropped:
        warnings.warn(
            f"`{name}`: dropped {n_dropped} row(s) with missing values in required columns.",
            UserWarning,
            stacklevel=3,
        )


def _prepare_rate_frame(
    df: pd.DataFrame,
    min_passes: int,
    *,
    name: str,
    id_col: str,
    entity: str,
    extra_numeric=(),
) -> pd.DataFrame:
    """Validate and filter a per-player / per-team summary table.

    Checks columns, numeric types, non-negative `total_passes`, the `min_passes` filter
    (must leave at least one row), unique non-null identifiers and [0, 1] completion rates.
    Returns a filtered copy with a clean RangeIndex.
    """
    _check_columns(df, [id_col, "total_passes", *RATE_COLUMNS, *extra_numeric], name)
    out = _to_numeric_strict(df, ["total_passes", *RATE_COLUMNS, *extra_numeric], name)

    _require_no_missing(out, ["total_passes"], name)
    _require_non_negative_passes(out, name)

    filtered = out[out["total_passes"] >= min_passes].reset_index(drop=True)
    if filtered.empty:
        max_passes = f"{out['total_passes'].max():g}" if len(out) else "n/a"
        raise ValueError(
            f"No {entity} remain after applying min_passes={min_passes} "
            f"(input rows: {len(out)}, max total_passes: {max_passes})."
        )

    if filtered[id_col].isna().any():
        raise ValueError(f"`{name}['{id_col}']` contains missing values.")
    duplicated = filtered.loc[filtered[id_col].duplicated(), id_col].unique()
    if len(duplicated):
        raise ValueError(
            f"`{name}` must be a pre-aggregated summary with one row per `{id_col}`; "
            f"duplicated values: {list(duplicated[:5])}."
        )

    _validate_rate_columns(
        filtered, RATE_COLUMNS, name,
        missing_hint="Rows with zero passes have undefined rates; raise `min_passes` to exclude them.",
    )
    _require_no_missing(filtered, extra_numeric, name)
    return filtered


def draw_leaderboard_table(
    player_df: pd.DataFrame,
    top_n: int = 10,
    min_passes: int = 300
) -> plt.Figure:
    """Renders a dark-slate scouting leaderboard table."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    extra = ("total_pva", "CPOE") if "CPOE" in player_df.columns else ("total_pva",)
    df = _prepare_rate_frame(
        player_df, min_passes, name="player_df", id_col="player_name",
        entity="players", extra_numeric=extra,
    )

    if 'CPOE' not in df.columns:
        df['CPOE'] = df['actual_completion_rate'] - df['expected_completion_rate']

    top_players = (
        df.sort_values(by='total_pva', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    headers = ['Rank', 'Player', 'Passes', 'Completed', 'Mean xP', 'CPOE', 'Total PVA']
    col_widths = [0.06, 0.33, 0.10, 0.14, 0.12, 0.11, 0.14]

    cell_text = []
    cell_colors = []

    for rank, row in top_players.iterrows():
        cpoe_val = row['CPOE'] * 100
        cpoe_str = f'+{cpoe_val:.1f}%' if cpoe_val > 0 else f'{cpoe_val:.1f}%'
        pva_str = f"+{row['total_pva']:.2f}" if row['total_pva'] > 0 else f"{row['total_pva']:.2f}"

        row_data = [
            str(rank + 1),
            str(row['player_name']),
            f"{int(row['total_passes']):,}",
            f"{row['actual_completion_rate'] * 100:.1f}%",
            f"{row['expected_completion_rate'] * 100:.1f}%",
            cpoe_str,
            pva_str,
        ]
        cell_text.append(row_data)

        if rank < 3:
            cell_colors.append(['#f0fdf4'] * len(headers))
        elif rank % 2 == 0:
            cell_colors.append(['#f8fafc'] * len(headers))
        else:
            cell_colors.append(['#ffffff'] * len(headers))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=col_widths,
        cellColours=cell_colors,
        colColours=['#1e293b'] * len(headers),
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.88]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=12, fontname='DejaVu Sans')
            cell.set_edgecolor('#0f172a')
        else:
            cell.set_edgecolor('#cbd5e1')

            if col_idx == 0:
                cell.set_text_props(ha='center', fontname='DejaVu Sans')
            elif col_idx == 1:
                cell.set_text_props(ha='left', fontname='DejaVu Sans')
            elif col_idx > 1:
                cell.set_text_props(ha='right', fontname='DejaVu Sans')

            if col_idx == 6:
                cell.set_text_props(weight='bold', color='#047857', ha='right', fontname='DejaVu Sans')

    fig.text(
        0.5, 0.96,
        f'Top {len(top_players)} Playmakers by Pass Value Added (PVA)',
        ha='center', va='center', fontsize=20, weight='bold', color='#0f172a', fontname='DejaVu Sans'
    )
    fig.text(
        0.5, 0.91,
        f'Minimum Threshold: {min_passes} Passes  |  Model: Calibrated XGBoost',
        ha='center', va='center', fontsize=12, color='#64748b', fontname='DejaVu Sans'
    )

    return fig


PALETTE = {
    "bg_figure":     "#ffffff",
    "bg_axes":       "#f6f8fb",
    "grid":          "#8a95a6",
    "spine":         "#c7cfd9",
    "text_dark":     "#111827",
    "text_mid":      "#4b5563",
    "text_light":    "#9ca3af",
    "baseline":      "#64748b",
    "average_fill":  "#8b96a8",
    "average_edge":  "#ffffff",
    "elite_fill":    "#059669",
    "poor_fill":     "#e11d48",
    "crosshair":     "#7c8797",
    "legend_face":   "#ffffff",
}

FONT = "DejaVu Sans"

Z_GRADIENT = 0
Z_GRID = 1
Z_GUIDES = 2
Z_SCATTER_BG = 3
Z_SCATTER_FG = 4
Z_LABELS = 6
Z_CALLOUTS = 10
Z_LEGEND = 20


def shorten_player_name(full_name: str, name_overrides: Optional[Dict[str, str]] = None) -> str:
    """Returns a short display name for a player."""
    if name_overrides and full_name in name_overrides:
        return name_overrides[full_name]
    return full_name.split()[-1]


def _check_quantile(value: float, label: str) -> float:
    if not (np.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be a number between 0 and 1, got {value!r}.")
    return float(value)


def _compute_thresholds(
    cpoe: pd.Series,
    elite_threshold: Optional[float],
    poor_threshold: Optional[float],
    elite_quantile: float,
    poor_quantile: float,
) -> Tuple[float, float]:
    """Resolves elite/poor CPOE cutoffs and guarantees elite_threshold > poor_threshold."""
    if elite_threshold is None:
        elite_threshold = float(cpoe.quantile(_check_quantile(elite_quantile, "elite_quantile")))
    if poor_threshold is None:
        poor_threshold = float(cpoe.quantile(_check_quantile(poor_quantile, "poor_quantile")))

    elite_threshold, poor_threshold = float(elite_threshold), float(poor_threshold)
    if not (np.isfinite(elite_threshold) and np.isfinite(poor_threshold)):
        raise ValueError(
            f"CPOE thresholds must be finite (elite={elite_threshold}, poor={poor_threshold})."
        )
    if not elite_threshold > poor_threshold:
        raise ValueError(
            f"elite_threshold ({elite_threshold:+.4f}) must be strictly greater than "
            f"poor_threshold ({poor_threshold:+.4f}), otherwise tiers overlap or are ambiguous. "
            f"With quantile-based cutoffs this happens when elite_quantile <= poor_quantile or "
            f"when too many rows share the same CPOE (e.g. a single row); pass explicit "
            f"`elite_threshold` / `poor_threshold` values instead."
        )
    return elite_threshold, poor_threshold


def _scaled_bubble_sizes(volume: pd.Series, base: float = 90.0, span: float = 900.0) -> pd.Series:
    """Scatter areas scaled linearly by pass volume; rejects non-positive maxima."""
    vmax = float(volume.max())
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Bubble sizing requires at least one row with total_passes > 0.")
    return base + (volume / vmax) * span


def _adjust_text_is_modern() -> bool:
    """True for adjustText >= 1.0 (`expand`, `objects`); False for the legacy 0.x API."""
    params = inspect.signature(adjust_text).parameters
    return "expand" in params and "expand_points" not in params


def _apply_adjust_text(texts, ax, avoid_objects, x=None, y=None) -> None:
    """Run adjustText with the keyword set matching the installed version (no-op if absent)."""
    if not texts or not _HAS_ADJUST_TEXT:
        return
    arrowprops = dict(arrowstyle="-", color=PALETTE["text_light"],
                      lw=0.9, alpha=0.8, shrinkA=5, shrinkB=5)
    if _adjust_text_is_modern():
        adjust_text(
            texts, x=x, y=y, ax=ax, objects=avoid_objects,
            expand=(1.15, 1.35), force_text=(0.3, 0.5), force_static=(0.3, 0.5),
            arrowprops=arrowprops,
        )
    else:
        adjust_text(
            texts, x=x, y=y, ax=ax, add_objects=avoid_objects,
            expand_points=(1.5, 1.7), expand_text=(1.15, 1.25),
            arrowprops=arrowprops,
        )


def plot_pass_risk_execution(
    player_df: pd.DataFrame,
    min_passes: int = 300,
    elite_threshold: Optional[float] = None,
    poor_threshold: Optional[float] = None,
    elite_quantile: float = 0.75,
    poor_quantile: float = 0.25,
    label_elite: bool = True,
    label_poor_n: int = 3,
    label_average: bool = False,
    name_overrides: Optional[Dict[str, str]] = None,
    legend_anchor: Tuple[float, float] = (0.985, 0.5)
) -> plt.Figure:
    """Renders a Risk vs. Execution scatter plot mapping Mean xP against Actual Completion %."""
    df = _prepare_rate_frame(
        player_df, min_passes, name="player_df", id_col="player_name", entity="players",
    )
    df['CPOE'] = df['actual_completion_rate'] - df['expected_completion_rate']

    elite_threshold, poor_threshold = _compute_thresholds(
        df['CPOE'], elite_threshold, poor_threshold, elite_quantile, poor_quantile
    )

    league_xp = df['expected_completion_rate'].mean()
    league_actual = df['actual_completion_rate'].mean()
    min_val = min(df['expected_completion_rate'].min(), df['actual_completion_rate'].min()) - 0.022
    max_val = max(df['expected_completion_rate'].max(), df['actual_completion_rate'].max()) + 0.022

    elite = df['CPOE'] >= elite_threshold
    poor = df['CPOE'] <= poor_threshold
    average = ~(elite | poor)

    sizes = _scaled_bubble_sizes(df['total_passes'])

    plt.rcParams['font.family'] = FONT
    fig, ax = plt.subplots(figsize=(13, 8.8), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg_figure"])
    ax.set_facecolor(PALETTE["bg_axes"])
    fig.subplots_adjust(top=0.89, bottom=0.11, left=0.08, right=0.96)

    cap = max_val - min_val
    grid_res = np.linspace(min_val, max_val, 400)
    XX, YY = np.meshgrid(grid_res, grid_res)
    ZZ = YY - XX

    performance_cmap = LinearSegmentedColormap.from_list(
        "cpoe_gradient", [PALETTE["poor_fill"], "#ffffff", PALETTE["elite_fill"]]
    )
    ax.imshow(
        ZZ, extent=(min_val, max_val, min_val, max_val), origin="lower",
        cmap=performance_cmap, vmin=-cap, vmax=cap, alpha=0.5,
        interpolation="bilinear", aspect="auto", zorder=Z_GRADIENT,
    )

    ax.plot([min_val, max_val], [min_val, max_val],
            color=PALETTE["baseline"], linestyle=(0, (5, 3)), linewidth=1.6,
            alpha=0.9, zorder=Z_GUIDES)

    ax.axvline(league_xp, color=PALETTE["crosshair"], linestyle=":", linewidth=1.3, zorder=Z_GUIDES, alpha=0.8)
    ax.axhline(league_actual, color=PALETTE["crosshair"], linestyle=":", linewidth=1.3, zorder=Z_GUIDES, alpha=0.8)

    def _pill(x, y, text, color, ha, va):
        return ax.text(x, y, text, fontsize=10.5, weight="bold", color=color,
                        ha=ha, va=va, zorder=Z_CALLOUTS, alpha=0.95,
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                  edgecolor=color, linewidth=1.2, alpha=0.92))

    pill_elite = _pill(min_val + 0.012, max_val - 0.012, "HIGH RISK / OVERPERFORMANCE\nlow xP · high actual",
                        PALETTE["elite_fill"], "left", "top")
    pill_poor = _pill(max_val - 0.012, min_val + 0.012, "LOW RISK / UNDERPERFORMANCE\nhigh xP · low actual",
                       PALETTE["poor_fill"], "right", "bottom")

    glow_effect = [pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)]

    ax.scatter(df.loc[average, 'expected_completion_rate'], df.loc[average, 'actual_completion_rate'],
               s=sizes[average], color=PALETTE["average_fill"], alpha=0.8,
               edgecolors=PALETTE["average_edge"], linewidth=1.1, zorder=Z_SCATTER_BG)

    ax.scatter(df.loc[poor, 'expected_completion_rate'], df.loc[poor, 'actual_completion_rate'],
               s=sizes[poor], color=PALETTE["poor_fill"], alpha=0.9,
               edgecolors="white", linewidth=1.4, zorder=Z_SCATTER_FG)

    ax.scatter(df.loc[elite, 'expected_completion_rate'], df.loc[elite, 'actual_completion_rate'],
               s=sizes[elite], color=PALETTE["elite_fill"], alpha=0.92,
               edgecolors="white", linewidth=1.4, zorder=Z_SCATTER_FG)

    rows_to_label = []
    if label_elite:
        rows_to_label.append(df[elite])
    if label_poor_n > 0:
        rows_to_label.append(df[poor].nsmallest(label_poor_n, 'CPOE'))
    if label_average:
        rows_to_label.append(df[average])

    texts = []
    if rows_to_label:
        label_df = pd.concat(rows_to_label).drop_duplicates(subset='player_name')
        for _, row in label_df.iterrows():
            short_name = shorten_player_name(str(row['player_name']), name_overrides)
            t = ax.text(
                row['expected_completion_rate'], row['actual_completion_rate'] + 0.006,
                short_name, fontsize=10.5, weight="bold", color=PALETTE["text_dark"],
                ha="center", va="bottom", zorder=Z_LABELS,
                path_effects=glow_effect,
            )
            texts.append(t)

    _apply_adjust_text(texts, ax, [pill_elite, pill_poor])

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel('Expected Pass Completion % (Mean xP)  →  safer passes to the right',
                  fontsize=11.5, weight='bold', color=PALETTE["text_mid"], labelpad=12)
    ax.set_ylabel('Actual Pass Completion %', fontsize=11.5, weight='bold',
                  color=PALETTE["text_mid"], labelpad=12)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.tick_params(colors=PALETTE["text_mid"], labelsize=10)

    ax.grid(True, color=PALETTE["grid"], linestyle="-", linewidth=0.9, alpha=0.35, zorder=Z_GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["spine"])
        ax.spines[side].set_linewidth(1.1)

    fig.text(0.52, 0.97, "Pass Risk vs. Execution", fontsize=24, weight="bold",
              color=PALETTE["text_dark"], ha="center")
    fig.text(0.52, 0.935, "Completion Over Expectation (CPOE) — who beats their model, and by how much",
              fontsize=13, color=PALETTE["text_mid"], ha="center")
    fig.text(0.96, 0.97, f"n = {len(df)}", fontsize=11, color=PALETTE["text_light"],
              ha="right", weight="bold")
    fig.text(0.96, 0.948, f"min. {min_passes} passes", fontsize=9.5, color=PALETTE["text_light"], ha="right")

    fig.text(0.08, 0.015,
              "Dashed line: y = x (expected completion)  ·  Model: calibrated XGBoost expected-completion model  ·  "
              "bubble size ∝ total pass volume  ·  background shade ∝ CPOE",
              fontsize=9.5, color=PALETTE["text_light"], ha="left", style="italic")

    category_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["elite_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label=f'Elite Overperformer (CPOE ≥ {elite_threshold:+.1%})'),
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["average_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label='Within expectation'),
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["poor_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label=f'Underperformer (CPOE ≤ {poor_threshold:+.1%})'),
    ]
    legend_volumes = pd.Series(
        [df['total_passes'].min(), df['total_passes'].median(), df['total_passes'].max()], dtype=float
    )
    legend_areas = _scaled_bubble_sizes(legend_volumes)
    size_handles = [
        Line2D([0], [0], marker='o', linestyle='', markeredgecolor="white", alpha=0.85,
               markersize=np.sqrt(area) * 0.62,
               markerfacecolor=PALETTE["text_light"], label=f"{int(v):,} passes")
        for v, area in zip(legend_volumes, legend_areas)
    ]

    blank = Line2D([], [], linestyle='none', marker='', color='none')
    combo_handles = [blank] + category_handles + [blank] + [blank] + size_handles
    combo_labels = (
        ["PERFORMANCE TIER"] + [h.get_label() for h in category_handles]
        + [""] + ["TOTAL PASS VOLUME"] + [h.get_label() for h in size_handles]
    )
    header_rows = {0, 5}
    spacer_rows = {4}

    combo_legend = ax.legend(
        combo_handles, combo_labels,
        loc='center right', bbox_to_anchor=legend_anchor,
        frameon=True, fancybox=True, framealpha=0.94,
        facecolor=PALETTE["legend_face"], edgecolor=PALETTE["spine"],
        fontsize=10.5, handletextpad=0.9, labelspacing=0.75, borderpad=1.0,
    )
    combo_legend.set_zorder(Z_LEGEND)

    for i, txt in enumerate(combo_legend.get_texts()):
        if i in header_rows:
            txt.set_weight("bold")
            txt.set_color(PALETTE["text_dark"])
            txt.set_fontsize(10.5)
        elif i in spacer_rows:
            txt.set_text("")
        else:
            txt.set_color(PALETTE["text_mid"])

    return fig


def plot_team_pass_quadrants(
    team_df: pd.DataFrame,
    min_passes: int = 0,
    elite_threshold: Optional[float] = None,
    poor_threshold: Optional[float] = None,
    elite_quantile: float = 0.75,
    poor_quantile: float = 0.25,
    label_all: bool = True,
    legend_anchor: Tuple[float, float] = (1.02, 0.5),
) -> plt.Figure:
    """Renders a quadrant scatter: Mean xP (risk) vs. CPOE (execution), bubble size = pass volume."""
    df = _prepare_rate_frame(
        team_df, min_passes, name="team_df", id_col="team_name", entity="teams",
    )

    df['xP'] = df['expected_completion_rate']
    df['CPOE'] = df['actual_completion_rate'] - df['expected_completion_rate']

    elite_threshold, poor_threshold = _compute_thresholds(
        df['CPOE'], elite_threshold, poor_threshold, elite_quantile, poor_quantile
    )

    league_xp = df['xP'].mean()
    league_cpoe = df['CPOE'].mean()

    x_span = df['xP'].max() - df['xP'].min()
    x_pad = max(x_span * 0.16, 0.006)
    min_x, max_x = df['xP'].min() - x_pad, df['xP'].max() + x_pad
    cap = max(df['CPOE'].abs().max(), 1e-4) * 1.30
    min_y, max_y = -cap, cap

    elite = df['CPOE'] >= elite_threshold
    poor = df['CPOE'] <= poor_threshold
    average = ~(elite | poor)

    sizes = _scaled_bubble_sizes(df['total_passes'])

    plt.rcParams['font.family'] = FONT
    fig, ax = plt.subplots(figsize=(14, 8.8), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg_figure"])
    ax.set_facecolor(PALETTE["bg_axes"])
    left, right = 0.08, 0.74
    fig.subplots_adjust(top=0.87, bottom=0.11, left=left, right=right)

    # Vertical CPOE gradient background
    grid_y = np.linspace(min_y, max_y, 400)
    ZZ = np.tile(grid_y[:, None], (1, 400))
    performance_cmap = LinearSegmentedColormap.from_list(
        "cpoe_gradient", [PALETTE["poor_fill"], "#ffffff", PALETTE["elite_fill"]]
    )
    ax.imshow(
        ZZ, extent=(min_x, max_x, min_y, max_y), origin="lower",
        cmap=performance_cmap, vmin=-cap, vmax=cap, alpha=0.5,
        interpolation="bilinear", aspect="auto", zorder=Z_GRADIENT,
    )

    # Zero line (= actual matches expectation) and league-average crosshairs
    ax.axhline(0, color=PALETTE["baseline"], linestyle=(0, (5, 3)), linewidth=1.6,
               alpha=0.9, zorder=Z_GUIDES)
    ax.axvline(league_xp, color=PALETTE["crosshair"], linestyle=":", linewidth=1.3,
               zorder=Z_GUIDES, alpha=0.8)
    ax.axhline(league_cpoe, color=PALETTE["crosshair"], linestyle=":", linewidth=1.3,
               zorder=Z_GUIDES, alpha=0.8)

    def _pill(x, y, text, color, ha, va):
        return ax.text(x, y, text, fontsize=10.5, weight="bold", color=color,
                       ha=ha, va=va, zorder=Z_CALLOUTS, alpha=0.95,
                       bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                 edgecolor=color, linewidth=1.2, alpha=0.92))

    dx = (max_x - min_x) * 0.012
    dy = (max_y - min_y) * 0.014
    pills = [
        _pill(min_x + dx, max_y - dy, "AGGRESSIVE & EXECUTING\nlow xP · above expectation",
              PALETTE["elite_fill"], "left", "top"),
        _pill(max_x - dx, max_y - dy, "SAFE & EXECUTING\nhigh xP · above expectation",
              PALETTE["elite_fill"], "right", "top"),
        _pill(min_x + dx, min_y + dy, "AGGRESSIVE & STRUGGLING\nlow xP · below expectation",
              PALETTE["poor_fill"], "left", "bottom"),
        _pill(max_x - dx, min_y + dy, "SAFE & STRUGGLING\nhigh xP · below expectation",
              PALETTE["poor_fill"], "right", "bottom"),
    ]

    glow_effect = [pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)]

    for mask, color, alpha, edge, lw, z in (
        (average, PALETTE["average_fill"], 0.8, PALETTE["average_edge"], 1.1, Z_SCATTER_BG),
        (poor, PALETTE["poor_fill"], 0.9, "white", 1.4, Z_SCATTER_FG),
        (elite, PALETTE["elite_fill"], 0.92, "white", 1.4, Z_SCATTER_FG),
    ):
        if mask.any():
            ax.scatter(df.loc[mask, 'xP'], df.loc[mask, 'CPOE'],
                       s=sizes[mask], color=color, alpha=alpha,
                       edgecolors=edge, linewidth=lw, zorder=z)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    fig.canvas.draw()  # finalize transforms so labels can start just above each bubble's edge

    label_df = df if label_all else df[elite | poor]
    texts = []
    for idx, row in label_df.iterrows():
        radius_px = np.sqrt(sizes.loc[idx]) / 2 * fig.dpi / 72
        px, py = ax.transData.transform((row['xP'], row['CPOE']))
        _, y_lab = ax.transData.inverted().transform((px, py + radius_px + 2))
        t = ax.text(
            row['xP'], y_lab, str(row['team_name']),
            fontsize=10.5, weight="bold", color=PALETTE["text_dark"],
            ha="center", va="bottom", zorder=Z_LABELS,
            path_effects=glow_effect,
        )
        texts.append(t)

    # Copies: legacy adjustText mutates x/y in place, and pandas views can be read-only.
    _apply_adjust_text(texts, ax, pills, x=df['xP'].to_numpy(copy=True), y=df['CPOE'].to_numpy(copy=True))

    ax.set_xlabel('Mean Expected Pass Completion % (xP)  →  safer passes to the right',
                  fontsize=11.5, weight='bold', color=PALETTE["text_mid"], labelpad=12)
    ax.set_ylabel('Completion Over Expectation (CPOE)', fontsize=11.5, weight='bold',
                  color=PALETTE["text_mid"], labelpad=12)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v * 100:+.1f}%"))
    ax.tick_params(colors=PALETTE["text_mid"], labelsize=10)

    ax.grid(True, color=PALETTE["grid"], linestyle="-", linewidth=0.9, alpha=0.35, zorder=Z_GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["spine"])
        ax.spines[side].set_linewidth(1.1)

    cx = (left + right) / 2
    fig.text(cx, 0.945, "Team Pass Risk vs. Execution", fontsize=24, weight="bold",
             color=PALETTE["text_dark"], ha="center")
    fig.text(cx, 0.905, "Mean xP vs. Completion Over Expectation (CPOE) — who takes risk, who delivers",
             fontsize=13, color=PALETTE["text_mid"], ha="center")
    fig.text(0.96, 0.945, f"n = {len(df)}", fontsize=11, color=PALETTE["text_light"],
             ha="right", weight="bold")
    if min_passes > 0:
        fig.text(0.96, 0.922, f"min. {min_passes:,} passes", fontsize=9.5,
                 color=PALETTE["text_light"], ha="right")

    fig.text(left, 0.015,
             "Dashed line: CPOE = 0 (actual = expected)  ·  Dotted lines: league mean xP / CPOE  ·  "
             "bubble size ∝ total pass volume  ·  background shade ∝ CPOE",
             fontsize=9.5, color=PALETTE["text_light"], ha="left", style="italic")

    category_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["elite_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label=f'Elite (CPOE ≥ {elite_threshold:+.1%})'),
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["average_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label='Within expectation'),
        Line2D([0], [0], marker='o', linestyle='', markersize=11,
               markerfacecolor=PALETTE["poor_fill"], markeredgecolor="white", markeredgewidth=1.2,
               label=f'Under (CPOE ≤ {poor_threshold:+.1%})'),
    ]
    legend_volumes = pd.Series(
        [df['total_passes'].min(), df['total_passes'].median(), df['total_passes'].max()], dtype=float
    )
    legend_areas = _scaled_bubble_sizes(legend_volumes)
    size_handles = [
        Line2D([0], [0], marker='o', linestyle='', markeredgecolor="white", alpha=0.85,
               markersize=np.sqrt(area) * 0.62,
               markerfacecolor=PALETTE["text_light"], label=f"{int(v):,} passes")
        for v, area in zip(legend_volumes, legend_areas)
    ]

    blank = Line2D([], [], linestyle='none', marker='', color='none')
    combo_handles = [blank] + category_handles + [blank] + [blank] + size_handles
    combo_labels = (
        ["PERFORMANCE TIER"] + [h.get_label() for h in category_handles]
        + [""] + ["TOTAL PASS VOLUME"] + [h.get_label() for h in size_handles]
    )
    header_rows = {0, 5}
    spacer_rows = {4}

    combo_legend = ax.legend(
        combo_handles, combo_labels,
        loc='center left', bbox_to_anchor=legend_anchor,
        frameon=True, fancybox=True, framealpha=0.94,
        facecolor=PALETTE["legend_face"], edgecolor=PALETTE["spine"],
        fontsize=10, handletextpad=0.9, labelspacing=0.75, borderpad=1.0,
    )
    combo_legend.set_zorder(Z_LEGEND)

    for i, txt in enumerate(combo_legend.get_texts()):
        if i in header_rows:
            txt.set_weight("bold")
            txt.set_color(PALETTE["text_dark"])
            txt.set_fontsize(10.5)
        elif i in spacer_rows:
            txt.set_text("")
        else:
            txt.set_color(PALETTE["text_mid"])

    return fig


PVA_REQUIRED_COLUMNS = ["player_name", "total_passes", "total_pva"]
PASS_REQUIRED_COLUMNS = [
    "player_name", "start_x", "start_y", "end_x", "end_y",
    "pass_outcome", "xP",
]

BG_COLOR = "#0f172a"
LINE_COLOR = "#334155"
TITLE_COLOR = "#f8fafc"
SUBTITLE_COLOR = "#94a3b8"
COMPLETE_COLOR = "#10b981"
FAILED_COLOR = "#ef4444"


def _prepare_pva(pva_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the per-player PVA summary."""
    _check_columns(pva_df, PVA_REQUIRED_COLUMNS, "pva_df")
    pva_df = _to_numeric_strict(pva_df, ["total_passes", "total_pva"], "pva_df")

    n_before = len(pva_df)
    pva_df = pva_df.dropna(subset=["player_name", "total_passes", "total_pva"])
    _warn_dropped(n_before - len(pva_df), "pva_df")

    _require_non_negative_passes(pva_df, "pva_df")
    return pva_df


def _prepare_passes(passes_df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce dtypes and drop unusable rows."""
    _check_columns(passes_df, PASS_REQUIRED_COLUMNS, "passes_df")
    passes_df = passes_df.copy()

    numeric_cols = ["start_x", "start_y", "end_x", "end_y", "xP"]
    passes_df = _to_numeric_strict(passes_df, numeric_cols, "passes_df")
    passes_df = _to_numeric_strict(
        passes_df, ["pass_outcome"], "passes_df", hint=PASS_OUTCOME_CONTRACT
    )

    n_before = len(passes_df)
    passes_df = passes_df.dropna(subset=["player_name"] + numeric_cols + ["pass_outcome"])
    _warn_dropped(n_before - len(passes_df), "passes_df")
    if passes_df.empty:
        raise ValueError("`passes_df` has no usable rows after removing rows with missing values.")

    # Validation: pass outcome is strictly binary (1 = completed, 0 = incomplete/failed).
    invalid_outcome = ~passes_df["pass_outcome"].isin([0, 1])
    if invalid_outcome.any():
        bad_values = sorted(passes_df.loc[invalid_outcome, "pass_outcome"].unique().tolist())[:5]
        raise ValueError(
            f"`passes_df['pass_outcome']` contains unsupported values {bad_values}. "
            f"{PASS_OUTCOME_CONTRACT} Other values are not reinterpreted; recode them upstream."
        )

    _validate_rate_columns(passes_df, ["xP"], "passes_df")

    passes_df["pass_outcome"] = passes_df["pass_outcome"].astype(int)
    return passes_df.reset_index(drop=True)


def _select_top_player(
    pva_df: pd.DataFrame, passes_df: pd.DataFrame, min_passes: int
) -> tuple[str, float]:
    """Return (player_name, total_pva) for the highest total_pva player."""
    if pva_df.duplicated(subset=["player_name"]).any():
        raise ValueError("pva_df must be a pre-aggregated per-player summary without duplicate player names.")

    summary = pva_df.set_index("player_name")
    summary = summary[summary.index.isin(passes_df["player_name"].unique())]
    eligible = summary[summary["total_passes"] >= min_passes]

    if eligible.empty:
        biggest = int(summary["total_passes"].max()) if not summary.empty else 0
        raise ValueError(
            f"No player has >= {min_passes} passes (max among players found in "
            f"both dataframes: {biggest}). Lower `min_passes` and try again."
        )

    top_name = eligible["total_pva"].idxmax()
    return top_name, float(eligible.loc[top_name, "total_pva"])


def plot_top_player_pass_map(
    pva_df: pd.DataFrame,
    passes_df: pd.DataFrame,
    min_passes: int = 300,
    figsize: tuple[float, float] = (13, 9),
) -> Figure:
    """Plot every attempted pass of the top-PVA player on a StatsBomb pitch."""
    pva_data = _prepare_pva(pva_df)
    pass_data = _prepare_passes(passes_df)

    player_name, total_pva = _select_top_player(
        pva_data, pass_data, min_passes=min_passes
    )
    player_df = pass_data[pass_data["player_name"] == player_name].copy()

    n_passes = len(player_df)
    summary_passes = float(pva_data.loc[pva_data["player_name"] == player_name, "total_passes"].iloc[0])
    if n_passes != summary_passes:
        warnings.warn(
            f"{player_name}: `pva_df` reports {summary_passes:g} passes but `passes_df` has "
            f"{n_passes} usable pass rows; the map and the PVA summary may not cover the same passes.",
            UserWarning,
            stacklevel=2,
        )

    actual_pct = player_df["pass_outcome"].mean() * 100
    expected_pct = player_df["xP"].mean() * 100
    cpoe = actual_pct - expected_pct

    completed = player_df[player_df["pass_outcome"] == 1]
    failed = player_df[player_df["pass_outcome"] == 0]

    plt.rcParams["font.family"] = "DejaVu Sans"

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=BG_COLOR,
        line_color=LINE_COLOR,
        linewidth=1.6,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(BG_COLOR)

    if not completed.empty:
        pitch.arrows(
            completed["start_x"], completed["start_y"],
            completed["end_x"], completed["end_y"],
            ax=ax, color=COMPLETE_COLOR, alpha=0.70,
            width=1.6, headwidth=3, headlength=3, zorder=2,
        )
    if not failed.empty:
        pitch.arrows(
            failed["start_x"], failed["start_y"],
            failed["end_x"], failed["end_y"],
            ax=ax, color=FAILED_COLOR, alpha=0.90,
            width=1.8, headwidth=3, headlength=3, zorder=3,
        )

    pva_sign = "+" if total_pva >= 0 else "−"
    cpoe_sign = "+" if cpoe >= 0 else "−"

    ax.text(
        0.0, 1.11, f"{player_name} — Pass Map & Value Added (PVA)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=21, fontweight="bold", color=TITLE_COLOR,
    )
    subtitle = (
        f"{n_passes:,} passes  |  Completion {actual_pct:.1f}% vs. expected {expected_pct:.1f}%  |  "
        f"CPOE {cpoe_sign}{abs(cpoe):.1f} pp  |  Net PVA {pva_sign}{abs(total_pva):.2f}"
    )
    ax.text(
        0.0, 1.05, subtitle,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, color=SUBTITLE_COLOR,
    )

    ax.annotate(
        "", xy=(0.60, -0.035), xytext=(0.40, -0.035),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=SUBTITLE_COLOR, lw=1.5),
        annotation_clip=False,
    )
    ax.text(
        0.50, -0.06, "Attacking Direction",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color=SUBTITLE_COLOR,
    )

    handles = [
        Line2D([0], [0], color=COMPLETE_COLOR, lw=2.5, marker=">", markersize=8,
               markevery=[1], label="Completed Pass"),
        Line2D([0], [0], color=FAILED_COLOR, lw=2.5, marker=">", markersize=8,
               markevery=[1], label="Incomplete / Turnovers"),
    ]
    legend = ax.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.03),
        ncol=2, frameon=False, fontsize=10, handlelength=2.2, columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(SUBTITLE_COLOR)

    return fig