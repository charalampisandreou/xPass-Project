from __future__ import annotations

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


def draw_leaderboard_table(
    player_df: pd.DataFrame,
    top_n: int = 10,
    min_passes: int = 300
) -> plt.Figure:
    """Renders a dark-slate scouting leaderboard table."""
    df = player_df[player_df['total_passes'] >= min_passes].copy()
    
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
        'Top 10 Playmakers by Pass Value Added (PVA)', 
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


def _compute_thresholds(
    cpoe: pd.Series,
    elite_threshold: Optional[float],
    poor_threshold: Optional[float],
    elite_quantile: float,
    poor_quantile: float,
) -> Tuple[float, float]:
    """Resolves elite/poor CPOE cutoffs."""
    if elite_threshold is None:
        elite_threshold = float(cpoe.quantile(elite_quantile))
    if poor_threshold is None:
        poor_threshold = float(cpoe.quantile(poor_quantile))
    return elite_threshold, poor_threshold


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
    df = player_df[player_df['total_passes'] >= min_passes].copy()
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

    sizes = 90 + (df['total_passes'] / df['total_passes'].max()) * 900

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
            short_name = shorten_player_name(row['player_name'], name_overrides)
            t = ax.text(
                row['expected_completion_rate'], row['actual_completion_rate'] + 0.006,
                short_name, fontsize=10.5, weight="bold", color=PALETTE["text_dark"],
                ha="center", va="bottom", zorder=Z_LABELS,
                path_effects=glow_effect,
            )
            texts.append(t)

    if texts and _HAS_ADJUST_TEXT:
        adjust_text(
            texts, ax=ax,
            add_objects=[pill_elite, pill_poor],
            expand_points=(1.5, 1.7),
            expand_text=(1.15, 1.25),
            arrowprops=dict(arrowstyle="-", color=PALETTE["text_light"],
                             lw=0.9, alpha=0.8, shrinkA=5, shrinkB=5),
        )

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
    vol_min, vol_mid, vol_max = df['total_passes'].min(), df['total_passes'].median(), df['total_passes'].max()
    size_handles = [
        Line2D([0], [0], marker='o', linestyle='', markeredgecolor="white", alpha=0.85,
               markersize=np.sqrt(90 + (v / df['total_passes'].max()) * 900) * 0.62,
               markerfacecolor=PALETTE["text_light"], label=f"{int(v):,} passes")
        for v in (vol_min, vol_mid, vol_max)
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


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"`{name}` is missing required columns: {missing}")


def _prepare_pva(pva_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the per-player PVA summary."""
    _check_columns(pva_df, PVA_REQUIRED_COLUMNS, "pva_df")
    pva_df = pva_df.copy()
    for col in ["total_passes", "total_pva"]:
        pva_df[col] = pd.to_numeric(pva_df[col], errors="coerce")
    return pva_df.dropna(subset=["player_name", "total_passes", "total_pva"])


def _prepare_passes(passes_df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce dtypes and drop unusable rows."""
    _check_columns(passes_df, PASS_REQUIRED_COLUMNS, "passes_df")
    passes_df = passes_df.copy()

    numeric_cols = ["start_x", "start_y", "end_x", "end_y", "pass_outcome", "xP"]
    for col in numeric_cols:
        passes_df[col] = pd.to_numeric(passes_df[col], errors="coerce")

    passes_df = passes_df.dropna(subset=["player_name"] + numeric_cols)
    
    # Validation: Ensure pass outcome is strictly binary
    if not set(passes_df["pass_outcome"].unique()).issubset({0, 1}):
        raise ValueError("pass_outcome must contain only binary values (0 or 1).")
        
    passes_df["pass_outcome"] = passes_df["pass_outcome"].astype(int)
    return passes_df


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