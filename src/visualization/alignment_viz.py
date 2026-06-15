from typing import List, Dict, Optional
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.amino_acids import AMINO_ACIDS, AMINO_ACID_INDEX, HYDROPHOBICITY
from ..prediction.base import STRUCTURE_COLORS, PredictionResult


HYDROPHOBIC = set("AVLIPFMW")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SPECIAL = set("GP")


COLOR_SCHEMES = {
    "chemical": "By Chemical Properties",
    "conservation": "By Conservation",
    "structure": "By Predicted Structure",
}


def _chemical_color(aa: str) -> str:
    if aa == "-":
        return "#FFFFFF"
    if aa in HYDROPHOBIC:
        return "#F39C12"
    if aa in POLAR:
        return "#3498DB"
    if aa in POSITIVE:
        return "#E74C3C"
    if aa in NEGATIVE:
        return "#9B59B6"
    if aa in SPECIAL:
        return "#2ECC71"
    return "#BDC3C7"


def _conservation_color(score: float) -> str:
    r = int(44 + (231 - 44) * (1 - score))
    g = int(62 + (76 - 62) * (1 - score))
    b = int(80 + (60 - 80) * (1 - score))
    return f"rgb({r}, {g}, {b})"


def _structure_color(state: str) -> str:
    return STRUCTURE_COLORS.get(state, "#BDC3C7")


def plot_alignment(
    aligned_names: List[str],
    aligned_sequences: List[str],
    consensus: str = "",
    column_scores: Optional[np.ndarray] = None,
    color_scheme: str = "chemical",
    structure_predictions: Optional[List[PredictionResult]] = None,
    start_col: int = 0,
    end_col: int = None,
) -> go.Figure:
    if not aligned_sequences:
        return go.Figure()

    n_seq = len(aligned_sequences)
    aln_len = len(aligned_sequences[0])

    if end_col is None:
        end_col = min(aln_len, start_col + 80)
    end_col = min(end_col, aln_len)

    n_rows = n_seq + (1 if consensus else 0)
    n_cols_display = end_col - start_col

    subplot_titles = aligned_names[:]
    if consensus:
        subplot_titles.append("Consensus")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    for row_idx, seq in enumerate(aligned_sequences):
        display_seq = seq[start_col:end_col]

        for col_offset, aa in enumerate(display_seq):
            col_idx = start_col + col_offset

            if color_scheme == "chemical":
                color = _chemical_color(aa)
            elif color_scheme == "conservation" and column_scores is not None:
                if col_idx < len(column_scores):
                    color = _conservation_color(column_scores[col_idx])
                else:
                    color = "#FFFFFF"
            elif color_scheme == "structure" and structure_predictions is not None:
                if row_idx < len(structure_predictions):
                    pred = structure_predictions[row_idx]
                    orig_pos = 0
                    for k in range(col_idx):
                        if seq[k] != "-":
                            orig_pos += 1
                    if orig_pos < len(pred.states):
                        color = _structure_color(pred.states[orig_pos])
                    else:
                        color = "#FFFFFF"
                else:
                    color = "#FFFFFF"
            else:
                color = "#FFFFFF"

            text_color = "#FFFFFF" if aa != "-" else "#BDC3C7"
            if aa in "GP":
                text_color = "#FFFFFF"

            fig.add_trace(
                go.Scatter(
                    x=[col_offset + 0.5],
                    y=[0.5],
                    mode="markers+text",
                    marker=dict(
                        size=28,
                        color=color,
                        line=dict(width=1, color="#2C3E50" if aa != "-" else "#ECF0F1"),
                    ),
                    text=[aa if aa != "-" else "·"],
                    textfont=dict(size=14, family="monospace", color=text_color),
                    textposition="middle center",
                    hovertemplate=f"Position: {col_idx + 1}<br>Residue: {aa if aa != '-' else 'gap'}",
                    showlegend=False,
                ),
                row=row_idx + 1,
                col=1,
            )

        fig.update_yaxes(
            showticklabels=False,
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            row=row_idx + 1,
            col=1,
        )

    if consensus:
        display_consensus = consensus[start_col:end_col]
        for col_offset, ch in enumerate(display_consensus):
            col_idx = start_col + col_offset
            if ch == ".":
                color = "#FFFFFF"
                text_color = "#BDC3C7"
                display_ch = "."
            elif ch.isupper():
                color = "#2C3E50"
                text_color = "#FFFFFF"
                display_ch = ch
            else:
                color = "#7F8C8D"
                text_color = "#FFFFFF"
                display_ch = ch.upper()

            fig.add_trace(
                go.Scatter(
                    x=[col_offset + 0.5],
                    y=[0.5],
                    mode="markers+text",
                    marker=dict(
                        size=28,
                        color=color,
                        line=dict(width=2, color="#F1C40F" if ch.isupper() else "#BDC3C7"),
                    ),
                    text=[display_ch],
                    textfont=dict(size=14, family="monospace", color=text_color, weight="bold"),
                    textposition="middle center",
                    hovertemplate=(
                        f"Position: {col_idx + 1}<br>"
                        f"Consensus: {'100% conserved' if ch.isupper() else '>50% conserved' if ch != '.' else 'variable'}"
                    ),
                    showlegend=False,
                ),
                row=n_rows,
                col=1,
            )

        fig.update_yaxes(
            showticklabels=False,
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            row=n_rows,
            col=1,
        )

    fig.update_xaxes(
        title_text=f"Alignment Position (showing {start_col + 1}-{end_col} of {aln_len})",
        range=[0, n_cols_display],
        tickmode="array",
        tickvals=[i + 0.5 for i in range(0, n_cols_display, max(1, n_cols_display // 20))],
        ticktext=[str(start_col + i + 1) for i in range(0, n_cols_display, max(1, n_cols_display // 20))],
        row=n_rows,
        col=1,
    )

    for r in range(1, n_rows + 1):
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            range=[0, n_cols_display],
            row=r,
            col=1,
        )

    fig.update_layout(
        height=80 + n_rows * 70,
        title=f"Multiple Sequence Alignment ({aln_len} positions)",
        showlegend=False,
        plot_bgcolor="#FAFAFA",
    )

    return fig


def plot_sequence_logo(
    aligned_sequences: List[str],
    start_col: int = 0,
    end_col: int = None,
    full: bool = False,
) -> go.Figure:
    if not aligned_sequences:
        return go.Figure()

    aln_len = len(aligned_sequences[0])
    n_seq = len(aligned_sequences)

    if end_col is None:
        if full:
            end_col = aln_len
        else:
            end_col = min(aln_len, start_col + 20)
    end_col = min(end_col, aln_len)

    display_len = end_col - start_col

    position_heights = []
    position_letters = []

    for col in range(start_col, end_col):
        counts: Dict[str, int] = {}
        n_valid = 0
        for seq in aligned_sequences:
            aa = seq[col]
            if aa != "-":
                counts[aa] = counts.get(aa, 0) + 1
                n_valid += 1

        if n_valid == 0:
            position_heights.append(0.0)
            position_letters.append([])
            continue

        entropy = 0.0
        freqs = {}
        for aa, count in counts.items():
            freq = count / n_valid
            freqs[aa] = freq
            if freq > 0:
                entropy -= freq * np.log2(freq)

        total_height = 2.0 - entropy
        sorted_aas = sorted(freqs.items(), key=lambda x: x[1])

        letters = []
        for aa, freq in sorted_aas:
            letters.append((aa, freq * total_height))

        position_heights.append(total_height)
        position_letters.append(letters)

    fig = go.Figure()

    for col_offset in range(display_len):
        col = start_col + col_offset
        y_base = 0.0
        for aa, height in position_letters[col_offset]:
            if height <= 0:
                continue

            color = _chemical_color(aa)
            fig.add_trace(
                go.Scatter(
                    x=[col_offset + 0.5],
                    y=[y_base + height / 2],
                    mode="text",
                    text=[aa],
                    textfont=dict(
                        size=int(8 + height * 18),
                        color=color,
                        family="monospace",
                        weight="bold",
                    ),
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=f"Position: {col + 1}<br>Residue: {aa}<br>Information: {height:.2f} bits",
                )
            )
            y_base += height

    fig.update_layout(
        title=f"Sequence Logo (positions {start_col + 1}-{end_col})",
        xaxis_title="Alignment Position",
        yaxis_title="Information Content (bits)",
        yaxis=dict(range=[0, 2.1], dtick=0.5, gridcolor="#ECF0F1"),
        xaxis=dict(
            range=[0, display_len],
            tickmode="array",
            tickvals=[i + 0.5 for i in range(0, display_len, max(1, display_len // 20))],
            ticktext=[str(start_col + i + 1) for i in range(0, display_len, max(1, display_len // 20))],
        ),
        height=350,
        plot_bgcolor="#FAFAFA",
    )

    return fig
