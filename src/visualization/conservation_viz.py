from typing import List, Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..alignment.conservation import (
    ConservationResult,
    get_top_amino_acids,
    find_position_in_regions,
)


def plot_conservation_profile(
    conservation_result: ConservationResult,
    conserved_threshold: float = 0.3,
    variable_threshold: float = 0.7,
) -> go.Figure:
    n_cols = conservation_result.total_columns
    if n_cols == 0:
        return go.Figure()

    positions = list(range(1, n_cols + 1))
    shannon = conservation_result.shannon_entropy
    smoothed = conservation_result.smoothed_entropy

    fig = go.Figure()

    for reg in conservation_result.conserved_regions:
        fig.add_vrect(
            x0=reg["start"] + 1,
            x1=reg["end"] + 1,
            fillcolor="rgba(46, 204, 113, 0.25)",
            line=dict(color="rgba(46, 204, 113, 0.8)", width=1),
            layer="below",
            annotation_text=f"C{reg['id']}",
            annotation_position="top left",
            annotation_font=dict(color="#27AE60", size=10),
        )

    for reg in conservation_result.variable_regions:
        fig.add_vrect(
            x0=reg["start"] + 1,
            x1=reg["end"] + 1,
            fillcolor="rgba(231, 76, 60, 0.25)",
            line=dict(color="rgba(231, 76, 60, 0.8)", width=1),
            layer="below",
            annotation_text=f"V{reg['id']}",
            annotation_position="top left",
            annotation_font=dict(color="#C0392B", size=10),
        )

    hover_texts = []
    for i in range(n_cols):
        counts = conservation_result.column_amino_acid_counts[i]
        n_valid = sum(counts.values())
        top_aas = get_top_amino_acids(counts, n_valid, top_n=3)
        pos_info = find_position_in_regions(
            i, conservation_result.conserved_regions, conservation_result.variable_regions
        )

        aa_text = ""
        if top_aas:
            aa_parts = [f"{aa}: {pct:.1f}%" for aa, pct in top_aas]
            aa_text = "<br>".join(aa_parts)
        else:
            aa_text = "All gaps"

        region_text = ""
        if pos_info:
            rtype = "Conserved" if pos_info["type"] == "conserved" else "Variable Hotspot"
            reg = pos_info["region"]
            region_text = f"<br><b>{rtype} Region</b>: {reg['id']} (cols {reg['start']+1}-{reg['end']+1})"

        hover_text = (
            f"<b>Column {i+1}</b><br>"
            f"Shannon Entropy: {shannon[i]:.4f}<br>"
            f"Weighted Score: {conservation_result.weighted_score[i]:.4f}<br>"
            f"Smoothed Entropy: {smoothed[i]:.4f}<br>"
            f"<br><b>Top Amino Acids:</b><br>{aa_text}"
            f"{region_text}"
        )
        hover_texts.append(hover_text)

    fig.add_trace(go.Scatter(
        x=positions,
        y=shannon,
        mode="lines+markers",
        name="Shannon Entropy",
        line=dict(color="#3498DB", width=1.5),
        marker=dict(size=3, color="#3498DB"),
        hovertext=hover_texts,
        hoverinfo="text",
        opacity=0.7,
    ))

    fig.add_trace(go.Scatter(
        x=positions,
        y=smoothed,
        mode="lines",
        name="Smoothed (window avg)",
        line=dict(color="#E67E22", width=2.5),
        hoverinfo="skip",
    ))

    fig.add_hline(
        y=conserved_threshold,
        line_dash="dash",
        line_color="#27AE60",
        annotation_text=f"Conserved threshold ({conserved_threshold})",
        annotation_position="bottom right",
        annotation_font=dict(color="#27AE60"),
    )

    fig.add_hline(
        y=variable_threshold,
        line_dash="dash",
        line_color="#C0392B",
        annotation_text=f"Variable threshold ({variable_threshold})",
        annotation_position="top right",
        annotation_font=dict(color="#C0392B"),
    )

    fig.update_layout(
        title="Sequence Conservation Profile",
        xaxis_title="Alignment Position (column)",
        yaxis_title="Shannon Entropy (0=conserved, 1=variable)",
        yaxis=dict(range=[-0.05, 1.05], dtick=0.2, gridcolor="#ECF0F1"),
        xaxis=dict(gridcolor="#ECF0F1"),
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FAFAFA",
    )

    return fig


def plot_entropy_histogram(shannon_entropy: np.ndarray) -> go.Figure:
    if len(shannon_entropy) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=shannon_entropy,
        nbinsx=20,
        marker=dict(
            color="#3498DB",
            line=dict(color="#2980B9", width=1),
        ),
        opacity=0.8,
    ))

    fig.update_layout(
        title="Entropy Distribution",
        xaxis_title="Shannon Entropy",
        yaxis_title="Column Count",
        xaxis=dict(range=[0, 1], dtick=0.2, gridcolor="#ECF0F1"),
        yaxis=dict(gridcolor="#ECF0F1"),
        height=350,
        plot_bgcolor="#FAFAFA",
        bargap=0.05,
    )

    return fig
