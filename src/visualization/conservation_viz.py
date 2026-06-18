from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..alignment.conservation import (
    ConservationResult,
    get_top_amino_acids,
    find_position_in_regions,
)
from ..prediction.base import (
    PredictionResult,
    STRUCTURE_STATES,
    STRUCTURE_NAMES,
    STRUCTURE_COLORS,
)


def get_consensus_structure(
    predictions: List[PredictionResult],
) -> Tuple[List[str], List[bool], List[List[str]]]:
    if not predictions:
        return [], [], []

    seq_len = len(predictions[0])
    consensus = []
    is_agreement = []
    all_preds = []

    for pos in range(seq_len):
        preds_at_pos = [res.states[pos] for res in predictions]
        all_preds.append(preds_at_pos)
        unique_preds = set(preds_at_pos)
        if len(unique_preds) == 1:
            consensus.append(preds_at_pos[0])
            is_agreement.append(True)
        else:
            counter = Counter(preds_at_pos)
            most_common = counter.most_common(1)[0][0]
            consensus.append(most_common)
            is_agreement.append(False)

    return consensus, is_agreement, all_preds


def plot_conservation_profile(
    conservation_result: ConservationResult,
    conserved_threshold: float = 0.3,
    variable_threshold: float = 0.7,
    predictions: Optional[List[PredictionResult]] = None,
) -> go.Figure:
    n_cols = conservation_result.total_columns
    if n_cols == 0:
        return go.Figure()

    positions = list(range(1, n_cols + 1))
    shannon = conservation_result.shannon_entropy
    smoothed = conservation_result.smoothed_entropy

    consensus_struct = None
    struct_agreement = None
    struct_all_preds = None
    has_structure = predictions is not None and len(predictions) > 0

    if has_structure:
        consensus_struct, struct_agreement, struct_all_preds = get_consensus_structure(predictions)
        has_structure = len(consensus_struct) > 0

    if has_structure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.82, 0.18],
        )
    else:
        fig = go.Figure()

    target_fig = fig if not has_structure else fig

    for reg in conservation_result.conserved_regions:
        target_fig.add_vrect(
            x0=reg["start"] + 1,
            x1=reg["end"] + 1,
            fillcolor="rgba(46, 204, 113, 0.4)",
            line=dict(color="rgba(39, 174, 96, 1.0)", width=2),
            layer="below",
            annotation_text=f"C{reg['id']}",
            annotation_position="top left",
            annotation_font=dict(color="#27AE60", size=11, weight="bold"),
        )

    for reg in conservation_result.variable_regions:
        target_fig.add_vrect(
            x0=reg["start"] + 1,
            x1=reg["end"] + 1,
            fillcolor="rgba(231, 76, 60, 0.4)",
            line=dict(color="rgba(192, 57, 43, 1.0)", width=2),
            layer="below",
            annotation_text=f"V{reg['id']}",
            annotation_position="top left",
            annotation_font=dict(color="#C0392B", size=11, weight="bold"),
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

        struct_text = ""
        if has_structure and consensus_struct and i < len(consensus_struct):
            s = consensus_struct[i]
            agree = struct_agreement[i] if struct_agreement else False
            struct_name = STRUCTURE_NAMES.get(s, s)
            if agree:
                struct_text = f"<br><b>Structure</b>: {s} ({struct_name}) ✅ 3-method agreement"
            else:
                preds_str = ", ".join(struct_all_preds[i]) if struct_all_preds else "N/A"
                struct_text = f"<br><b>Structure</b>: {s} ({struct_name}) ❌ Disagreement [{preds_str}]"

        hover_text = (
            f"<b>Column {i+1}</b><br>"
            f"Shannon Entropy: {shannon[i]:.4f}<br>"
            f"Weighted Score: {conservation_result.weighted_score[i]:.4f}<br>"
            f"Smoothed Entropy: {smoothed[i]:.4f}<br>"
            f"<br><b>Top Amino Acids:</b><br>{aa_text}"
            f"{region_text}"
            f"{struct_text}"
        )
        hover_texts.append(hover_text)

    entropy_trace = go.Scatter(
        x=positions,
        y=shannon,
        mode="lines+markers",
        name="Shannon Entropy",
        line=dict(color="#3498DB", width=1.5),
        marker=dict(size=3, color="#3498DB"),
        hovertext=hover_texts,
        hoverinfo="text",
        opacity=0.7,
    )

    smoothed_trace = go.Scatter(
        x=positions,
        y=smoothed,
        mode="lines",
        name="Smoothed (window avg)",
        line=dict(color="#E67E22", width=2.5),
        hoverinfo="skip",
    )

    if has_structure:
        fig.add_trace(entropy_trace, row=1, col=1)
        fig.add_trace(smoothed_trace, row=1, col=1)
    else:
        fig.add_trace(entropy_trace)
        fig.add_trace(smoothed_trace)

    if has_structure:
        fig.add_hline(
            y=conserved_threshold,
            line_dash="dash",
            line_color="#27AE60",
            annotation_text=f"Conserved threshold ({conserved_threshold})",
            annotation_position="bottom right",
            annotation_font=dict(color="#27AE60"),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=variable_threshold,
            line_dash="dash",
            line_color="#C0392B",
            annotation_text=f"Variable threshold ({variable_threshold})",
            annotation_position="top right",
            annotation_font=dict(color="#C0392B"),
            row=1,
            col=1,
        )
    else:
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

    if has_structure and consensus_struct:
        for i in range(n_cols):
            if i >= len(consensus_struct):
                break
            s = consensus_struct[i]
            color = STRUCTURE_COLORS.get(s, "#95A5A6")
            agree = struct_agreement[i] if struct_agreement else False

            border_color = "#2C3E50" if not agree else color
            border_width = 1.5 if not agree else 0.5

            fig.add_trace(
                go.Bar(
                    x=[i + 1],
                    y=[1],
                    marker_color=color,
                    marker_line_color=border_color,
                    marker_line_width=border_width,
                    showlegend=False,
                    hoverinfo="skip",
                    name="",
                ),
                row=2,
                col=1,
            )

        fig.update_yaxes(
            showticklabels=False,
            range=[0, 1],
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Structure",
            row=2,
            col=1,
            title_font=dict(size=10),
        )

        legend_items = []
        for s in STRUCTURE_STATES:
            legend_items.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=STRUCTURE_COLORS[s]),
                    name=f"{s} - {STRUCTURE_NAMES[s]}",
                    showlegend=True,
                )
            )
        for item in legend_items:
            fig.add_trace(item, row=2, col=1)

    if has_structure:
        fig.update_yaxes(
            title_text="Shannon Entropy (0=conserved, 1=variable)",
            row=1,
            col=1,
            range=[-0.05, 1.05],
            dtick=0.2,
            gridcolor="#ECF0F1",
        )
        fig.update_xaxes(
            title_text="Alignment Position (column)",
            row=2,
            col=1,
            gridcolor="#ECF0F1",
        )
        total_height = 580
    else:
        fig.update_layout(
            yaxis_title="Shannon Entropy (0=conserved, 1=variable)",
            yaxis=dict(range=[-0.05, 1.05], dtick=0.2, gridcolor="#ECF0F1"),
            xaxis_title="Alignment Position (column)",
            xaxis=dict(gridcolor="#ECF0F1"),
        )
        total_height = 450

    fig.update_layout(
        title="Sequence Conservation Profile" + (" with Secondary Structure Overlay" if has_structure else ""),
        height=total_height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FAFAFA",
        barmode="stack",
        bargap=0.05,
    )

    return fig


def compute_structure_conservation_stats(
    conservation_result: ConservationResult,
    consensus_structure: List[str],
    conserved_threshold: float = 0.3,
    variable_threshold: float = 0.7,
) -> Dict[str, Dict]:
    stats = {}
    n_cols = min(conservation_result.total_columns, len(consensus_structure))

    for s in STRUCTURE_STATES:
        indices = [i for i in range(n_cols) if consensus_structure[i] == s]
        if not indices:
            stats[s] = {
                "count": 0,
                "avg_entropy": 0.0,
                "avg_weighted": 0.0,
                "pct_conserved": 0.0,
                "pct_variable": 0.0,
            }
            continue

        entropies = [conservation_result.shannon_entropy[i] for i in indices]
        weighted = [conservation_result.weighted_score[i] for i in indices]
        smoothed = [conservation_result.smoothed_entropy[i] for i in indices]

        n_conserved = sum(1 for v in smoothed if v < conserved_threshold)
        n_variable = sum(1 for v in smoothed if v > variable_threshold)

        stats[s] = {
            "count": len(indices),
            "avg_entropy": float(np.mean(entropies)),
            "avg_weighted": float(np.mean(weighted)),
            "pct_conserved": n_conserved / len(indices) * 100 if indices else 0.0,
            "pct_variable": n_variable / len(indices) * 100 if indices else 0.0,
        }

    return stats


def plot_structure_conservation_association(
    stats: Dict[str, Dict],
) -> go.Figure:
    fig = go.Figure()

    metric_configs = [
        ("avg_entropy", "Avg Shannon Entropy", "#3498DB", False),
        ("avg_weighted", "Avg Weighted Conservation", "#2ECC71", False),
        ("pct_conserved", "% in Conserved Regions", "#F39C12", True),
        ("pct_variable", "% in Variable Hotspots", "#E74C3C", True),
    ]

    x_labels = [f"{STRUCTURE_NAMES[s]} ({s})" for s in STRUCTURE_STATES]

    for metric_key, metric_name, color, is_percent in metric_configs:
        values = []
        for s in STRUCTURE_STATES:
            val = stats.get(s, {}).get(metric_key, 0.0)
            if is_percent:
                values.append(round(val, 1))
            else:
                values.append(round(val, 4))
        fig.add_trace(go.Bar(
            name=metric_name,
            x=x_labels,
            y=values,
            marker_color=color,
            text=[f"{v}{'%' if is_percent else ''}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Structure-Conservation Association Statistics",
        xaxis_title="Secondary Structure Type",
        yaxis_title="Value",
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FAFAFA",
        yaxis=dict(gridcolor="#ECF0F1"),
    )

    return fig


def find_structure_boundaries(
    consensus_structure: List[str],
    window: int = 2,
) -> set:
    boundaries = set()
    n = len(consensus_structure)
    for i in range(1, n):
        if consensus_structure[i] != consensus_structure[i - 1]:
            for j in range(max(0, i - window), min(n, i + window + 1)):
                boundaries.add(j)
    return boundaries


def detect_hotspot_boundary_overlap(
    conservation_result: ConservationResult,
    consensus_structure: List[str],
    boundary_window: int = 2,
    overlap_threshold: float = 0.5,
) -> List[Dict]:
    if not conservation_result.variable_regions or not consensus_structure:
        return []

    boundaries = find_structure_boundaries(consensus_structure, boundary_window)
    results = []

    for reg in conservation_result.variable_regions:
        start = reg["start"]
        end = reg["end"]
        region_len = end - start + 1
        overlap_positions = []
        for i in range(start, min(end + 1, len(consensus_structure))):
            if i in boundaries:
                overlap_positions.append(i + 1)

        overlap_count = len(overlap_positions)
        overlap_ratio = overlap_count / region_len if region_len > 0 else 0.0
        is_boundary_hotspot = overlap_ratio >= overlap_threshold

        results.append({
            "region_id": f"V{reg['id']}",
            "start_col": start + 1,
            "end_col": end + 1,
            "region_len": region_len,
            "overlap_count": overlap_count,
            "overlap_positions": overlap_positions,
            "overlap_ratio": overlap_ratio,
            "is_boundary_hotspot": is_boundary_hotspot,
        })

    return results


def get_region_structure_info(
    region: Dict,
    consensus_structure: List[str],
) -> Tuple[str, int]:
    start = region["start"]
    end = region["end"]
    structs = []
    for i in range(start, min(end + 1, len(consensus_structure))):
        structs.append(consensus_structure[i])

    if not structs:
        return "N/A", 0

    counter = Counter(structs)
    dominant = counter.most_common(1)[0][0]

    boundaries = find_structure_boundaries(consensus_structure, window=2)
    boundary_count = sum(1 for i in range(start, min(end + 1, len(consensus_structure))) if i in boundaries)

    return dominant, boundary_count


def plot_entropy_histogram(shannon_entropy: np.ndarray) -> go.Figure:
    if len(shannon_entropy) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=shannon_entropy,
        xbins=dict(
            start=0.0,
            end=1.0,
            size=0.05,
        ),
        marker=dict(
            color="#3498DB",
            line=dict(color="#2980B9", width=1),
        ),
        opacity=0.85,
    ))

    fig.update_layout(
        title="Entropy Distribution (20 bins)",
        xaxis_title="Shannon Entropy",
        yaxis_title="Column Count",
        xaxis=dict(range=[-0.02, 1.02], dtick=0.2, gridcolor="#ECF0F1"),
        yaxis=dict(gridcolor="#ECF0F1"),
        height=350,
        plot_bgcolor="#FAFAFA",
        bargap=0.05,
    )

    return fig
