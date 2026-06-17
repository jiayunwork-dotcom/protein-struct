from typing import List, Dict, Optional, Tuple
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..prediction.base import (
    PredictionResult,
    STRUCTURE_STATES,
    STRUCTURE_NAMES,
    STRUCTURE_COLORS,
)
from ..data.amino_acids import HYDROPHOBICITY


def plot_structure_prediction(
    results: List[PredictionResult],
    sequence_name: str = "",
) -> go.Figure:
    if not results:
        return go.Figure()

    n_methods = len(results)
    seq_len = len(results[0])

    fig = make_subplots(
        rows=n_methods + 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4] + [0.2] * n_methods + [0.4],
        subplot_titles=["Sequence"] + [r.method for r in results] + ["Predicted State"],
    )

    for i, res in enumerate(results):
        row_idx = i + 2
        colors = [STRUCTURE_COLORS[s] for s in res.states]

        for pos in range(seq_len):
            fig.add_trace(
                go.Bar(
                    x=[pos + 1],
                    y=[1],
                    marker_color=colors[pos],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Position {pos + 1}</b><br>"
                        f"Residue: {res.sequence[pos]}<br>"
                        f"State: {res.states[pos]} ({STRUCTURE_NAMES[res.states[pos]]})<br>"
                        f"P(H)={res.probabilities[pos, 0]:.3f}<br>"
                        f"P(E)={res.probabilities[pos, 1]:.3f}<br>"
                        f"P(C)={res.probabilities[pos, 2]:.3f}"
                    ),
                    name="",
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(
            showticklabels=False,
            range=[0, 1],
            row=row_idx,
            col=1,
        )

    for pos in range(seq_len):
        fig.add_trace(
            go.Scatter(
                x=[pos + 1],
                y=[0.5],
                mode="text",
                text=[results[0].sequence[pos]],
                textfont=dict(size=14, family="monospace", color="#2C3E50"),
                showlegend=False,
                hoverinfo="none",
            ),
            row=1,
            col=1,
        )

    for i, res in enumerate(results):
        for pos in range(seq_len):
            fig.add_trace(
                go.Scatter(
                    x=[pos + 1],
                    y=[0.5],
                    mode="text",
                    text=[res.states[pos]],
                    textfont=dict(size=12, family="monospace", color=STRUCTURE_COLORS[res.states[pos]]),
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=n_methods + 2,
                col=1,
            )

    for row_idx in [1, n_methods + 2]:
        fig.update_yaxes(
            showticklabels=False,
            range=[0, 1],
            row=row_idx,
            col=1,
        )

    fig.update_xaxes(
        title_text="Residue Position",
        row=n_methods + 2,
        col=1,
        dtick=max(1, seq_len // 20),
    )

    title = f"Secondary Structure Prediction"
    if sequence_name:
        title += f" - {sequence_name}"

    fig.update_layout(
        height=200 + 80 * n_methods,
        title=title,
        showlegend=False,
        barmode="stack",
        bargap=0.05,
    )

    return fig


def plot_structure_composition(
    result: PredictionResult,
    sequence_name: str = "",
) -> go.Figure:
    states = result.states
    counts = {s: states.count(s) for s in STRUCTURE_STATES}
    total = len(states) if states else 1
    percentages = {s: c / total * 100 for s, c in counts.items()}

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[STRUCTURE_NAMES[s] for s in STRUCTURE_STATES],
                values=[counts[s] for s in STRUCTURE_STATES],
                marker_colors=[STRUCTURE_COLORS[s] for s in STRUCTURE_STATES],
                textinfo="label+percent",
                hovertemplate="%{label}<br>Count: %{value}<br>Percent: %{percent}",
            )
        ]
    )

    title = "Secondary Structure Composition"
    if sequence_name:
        title += f" - {sequence_name}"

    fig.update_layout(title=title, height=400)
    return fig


def plot_helix_length_distribution(
    result: PredictionResult,
) -> go.Figure:
    states = result.states
    lengths = []
    current = 0

    for s in states:
        if s == "H":
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)

    if not lengths:
        lengths = [0]

    max_len = max(lengths) if lengths else 5
    bins = list(range(1, max_len + 2))

    fig = go.Figure(
        data=[
            go.Histogram(
                x=lengths,
                nbinsx=max(5, max_len),
                marker_color=STRUCTURE_COLORS["H"],
                marker_line_color="#2C3E50",
                marker_line_width=1,
            )
        ]
    )

    fig.update_layout(
        title="α-Helix Segment Length Distribution",
        xaxis_title="Helix Length (residues)",
        yaxis_title="Count",
        height=350,
        bargap=0.1,
    )

    return fig


def _get_longest_beta_sheet(result: PredictionResult) -> Optional[Tuple[int, int]]:
    states = result.states
    best_start = -1
    best_end = -1
    best_len = 0
    current_start = -1
    current_len = 0

    for i, s in enumerate(states):
        if s == "E":
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = i
        else:
            current_len = 0
            current_start = -1

    if best_len > 0:
        return (best_start + 1, best_end + 1)
    return None


def plot_helical_wheel(
    result: PredictionResult,
    helix_start: Optional[int] = None,
    helix_end: Optional[int] = None,
) -> go.Figure:
    states = result.states
    sequence = result.sequence

    if helix_start is None or helix_end is None:
        helices = []
        current_start = None
        for i, s in enumerate(states):
            if s == "H" and current_start is None:
                current_start = i
            elif s != "H" and current_start is not None:
                helices.append((current_start, i - 1))
                current_start = None
        if current_start is not None:
            helices.append((current_start, len(states) - 1))

        if not helices:
            fig = go.Figure()
            fig.update_layout(
                title="Helical Wheel - No α-helix segments found",
                height=400,
                showlegend=False,
            )
            return fig

        helices.sort(key=lambda x: x[1] - x[0], reverse=True)
        start, end = helices[0]
    else:
        start, end = helix_start - 1, helix_end - 1

    helix_seq = sequence[start:end + 1]

    angle_per_residue = (2 * np.pi) / 3.6

    points_x, points_y = [], []
    colors = []
    labels = []
    hover_texts = []

    for i, aa in enumerate(helix_seq):
        angle = i * angle_per_residue
        radius = 1.0 + i * 0.08
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points_x.append(x)
        points_y.append(y)

        hydrophobicity = HYDROPHOBICITY.get(aa, 0)
        if hydrophobicity > 0:
            color = "#E67E22"
        else:
            color = "#3498DB"

        colors.append(color)
        labels.append(aa)
        hover_texts.append(
            f"Position: {start + i + 1}<br>"
            f"Residue: {aa}<br>"
            f"Hydrophobicity: {hydrophobicity:.1f}"
        )

    fig = go.Figure()

    for i in range(len(helix_seq)):
        fig.add_trace(
            go.Scatter(
                x=[points_x[i]],
                y=[points_y[i]],
                mode="markers+text",
                marker=dict(
                    size=32,
                    color=colors[i],
                    line=dict(width=2, color="#2C3E50"),
                ),
                text=[labels[i]],
                textfont=dict(size=14, color="white", family="monospace"),
                textposition="middle center",
                hovertext=[hover_texts[i]],
                hoverinfo="text",
                showlegend=False,
            )
        )

    circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))
    circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="#BDC3C7", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="none",
        )
    )

    max_radius = max(max(points_x), max(points_y), 1) * 1.3

    fig.update_layout(
        title=f"Helical Wheel (residues {start + 1}-{end + 1})",
        xaxis=dict(range=[-max_radius, max_radius], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-max_radius, max_radius], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        height=500,
        width=500,
        showlegend=True,
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=16, color="#E67E22"),
            name="Hydrophobic",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=16, color="#3498DB"),
            name="Hydrophilic",
        )
    )

    return fig


def plot_confidence_heatmap(
    results: List[PredictionResult],
    sequence_name: str = "",
) -> Tuple[go.Figure, np.ndarray, List[str], Dict]:
    if not results or len(results) < 2:
        return go.Figure(), np.zeros((0, 0)), [], {}

    seq_len = len(results[0])
    n_methods = len(results)
    method_names = [r.method for r in results]

    confidence_matrix = np.zeros((n_methods, seq_len), dtype=np.float64)
    for i, res in enumerate(results):
        sorted_probs = np.sort(res.probabilities, axis=1)
        confidence_matrix[i] = sorted_probs[:, -1] - sorted_probs[:, -2]

    consensus = []
    for pos in range(seq_len):
        preds_at_pos = [res.states[pos] for res in results]
        if len(set(preds_at_pos)) == 1:
            consensus.append(preds_at_pos[0])
        else:
            consensus.append("?")

    agreement_count = sum(1 for c in consensus if c != "?")
    agreement_rate = agreement_count / seq_len * 100 if seq_len > 0 else 0

    avg_confidences = confidence_matrix.mean(axis=1).tolist()

    stats = {
        "avg_confidences": dict(zip(method_names, avg_confidences)),
        "agreement_rate": agreement_rate,
        "agreement_count": agreement_count,
        "seq_len": seq_len,
    }

    y_labels = method_names + ["Consensus"]
    z_data = np.vstack([confidence_matrix, np.zeros((1, seq_len))])

    positions = list(range(1, seq_len + 1))

    hover_text = []
    for row_idx in range(n_methods):
        row_text = []
        for pos in range(seq_len):
            aa = results[0].sequence[pos]
            lines = [f"<b>Position {pos + 1} | {aa}</b>"]
            for m_idx, res in enumerate(results):
                h_pct = res.probabilities[pos, 0] * 100
                e_pct = res.probabilities[pos, 1] * 100
                c_pct = res.probabilities[pos, 2] * 100
                lines.append(
                    f"{res.method}: H={h_pct:.1f}% E={e_pct:.1f}% C={c_pct:.1f}% → {res.states[pos]}"
                )
            lines.append(f"Confidence: {confidence_matrix[row_idx, pos]:.3f}")
            row_text.append("<br>".join(lines))
        hover_text.append(row_text)

    consensus_hover = []
    for pos in range(seq_len):
        aa = results[0].sequence[pos]
        lines = [f"<b>Position {pos + 1} | {aa}</b>"]
        for m_idx, res in enumerate(results):
            lines.append(f"{res.method}: {res.states[pos]}")
        lines.append(f"Consensus: {consensus[pos]}")
        consensus_hover.append("<br>".join(lines))
    hover_text.append(consensus_hover)

    colorscale = [
        [0.0, "#FFFFFF"],
        [0.15, "#F2E6FF"],
        [0.35, "#D4A8FF"],
        [0.55, "#B366FF"],
        [0.75, "#8B2FC9"],
        [1.0, "#4A0E6B"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=positions,
            y=y_labels,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            hoverinfo="text",
            text=hover_text,
            showscale=True,
            colorbar=dict(
                title=dict(text="Confidence", side="right"),
                tickmode="linear",
                tick0=0,
                dtick=0.2,
                len=0.75,
            ),
            xgap=1,
            ygap=2,
        )
    )

    consensus_colors_map = []
    for pos in range(seq_len):
        if consensus[pos] != "?":
            consensus_colors_map.append("#2ECC71")
        else:
            consensus_colors_map.append("#E67E22")

    for pos in range(seq_len):
        bg_color = consensus_colors_map[pos]
        text_color = "white" if consensus[pos] == "?" else "#1A1A2E"
        fig.add_annotation(
            x=pos + 1,
            y="Consensus",
            text=consensus[pos],
            showarrow=False,
            font=dict(size=11, family="monospace", color=text_color),
            bgcolor=bg_color,
            bordercolor="#2C3E50",
            borderwidth=1,
            borderpad=2,
        )

    stats_text_lines = ["<b>Statistics</b>"]
    for m_idx, m_name in enumerate(method_names):
        avg_c = avg_confidences[m_idx]
        stats_text_lines.append(f"{m_name}: avg {avg_c:.3f}")
    stats_text_lines.append(f"Agreement: {agreement_rate:.1f}%")

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.12,
        y=0.5,
        text="<br>".join(stats_text_lines),
        showarrow=False,
        font=dict(size=11, family="sans-serif"),
        align="left",
        bordercolor="#BDC3C7",
        borderwidth=1,
        borderpad=6,
        bgcolor="#F8F9FA",
    )

    title = "Confidence Heatmap"
    if sequence_name:
        title += f" - {sequence_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Residue Position",
        yaxis_title="Method",
        height=100 + 70 * (n_methods + 1),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        xaxis=dict(
            dtick=max(1, seq_len // 30),
            tickfont=dict(size=10),
        ),
        margin=dict(r=140),
    )

    return fig, confidence_matrix, consensus, stats


def plot_ramachandran(result: PredictionResult) -> go.Figure:
    states = result.states

    phi_psi = {
        "H": {"phi": [], "psi": []},
        "E": {"phi": [], "psi": []},
        "C": {"phi": [], "psi": []},
    }

    np.random.seed(42)

    for i, s in enumerate(states):
        if s == "H":
            phi = np.random.normal(-60, 12)
            psi = np.random.normal(-50, 12)
        elif s == "E":
            phi = np.random.normal(-120, 15)
            psi = np.random.normal(120, 15)
        else:
            phi = np.random.uniform(-180, 180)
            psi = np.random.uniform(-180, 180)

        phi_psi[s]["phi"].append(phi)
        phi_psi[s]["psi"].append(psi)

    fig = go.Figure()

    for state in STRUCTURE_STATES:
        fig.add_trace(
            go.Scatter(
                x=phi_psi[state]["phi"],
                y=phi_psi[state]["psi"],
                mode="markers",
                name=STRUCTURE_NAMES[state],
                marker=dict(
                    color=STRUCTURE_COLORS[state],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color="#2C3E50"),
                ),
                hovertemplate="φ=%{x:.1f}°<br>ψ=%{y:.1f}°<br>State: " + STRUCTURE_NAMES[state],
            )
        )

    fig.update_layout(
        title="Ramachandran Plot (Estimated)",
        xaxis_title="φ Angle (degrees)",
        yaxis_title="ψ Angle (degrees)",
        xaxis=dict(range=[-180, 180], dtick=60, gridcolor="#ECF0F1"),
        yaxis=dict(range=[-180, 180], dtick=60, gridcolor="#ECF0F1"),
        height=500,
        width=500,
        legend_title="Secondary Structure",
    )

    return fig
