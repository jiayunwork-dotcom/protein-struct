import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import json
import pickle
import tempfile
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.sequence import (
    Sequence,
    SequenceManager,
    parse_fasta_text,
    parse_fasta_file,
    fetch_uniprot_sequence,
    validate_sequence,
)
from src.prediction import (
    GOR4Predictor,
    NNPredictor,
    LSTMPredictor,
    PredictionResult,
    STRUCTURE_NAMES,
    STRUCTURE_COLORS,
    STRUCTURE_STATES,
)
from src.alignment import (
    needleman_wunsch,
    smith_waterman,
    format_alignment,
    progressive_alignment,
    AlignmentResult,
    MultipleAlignmentResult,
    analyze_conservation,
    ConservationResult,
)
from src.visualization import (
    plot_structure_prediction,
    plot_structure_composition,
    plot_helix_length_distribution,
    plot_helical_wheel,
    plot_ramachandran,
    plot_confidence_heatmap,
    plot_alignment,
    plot_sequence_logo,
    COLOR_SCHEMES,
    plot_conservation_profile,
    plot_entropy_histogram,
    get_consensus_structure,
    compute_structure_conservation_stats,
    plot_structure_conservation_association,
    detect_hotspot_boundary_overlap,
    get_region_structure_info,
)
from src.evaluation import (
    compute_q3,
    compute_sov,
    compute_per_state_metrics,
    evaluate_predictions,
)

st.set_page_config(
    page_title="Protein Structure Analysis Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "seq_manager" not in st.session_state:
        st.session_state.seq_manager = SequenceManager()
    if "gor_predictor" not in st.session_state:
        st.session_state.gor_predictor = GOR4Predictor()
    if "nn_predictor" not in st.session_state:
        st.session_state.nn_predictor = NNPredictor()
    if "lstm_predictor" not in st.session_state:
        st.session_state.lstm_predictor = LSTMPredictor()
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    if "pairwise_result" not in st.session_state:
        st.session_state.pairwise_result = None
    if "msa_result" not in st.session_state:
        st.session_state.msa_result = None
    if "heatmap_clicked_pos" not in st.session_state:
        st.session_state.heatmap_clicked_pos = {}
    if "conservation_selected_pos" not in st.session_state:
        st.session_state.conservation_selected_pos = []


def _on_heatmap_select(sid, heatmap_key):
    state = st.session_state.get(heatmap_key)
    if state is None:
        return
    selection = None
    if isinstance(state, dict):
        selection = state.get("selection")
    elif hasattr(state, "selection"):
        selection = state.selection
    if not selection:
        return

    points = None
    if isinstance(selection, dict):
        points = selection.get("points", [])
    elif hasattr(selection, "points"):
        points = selection.points
    if not points or len(points) == 0:
        return

    pt = points[0]
    pos_val = None
    if isinstance(pt, dict):
        pos_val = pt.get("x")
    elif hasattr(pt, "x"):
        pos_val = pt.x
    if pos_val is not None:
        st.session_state.heatmap_clicked_pos[sid] = int(pos_val)


def _on_conservation_select(conserv_key):
    state = st.session_state.get(conserv_key)
    if state is None:
        return
    selection = None
    if isinstance(state, dict):
        selection = state.get("selection")
    elif hasattr(state, "selection"):
        selection = state.selection
    if not selection:
        return

    points = None
    if isinstance(selection, dict):
        points = selection.get("points", [])
    elif hasattr(selection, "points"):
        points = selection.points
    if not points or len(points) == 0:
        return

    pos_vals = []
    for pt in points:
        pos_val = None
        if isinstance(pt, dict):
            pos_val = pt.get("x")
        elif hasattr(pt, "x"):
            pos_val = pt.x
        if pos_val is not None:
            pos_vals.append(int(pos_val))

    if pos_vals:
        st.session_state.conservation_selected_pos = sorted(set(pos_vals))


init_session_state()


def add_sequences(sequences: List[Sequence]):
    for seq in sequences:
        is_valid, invalid_pos = validate_sequence(seq.sequence)
        if not is_valid:
            invalid_chars = set(seq.sequence[p] for p in invalid_pos)
            st.warning(
                f"Sequence '{seq.name}' contains invalid characters: {', '.join(sorted(invalid_chars))}. "
                f"These will be ignored during processing."
            )
            cleaned = "".join(c for c in seq.sequence if c in "ACDEFGHIKLMNPQRSTVWY")
            seq.sequence = cleaned
        if len(seq.sequence) > 0:
            st.session_state.seq_manager.add(seq)
        else:
            st.error(f"Sequence '{seq.name}' is empty after cleaning and was not added.")


def sidebar_sequence_manager():
    with st.sidebar:
        st.header("🧬 Sequences")
        seq_manager = st.session_state.seq_manager
        all_seqs = seq_manager.get_all()

        if all_seqs:
            st.write(f"**Loaded: {len(all_seqs)} sequences**")
            for seq in all_seqs:
                with st.expander(f"📋 {seq.name} ({len(seq)} aa)"):
                    st.write(f"Source: {seq.source}")
                    if seq.description:
                        st.write(f"Description: {seq.description[:100]}...")
                    st.code(seq.sequence[:60] + ("..." if len(seq.sequence) > 60 else ""))
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"🗑️ Delete", key=f"del_{seq.id}"):
                            seq_manager.remove(seq.id)
                            st.rerun()
                    with col2:
                        if st.button(f"📋 Copy", key=f"copy_{seq.id}"):
                            st.code(f">{seq.name}\n{seq.sequence}")
        else:
            st.info("No sequences loaded yet.")

        st.divider()
        if st.button("🗑️ Clear All Sequences", use_container_width=True):
            seq_manager.clear()
            st.session_state.predictions.clear()
            st.session_state.pairwise_result = None
            st.session_state.msa_result = None
            st.rerun()


def page_sequence_input():
    st.header("📥 Sequence Input")
    st.markdown("Input protein sequences via FASTA text, file upload, or UniProt accession.")

    tab1, tab2, tab3 = st.tabs(["📝 Paste FASTA", "📁 Upload File", "🔍 UniProt Lookup"])

    with tab1:
        st.subheader("Paste FASTA Format Sequences")
        st.markdown("One or more sequences in FASTA format: `>name` followed by amino acid sequence.")
        fasta_text = st.text_area(
            "FASTA sequences:",
            height=200,
            placeholder=">seq1\nMKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGPEV...\n>seq2\nMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERG...",
        )
        col_paste1, col_paste2 = st.columns([1, 5])
        with col_paste1:
            if st.button("📥 Parse & Add", key="btn_paste", type="primary"):
                if fasta_text.strip():
                    seqs = parse_fasta_text(fasta_text, source="paste")
                    if seqs:
                        add_sequences(seqs)
                        st.success(f"Successfully loaded {len(seqs)} sequence(s)!")
                    else:
                        st.error("No valid FASTA sequences found.")
                else:
                    st.warning("Please paste some FASTA sequences first.")

    with tab2:
        st.subheader("Upload FASTA File")
        uploaded_file = st.file_uploader(
            "Choose a FASTA file (.fasta, .fa, .fna)",
            type=["fasta", "fa", "fna", "txt"],
        )
        if uploaded_file:
            st.write(f"📄 File: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            if st.button("📥 Load File", key="btn_upload", type="primary"):
                content = uploaded_file.getvalue()
                seqs = parse_fasta_file(content)
                if seqs:
                    add_sequences(seqs)
                    st.success(f"Successfully loaded {len(seqs)} sequence(s) from file!")
                else:
                    st.error("No valid sequences found in the file.")

    with tab3:
        st.subheader("Fetch from UniProt")
        st.markdown("Enter a UniProt accession number (e.g., P01308 for insulin).")
        uniprot_id = st.text_input(
            "UniProt Accession:",
            placeholder="e.g., P01308, P68871",
        )
        if st.button("🔍 Fetch Sequence", key="btn_uniprot", type="primary"):
            if uniprot_id.strip():
                with st.spinner(f"Fetching {uniprot_id} from UniProt..."):
                    seq = fetch_uniprot_sequence(uniprot_id)
                    if seq:
                        add_sequences([seq])
                        st.success(f"Successfully fetched {seq.name} ({len(seq)} aa)!")
                        st.code(f">{seq.name}\n{seq.sequence}")
                    else:
                        st.error(f"Could not fetch sequence for '{uniprot_id}'. Check the accession.")
            else:
                st.warning("Please enter a UniProt accession.")

    st.divider()
    st.subheader("✅ Validation")
    all_seqs = st.session_state.seq_manager.get_all()
    if all_seqs:
        for seq in all_seqs:
            is_valid, invalid_pos = validate_sequence(seq.sequence)
            if is_valid:
                st.success(f"✅ **{seq.name}**: Valid ({len(seq)} residues)")
            else:
                invalid_chars = sorted(set(seq.sequence[p] for p in invalid_pos))
                st.error(f"❌ **{seq.name}**: Invalid characters found: {', '.join(invalid_chars)} at positions {[p+1 for p in invalid_pos[:10]]}{'...' if len(invalid_pos) > 10 else ''}")
    else:
        st.info("Load some sequences to see validation results.")


def page_prediction():
    st.header("🔮 Secondary Structure Prediction")
    st.markdown("Predict α-helix (H), β-sheet (E), and coil (C) using three methods.")

    seq_manager = st.session_state.seq_manager
    all_seqs = seq_manager.get_all()

    if not all_seqs:
        st.warning("Please load sequences first in the 'Sequence Input' page.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        use_gor = st.checkbox("🧠 GOR-IV (Information Theory)", value=True)
    with col2:
        use_nn = st.checkbox("🔗 Neural Network", value=True)
    with col3:
        use_lstm = st.checkbox("📊 Bidirectional LSTM", value=True)

    if not any([use_gor, use_nn, use_lstm]):
        st.warning("Please select at least one prediction method.")
        return

    target_ids = st.multiselect(
        "Select sequences to predict:",
        options=[s.id for s in all_seqs],
        format_func=lambda sid: f"{seq_manager.get(sid).name} ({len(seq_manager.get(sid))} aa)",
        default=[s.id for s in all_seqs],
    )

    if not target_ids:
        st.warning("Please select at least one sequence.")
        return

    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
        st.session_state.predictions.clear()
        progress_bar = st.progress(0)
        total_steps = len(target_ids) * sum([use_gor, use_nn, use_lstm])
        step = 0

        for sid in target_ids:
            seq = seq_manager.get(sid)
            if not seq:
                continue
            st.session_state.predictions[sid] = []

            if use_gor:
                with st.spinner(f"Running GOR-IV on {seq.name}..."):
                    res = st.session_state.gor_predictor.predict(seq.sequence)
                    st.session_state.predictions[sid].append(res)
                step += 1
                progress_bar.progress(step / total_steps)

            if use_nn:
                with st.spinner(f"Running Neural Network on {seq.name}..."):
                    res = st.session_state.nn_predictor.predict(seq.sequence)
                    st.session_state.predictions[sid].append(res)
                step += 1
                progress_bar.progress(step / total_steps)

            if use_lstm:
                with st.spinner(f"Running LSTM on {seq.name}..."):
                    res = st.session_state.lstm_predictor.predict(seq.sequence)
                    st.session_state.predictions[sid].append(res)
                step += 1
                progress_bar.progress(step / total_steps)

        progress_bar.progress(1.0)
        st.success("Predictions complete!")

    if not st.session_state.predictions:
        return

    for sid, results in st.session_state.predictions.items():
        seq = seq_manager.get(sid)
        if not seq or not results:
            continue

        st.divider()
        st.subheader(f"📊 Results: {seq.name}")

        fig = plot_structure_prediction(results, sequence_name=seq.name)
        st.plotly_chart(fig, use_container_width=True, key=f"main_pred_{sid}")

        tab_vis1, tab_vis2, tab_vis3, tab_vis4, tab_vis5 = st.tabs([
            "📈 Composition", "📏 Helix Lengths", "🌀 Helical Wheel",
            "🎯 Ramachandran", "📋 Raw Output",
        ])

        with tab_vis1:
            for res in results:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        plot_structure_composition(res, seq.name),
                        use_container_width=True,
                        key=f"comp_{sid}_{res.method}",
                    )
                with col_b:
                    counts = {s: res.states.count(s) for s in STRUCTURE_STATES}
                    total = len(res.states) if res.states else 1
                    data = []
                    for s in STRUCTURE_STATES:
                        data.append({
                            "State": STRUCTURE_NAMES[s],
                            "Count": counts[s],
                            "Percentage": f"{counts[s] / total * 100:.1f}%",
                        })
                    st.markdown(f"**{res.method}**")
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        with tab_vis2:
            for res in results:
                st.markdown(f"**{res.method}**")
                st.plotly_chart(
                    plot_helix_length_distribution(res),
                    use_container_width=True,
                    key=f"helixlen_{sid}_{res.method}",
                )
                states = res.states
                helices = []
                start = None
                for i, s in enumerate(states):
                    if s == "H" and start is None:
                        start = i
                    elif s != "H" and start is not None:
                        helices.append((start + 1, i))
                        start = None
                if start is not None:
                    helices.append((start + 1, len(states)))

                if helices:
                    longest = max(helices, key=lambda x: x[1] - x[0])
                    st.info(f"💡 Longest helix: positions {longest[0]}-{longest[1]} ({longest[1]-longest[0]+1} residues)")

                beta_segments = []
                start = None
                for i, s in enumerate(states):
                    if s == "E" and start is None:
                        start = i
                    elif s != "E" and start is not None:
                        beta_segments.append((start + 1, i))
                        start = None
                if start is not None:
                    beta_segments.append((start + 1, len(states)))
                if beta_segments:
                    longest_beta = max(beta_segments, key=lambda x: x[1] - x[0])
                    st.info(f"💡 Longest β-sheet: positions {longest_beta[0]}-{longest_beta[1]} ({longest_beta[1]-longest_beta[0]+1} residues)")

        with tab_vis3:
            for res in results:
                st.markdown(f"**{res.method}**")
                st.plotly_chart(
                    plot_helical_wheel(res),
                    use_container_width=True,
                    key=f"hwheel_{sid}_{res.method}",
                )

        with tab_vis4:
            for res in results:
                st.markdown(f"**{res.method}**")
                st.plotly_chart(
                    plot_ramachandran(res),
                    use_container_width=True,
                    key=f"ramach_{sid}_{res.method}",
                )

        with tab_vis5:
            for res in results:
                with st.expander(f"📋 {res.method} - Raw Prediction"):
                    st.code(f"Sequence:  {res.sequence}")
                    st.code(f"Structure: {res.states}")
                    st.markdown("**Probabilities (H/E/C):**")
                    prob_data = []
                    for i, (aa, state) in enumerate(zip(res.sequence, res.states)):
                        prob_data.append({
                            "Position": i + 1,
                            "Residue": aa,
                            "Predicted": state,
                            "P(H)": f"{res.probabilities[i, 0]:.3f}",
                            "P(E)": f"{res.probabilities[i, 1]:.3f}",
                            "P(C)": f"{res.probabilities[i, 2]:.3f}",
                        })
                    st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

        if len(results) >= 2:
            with st.expander("📊 置信度分析 (Confidence Analysis)", expanded=True):
                hm_fig, conf_matrix, consensus_list, conf_stats = plot_confidence_heatmap(
                    results, sequence_name=seq.name
                )
                if hm_fig.data:
                    heatmap_key = f"heatmap_{sid}"
                    st.plotly_chart(
                        hm_fig,
                        use_container_width=True,
                        key=heatmap_key,
                        on_select=lambda s=sid, k=heatmap_key: _on_heatmap_select(s, k),
                        selection_mode="points",
                    )

                    stat_cols = st.columns(len(results) + 1)
                    for m_idx, res in enumerate(results):
                        with stat_cols[m_idx]:
                            avg_c = conf_stats["avg_confidences"].get(res.method, 0)
                            st.metric(
                                label=res.method,
                                value=f"{avg_c:.3f}",
                                delta="avg confidence",
                            )
                    with stat_cols[len(results)]:
                        st.metric(
                            label="3-Method Agreement",
                            value=f"{conf_stats['agreement_rate']:.1f}%",
                            delta=f"{conf_stats['agreement_count']}/{conf_stats['seq_len']} positions",
                        )

                    st.markdown("---")
                    st.markdown(
                        "**🔬 Residue Detail Explorer** — Click any cell in the heatmap above, "
                        "or use the slider below to fine-tune position."
                    )
                    seq_len = len(results[0])
                    default_pos = st.session_state.heatmap_clicked_pos.get(sid, 1)

                    selected_pos = st.slider(
                        "Residue Position:",
                        min_value=1,
                        max_value=seq_len,
                        value=default_pos if 1 <= default_pos <= seq_len else 1,
                        key=f"conf_pos_{sid}",
                    )

                    if sid in st.session_state.heatmap_clicked_pos:
                        clicked = st.session_state.heatmap_clicked_pos[sid]
                        st.info(f"🖱️ Last clicked position: **{clicked}** | Slider value: **{selected_pos}**")

                    center = selected_pos - 1
                    ctx_start = max(0, center - 5)
                    ctx_end = min(seq_len, center + 6)
                    local_seq = results[0].sequence[ctx_start:ctx_end]

                    st.markdown(
                        f"**Local Sequence** (positions {ctx_start + 1}–{ctx_end}): "
                        f"`{''.join(local_seq)}`"
                    )

                    detail_rows = []
                    for offset_i, pos_i in enumerate(range(ctx_start, ctx_end)):
                        row = {
                            "Residue": results[0].sequence[pos_i],
                            "Position": pos_i + 1,
                        }
                        for res in results:
                            h_p = res.probabilities[pos_i, 0] * 100
                            e_p = res.probabilities[pos_i, 1] * 100
                            c_p = res.probabilities[pos_i, 2] * 100
                            row[f"{res.method}"] = (
                                f"{res.states[pos_i]} "
                                f"(H:{h_p:.1f}% E:{e_p:.1f}% C:{c_p:.1f}%)"
                            )
                        consensus_char = consensus_list[pos_i]
                        if consensus_char != "?":
                            row["Consensus"] = f"✅ {consensus_char}"
                        else:
                            row["Consensus"] = "❓ ?"
                        is_center = (pos_i == center)
                        if is_center:
                            for k in row:
                                row[k] = f"👉 {row[k]}"
                        detail_rows.append(row)

                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )


def page_pairwise_alignment():
    st.header("🔗 Pairwise Sequence Alignment")
    st.markdown("Global (Needleman-Wunsch) or local (Smith-Waterman) alignment with BLOSUM62 scoring.")

    seq_manager = st.session_state.seq_manager
    all_seqs = seq_manager.get_all()

    if len(all_seqs) < 2:
        st.warning("Please load at least 2 sequences first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        seq1_id = st.selectbox(
            "Sequence 1:",
            options=[s.id for s in all_seqs],
            format_func=lambda sid: f"{seq_manager.get(sid).name} ({len(seq_manager.get(sid))} aa)",
            index=0,
        )
    with col2:
        seq2_id = st.selectbox(
            "Sequence 2:",
            options=[s.id for s in all_seqs],
            format_func=lambda sid: f"{seq_manager.get(sid).name} ({len(seq_manager.get(sid))} aa)",
            index=1 if len(all_seqs) > 1 else 0,
        )

    st.subheader("⚙️ Alignment Parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gap_open = st.slider("Gap Open Penalty", -20.0, 0.0, -10.0, 0.5)
    with col_b:
        gap_extend = st.slider("Gap Extension Penalty", -5.0, 0.0, -0.5, 0.1)
    with col_c:
        algo = st.selectbox("Algorithm", ["Global (Needleman-Wunsch)", "Local (Smith-Waterman)"])

    if st.button("🔍 Align Sequences", type="primary"):
        seq1 = seq_manager.get(seq1_id)
        seq2 = seq_manager.get(seq2_id)
        if seq1 and seq2:
            with st.spinner("Running alignment..."):
                if "Global" in algo:
                    result = needleman_wunsch(seq1.sequence, seq2.sequence, gap_open, gap_extend)
                else:
                    result = smith_waterman(seq1.sequence, seq2.sequence, gap_open, gap_extend)
                st.session_state.pairwise_result = (seq1, seq2, result)

    if st.session_state.pairwise_result:
        seq1, seq2, result = st.session_state.pairwise_result
        st.divider()
        st.subheader("📊 Alignment Results")
        st.success(f"Alignment Score: **{result.score:.1f}**")

        if not result.is_global:
            st.info(
                f"Local match: {seq1.name} [{result.start1}-{result.end1}] vs "
                f"{seq2.name} [{result.start2}-{result.end2}]"
            )

        aligned_text = format_alignment(result)
        a1 = result.seq1_aligned
        a2 = result.seq2_aligned
        middle = []
        for c1, c2 in zip(a1, a2):
            if c1 == c2 and c1 != "-":
                middle.append("|")
            elif c1 != "-" and c2 != "-":
                middle.append(".")
            else:
                middle.append(" ")
        middle_str = "".join(middle)

        line_width = 80
        total_len = len(a1)

        n_lines = (total_len + line_width - 1) // line_width
        for line_idx in range(n_lines):
            start = line_idx * line_width
            end = min(start + line_width, total_len)

            st.markdown(f"**Positions {start + 1}-{end}:**")
            st.code(
                f"{seq1.name:15s} {a1[start:end]}\n"
                f"{'':15s} {middle_str[start:end]}\n"
                f"{seq2.name:15s} {a2[start:end]}"
            )

        matches = sum(1 for c1, c2 in zip(a1, a2) if c1 == c2 and c1 != "-")
        mismatches = sum(1 for c1, c2 in zip(a1, a2) if c1 != c2 and c1 != "-" and c2 != "-")
        gaps1 = a1.count("-")
        gaps2 = a2.count("-")
        aligned_pos = sum(1 for c1, c2 in zip(a1, a2) if c1 != "-" and c2 != "-")
        identity = matches / aligned_pos * 100 if aligned_pos > 0 else 0

        stats_data = [
            {"Metric": "Matches", "Value": matches},
            {"Metric": "Mismatches", "Value": mismatches},
            {"Metric": "Gaps (seq1)", "Value": gaps1},
            {"Metric": "Gaps (seq2)", "Value": gaps2},
            {"Metric": "Identity", "Value": f"{identity:.1f}%"},
            {"Metric": "Aligned Length", "Value": len(a1)},
        ]
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)


def page_msa():
    st.header("📊 Multiple Sequence Alignment")
    st.markdown("Progressive alignment using UPGMA guide tree and profile-profile alignment.")

    seq_manager = st.session_state.seq_manager
    all_seqs = seq_manager.get_all()

    if len(all_seqs) < 2:
        st.warning("Please load at least 2 sequences first.")
        return

    target_ids = st.multiselect(
        "Select sequences to align:",
        options=[s.id for s in all_seqs],
        format_func=lambda sid: f"{seq_manager.get(sid).name} ({len(seq_manager.get(sid))} aa)",
        default=[s.id for s in all_seqs],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        gap_open = st.slider("Gap Open Penalty", -20.0, 0.0, -10.0, 0.5, key="msa_go")
    with col_b:
        gap_extend = st.slider("Gap Extension Penalty", -5.0, 0.0, -0.5, 0.1, key="msa_ge")

    if st.button("🚀 Run MSA", type="primary") and target_ids:
        names = []
        sequences = []
        for sid in target_ids:
            s = seq_manager.get(sid)
            if s:
                names.append(s.name)
                sequences.append(s.sequence)

        if len(names) >= 2:
            with st.spinner("Running progressive multiple alignment..."):
                result = progressive_alignment(names, sequences, gap_open, gap_extend)
                st.session_state.msa_result = result
                st.success(f"Alignment complete! {result.alignment_length} columns.")
        else:
            st.warning("Please select at least 2 sequences.")

    if st.session_state.msa_result is None:
        return

    result: MultipleAlignmentResult = st.session_state.msa_result
    st.divider()

    col_scheme = st.selectbox(
        "Coloring Scheme:",
        options=list(COLOR_SCHEMES.keys()),
        format_func=lambda k: COLOR_SCHEMES[k],
        index=0,
    )

    total_cols = result.alignment_length
    view_start = st.slider(
        "View Window Start:",
        0, max(0, total_cols - 1), 0,
        key="view_start",
    )
    view_end = st.slider(
        "View Window End:",
        view_start + 1, total_cols, min(view_start + 80, total_cols),
        key="view_end",
    )

    struct_preds = []
    if col_scheme == "structure" and st.session_state.predictions:
        for name in result.names:
            for sid, preds in st.session_state.predictions.items():
                s = seq_manager.get(sid)
                if s and s.name == name and preds:
                    struct_preds.append(preds[0])
                    break
            else:
                struct_preds.append(None)

    fig = plot_alignment(
        result.names,
        result.aligned_sequences,
        result.consensus,
        result.column_scores,
        color_scheme=col_scheme,
        structure_predictions=struct_preds if struct_preds else None,
        start_col=view_start,
        end_col=view_end,
    )
    st.plotly_chart(fig, use_container_width=True)

    show_logo = st.checkbox("Show Sequence Logo", value=False)
    if show_logo:
        logo_full = st.checkbox("Show Full Logo", value=False)
        logo_fig = plot_sequence_logo(
            result.aligned_sequences,
            start_col=view_start,
            end_col=view_end,
            full=logo_full,
        )
        st.plotly_chart(logo_fig, use_container_width=True)

    st.subheader("📋 Export Alignment")
    fasta_out = ""
    for name, aln in zip(result.names, result.aligned_sequences):
        fasta_out += f">{name}\n{aln}\n"
    if result.consensus:
        fasta_out += f">consensus\n{result.consensus}\n"

    st.download_button(
        "⬇️ Download FASTA",
        data=fasta_out,
        file_name="alignment.fasta",
        mime="text/plain",
    )


def page_evaluation():
    st.header("📐 Batch Prediction & Evaluation")
    st.markdown("Upload test sequences with known structures and compare method performance.")

    st.subheader("📁 Upload Test Set")
    st.markdown(
        "Each test case should have: `>name` followed by the amino acid sequence on one line, "
        "then the true secondary structure (H/E/C) on the next line."
    )
    test_text = st.text_area(
        "Test set (FASTA-like with structure annotation):",
        height=200,
        placeholder=">protein1\nMKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGPEV\nCCCHHHHHHHHHHHHCCCCCEEEEEEECCCCCCCHHHHHHHHCC...\n>protein2\nMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERG\nCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHEEEEEEEECCCCCCCC...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        eval_gor = st.checkbox("Evaluate GOR-IV", value=True)
    with col2:
        eval_nn = st.checkbox("Evaluate Neural Network", value=True)
    with col3:
        eval_lstm = st.checkbox("Evaluate LSTM", value=True)

    if st.button("🚀 Evaluate All Methods", type="primary") and test_text.strip():
        lines = test_text.strip().splitlines()
        test_cases = []
        current_name = None
        current_seq = None
        current_struct = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name and current_seq and current_struct:
                    test_cases.append((current_name, current_seq, current_struct))
                current_name = line[1:]
                current_seq = None
                current_struct = None
            elif current_seq is None:
                current_seq = line.upper()
            elif current_struct is None:
                current_struct = line.upper()

        if current_name and current_seq and current_struct:
            test_cases.append((current_name, current_seq, current_struct))

        if not test_cases:
            st.error("No valid test cases found.")
            return

        st.info(f"Loaded {len(test_cases)} test cases.")

        all_results = []
        progress = st.progress(0)
        total = len(test_cases) * sum([eval_gor, eval_nn, eval_lstm])
        step = 0

        method_preds_map = {}

        if eval_gor:
            method_preds_map["GOR-IV"] = []
        if eval_nn:
            method_preds_map["Neural Network"] = []
        if eval_lstm:
            method_preds_map["Bidirectional LSTM"] = []

        actual_states_all = []

        for name, seq, actual in test_cases:
            actual_clean = "".join(c for c in actual if c in "HEC")
            actual_states_all.append(actual_clean)

            if eval_gor:
                pred = st.session_state.gor_predictor.predict(seq)
                method_preds_map["GOR-IV"].append(pred)
                step += 1
                progress.progress(step / total)

            if eval_nn:
                pred = st.session_state.nn_predictor.predict(seq)
                method_preds_map["Neural Network"].append(pred)
                step += 1
                progress.progress(step / total)

            if eval_lstm:
                pred = st.session_state.lstm_predictor.predict(seq)
                method_preds_map["Bidirectional LSTM"].append(pred)
                step += 1
                progress.progress(step / total)

        progress.progress(1.0)

        comparison_data = []
        all_method_results = {}

        for method, preds in method_preds_map.items():
            eval_result = evaluate_predictions(preds, actual_states_all)
            all_method_results[method] = eval_result

            row = {
                "Method": method,
                "Q3 Accuracy": f"{eval_result['q3_mean'] * 100:.2f} ± {eval_result['q3_std'] * 100:.2f}%",
                "SOV Score": f"{eval_result['sov_mean'] * 100:.2f} ± {eval_result['sov_std'] * 100:.2f}%",
            }
            for s in STRUCTURE_STATES:
                ps = eval_result["per_state"][s]
                row[f"{s} Precision"] = f"{ps['precision_mean'] * 100:.1f}%"
                row[f"{s} Recall"] = f"{ps['recall_mean'] * 100:.1f}%"
            comparison_data.append(row)

        st.subheader("📊 Performance Comparison")
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        fig_q3 = go.Figure()
        methods_list = list(method_preds_map.keys())
        q3_means = [all_method_results[m]["q3_mean"] * 100 for m in methods_list]
        q3_stds = [all_method_results[m]["q3_std"] * 100 for m in methods_list]

        fig_q3.add_trace(go.Bar(
            x=methods_list,
            y=q3_means,
            error_y=dict(type="data", array=q3_stds, visible=True),
            marker_color=["#E74C3C", "#3498DB", "#2ECC71"][:len(methods_list)],
            text=[f"{v:.1f}%" for v in q3_means],
            textposition="outside",
        ))
        fig_q3.update_layout(
            title="Q3 Accuracy Comparison",
            yaxis_title="Q3 Accuracy (%)",
            yaxis_range=[0, 100],
            height=400,
        )
        st.plotly_chart(fig_q3, use_container_width=True)

        fig_metrics = go.Figure()
        for s in STRUCTURE_STATES:
            fig_metrics.add_trace(go.Bar(
                name=STRUCTURE_NAMES[s],
                x=methods_list,
                y=[all_method_results[m]["per_state"][s]["recall_mean"] * 100 for m in methods_list],
                marker_color=STRUCTURE_COLORS[s],
            ))
        fig_metrics.update_layout(
            title="Per-State Recall (%)",
            yaxis_title="Recall (%)",
            yaxis_range=[0, 100],
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.subheader("📋 Per-Protein Results")
        per_protein = []
        for idx, (name, seq, actual) in enumerate(test_cases):
            for method, preds in method_preds_map.items():
                pred = preds[idx]
                q3 = compute_q3(pred.states, actual_states_all[idx])
                sov = compute_sov(pred.states, actual_states_all[idx])
                per_protein.append({
                    "Protein": name,
                    "Method": method,
                    "Length": len(seq),
                    "Q3": f"{q3 * 100:.1f}%",
                    "SOV": f"{sov * 100:.1f}%",
                })
        st.dataframe(pd.DataFrame(per_protein), use_container_width=True, hide_index=True)


def generate_regions_csv(
    conservation_result: ConservationResult,
    consensus_structure: Optional[List[str]] = None,
) -> str:
    rows = []
    has_struct = consensus_structure is not None and len(consensus_structure) > 0

    for reg in conservation_result.conserved_regions:
        row = {
            "region_id": f"C{reg['id']}",
            "type": "conserved",
            "start_col": reg["start"] + 1,
            "end_col": reg["end"] + 1,
            "span": reg["length"],
            "avg_entropy": f"{reg['avg_entropy']:.4f}",
            "max_entropy": f"{reg['max_entropy']:.4f}",
            "min_entropy": f"{reg['min_entropy']:.4f}",
            "consensus_sequence": reg["consensus_sequence"],
        }
        if has_struct:
            dominant, boundary_count = get_region_structure_info(reg, consensus_structure)
            row["dominant_structure"] = dominant
            row["structure_boundary_overlap_count"] = boundary_count
        rows.append(row)

    for reg in conservation_result.variable_regions:
        row = {
            "region_id": f"V{reg['id']}",
            "type": "variable",
            "start_col": reg["start"] + 1,
            "end_col": reg["end"] + 1,
            "span": reg["length"],
            "avg_entropy": f"{reg['avg_entropy']:.4f}",
            "max_entropy": f"{reg['max_entropy']:.4f}",
            "min_entropy": f"{reg['min_entropy']:.4f}",
            "consensus_sequence": reg["consensus_sequence"],
        }
        if has_struct:
            dominant, boundary_count = get_region_structure_info(reg, consensus_structure)
            row["dominant_structure"] = dominant
            row["structure_boundary_overlap_count"] = boundary_count
        rows.append(row)

    if not rows:
        base_header = "region_id,type,start_col,end_col,span,avg_entropy,max_entropy,min_entropy,consensus_sequence"
        if has_struct:
            base_header += ",dominant_structure,structure_boundary_overlap_count"
        return base_header + "\n"
    return pd.DataFrame(rows).to_csv(index=False)


def page_conservation():
    st.header("🛡️ Conservation Analysis")
    st.markdown(
        "Analyze sequence conservation and variable hotspots from a multiple sequence alignment (MSA). "
        "Computes Shannon entropy, BLOSUM62-weighted conservation scores, and identifies conserved regions and mutation hotspots."
    )

    if st.session_state.msa_result is None:
        st.warning("⚠️ Please run Multiple Sequence Alignment (MSA) first in the '📊 Multiple Alignment' page.")
        return

    result: MultipleAlignmentResult = st.session_state.msa_result
    seq_manager = st.session_state.seq_manager

    st.subheader("⚙️ Parameters")
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        window_size = st.slider(
            "Smoothing Window Size",
            min_value=1,
            max_value=21,
            value=7,
            step=2,
            help="Sliding window size for entropy smoothing (odd number recommended)",
        )
    with col_param2:
        conserved_threshold = st.slider(
            "Conserved Region Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Smoothed entropy below this value is considered conserved",
        )
    with col_param3:
        variable_threshold = st.slider(
            "Variable Hotspot Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Smoothed entropy above this value is considered a variable hotspot",
        )

    if variable_threshold <= conserved_threshold:
        st.error("❌ Variable threshold must be greater than conserved threshold.")
        return

    prediction_results = None
    consensus_structure = None
    struct_agreement = None
    struct_all_preds = None

    if st.session_state.predictions:
        first_sid = None
        for sid in st.session_state.predictions:
            first_sid = sid
            break
        if first_sid and st.session_state.predictions[first_sid]:
            prediction_results = st.session_state.predictions[first_sid]
            consensus_structure, struct_agreement, struct_all_preds = get_consensus_structure(prediction_results)

    with st.spinner("Analyzing conservation..."):
        conservation_result = analyze_conservation(
            result.aligned_sequences,
            window_size=window_size,
            conserved_threshold=conserved_threshold,
            variable_threshold=variable_threshold,
        )

    st.divider()
    st.subheader("📈 Conservation Profile")
    conserv_key = "conservation_profile"
    profile_fig = plot_conservation_profile(
        conservation_result,
        conserved_threshold=conserved_threshold,
        variable_threshold=variable_threshold,
        predictions=prediction_results,
    )
    st.plotly_chart(
        profile_fig,
        use_container_width=True,
        key=conserv_key,
        on_select=lambda k=conserv_key: _on_conservation_select(k),
        selection_mode="points",
    )

    if st.session_state.conservation_selected_pos:
        st.markdown("#### 🔍 Selected Position Details")
        selected = st.session_state.conservation_selected_pos
        center = selected[0] - 1
        ctx_start = max(0, center - 5)
        ctx_end = min(conservation_result.total_columns, center + 6)

        local_seq = "".join(
            conservation_result.column_consensus[i]
            for i in range(ctx_start, ctx_end)
        )
        st.markdown(
            f"**Local Sequence** (positions {ctx_start + 1}–{ctx_end}): "
            f"`{local_seq}`"
        )

        if consensus_structure:
            local_struct = "".join(
                consensus_structure[i] if i < len(consensus_structure) else "-"
                for i in range(ctx_start, ctx_end)
            )
            st.markdown(
                f"**Predicted Structure** (H=helix, E=sheet, C=coil): "
                f"`{local_struct}`"
            )

        detail_rows = []
        for offset_i, pos_i in enumerate(range(ctx_start, ctx_end)):
            row = {
                "Position": pos_i + 1,
                "Residue": conservation_result.column_consensus[pos_i],
                "Shannon Entropy": f"{conservation_result.shannon_entropy[pos_i]:.4f}",
                "Weighted Score": f"{conservation_result.weighted_score[pos_i]:.4f}",
            }
            if consensus_structure and pos_i < len(consensus_structure):
                s = consensus_structure[pos_i]
                agree = struct_agreement[pos_i] if struct_agreement else False
                if agree:
                    row["Structure"] = f"{s} ✅"
                else:
                    preds_str = "/".join(struct_all_preds[pos_i]) if struct_all_preds else "?"
                    row["Structure"] = f"{s} ❌ ({preds_str})"
            is_center = (pos_i == center)
            if is_center:
                for k in row:
                    row[k] = f"👉 {row[k]}"
            detail_rows.append(row)

        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("🎯 Sequence Logo")
    logo_full = st.checkbox("Show Full Alignment Logo", value=False)
    if logo_full:
        logo_fig = plot_sequence_logo(
            result.aligned_sequences,
            full=True,
        )
    else:
        logo_view_start = st.slider(
            "Logo View Start:",
            0, max(0, conservation_result.total_columns - 1), 0,
            key="logo_view_start",
        )
        logo_fig = plot_sequence_logo(
            result.aligned_sequences,
            start_col=logo_view_start,
            end_col=min(logo_view_start + 40, conservation_result.total_columns),
            full=False,
        )
    st.plotly_chart(logo_fig, use_container_width=True)

    st.divider()
    st.subheader("📊 Statistical Summary")

    stat_col1, stat_col2 = st.columns([1, 1])

    with stat_col1:
        st.markdown("#### 📋 Basic Statistics")
        total_cols = conservation_result.total_columns
        valid_cols = conservation_result.valid_columns
        valid_entropies = conservation_result.shannon_entropy[
            [i for i, c in enumerate(conservation_result.column_amino_acid_counts) if sum(c.values()) > 0]
        ]
        avg_entropy = float(np.mean(valid_entropies)) if len(valid_entropies) > 0 else 0.0
        median_entropy = float(np.median(valid_entropies)) if len(valid_entropies) > 0 else 0.0

        total_conserved_span = sum(r["length"] for r in conservation_result.conserved_regions)
        total_variable_span = sum(r["length"] for r in conservation_result.variable_regions)

        metrics_data = [
            {"Metric": "Total Columns", "Value": str(total_cols)},
            {"Metric": "Valid Columns (non-all-gap)", "Value": f"{valid_cols} ({valid_cols/total_cols*100:.1f}%)" if total_cols > 0 else str(valid_cols)},
            {"Metric": "Average Shannon Entropy", "Value": f"{avg_entropy:.4f}"},
            {"Metric": "Median Shannon Entropy", "Value": f"{median_entropy:.4f}"},
            {"Metric": "Conserved Regions", "Value": f"{len(conservation_result.conserved_regions)} regions, {total_conserved_span} cols"},
            {"Metric": "Variable Hotspots", "Value": f"{len(conservation_result.variable_regions)} regions, {total_variable_span} cols"},
        ]
        st.dataframe(pd.DataFrame(metrics_data).astype(str), use_container_width=True, hide_index=True)

        if conservation_result.conserved_regions:
            st.markdown("#### 🟢 Conserved Regions")
            conserved_data = []
            for reg in conservation_result.conserved_regions:
                row = {
                    "ID": f"C{reg['id']}",
                    "Columns": f"{reg['start']+1}-{reg['end']+1}",
                    "Span": str(reg["length"]),
                    "Avg Entropy": f"{reg['avg_entropy']:.4f}",
                    "Consensus": reg["consensus_sequence"],
                }
                if consensus_structure:
                    dominant, bcount = get_region_structure_info(reg, consensus_structure)
                    row["Dominant Struct"] = dominant
                    row["Boundary Overlaps"] = str(bcount)
                conserved_data.append(row)
            st.dataframe(pd.DataFrame(conserved_data).astype(str), use_container_width=True, hide_index=True)

        if conservation_result.variable_regions:
            st.markdown("#### 🔴 Variable Hotspots")
            variable_data = []
            for reg in conservation_result.variable_regions:
                row = {
                    "ID": f"V{reg['id']}",
                    "Columns": f"{reg['start']+1}-{reg['end']+1}",
                    "Span": str(reg["length"]),
                    "Avg Entropy": f"{reg['avg_entropy']:.4f}",
                    "Consensus": reg["consensus_sequence"],
                }
                if consensus_structure:
                    dominant, bcount = get_region_structure_info(reg, consensus_structure)
                    row["Dominant Struct"] = dominant
                    row["Boundary Overlaps"] = str(bcount)
                variable_data.append(row)
            st.dataframe(pd.DataFrame(variable_data).astype(str), use_container_width=True, hide_index=True)

    with stat_col2:
        hist_fig = plot_entropy_histogram(valid_entropies)
        st.plotly_chart(hist_fig, use_container_width=True)

    if consensus_structure:
        st.divider()
        st.subheader("🧬 Structure-Conservation Association")

        struct_stats = compute_structure_conservation_stats(
            conservation_result,
            consensus_structure,
            conserved_threshold=conserved_threshold,
            variable_threshold=variable_threshold,
        )

        assoc_col1, assoc_col2 = st.columns([1, 1])
        with assoc_col1:
            assoc_table_data = []
            for s in STRUCTURE_STATES:
                stat = struct_stats.get(s, {})
                assoc_table_data.append({
                    "Structure": f"{STRUCTURE_NAMES[s]} ({s})",
                    "Count": stat.get("count", 0),
                    "Avg Entropy": f"{stat.get('avg_entropy', 0.0):.4f}",
                    "Avg Weighted": f"{stat.get('avg_weighted', 0.0):.4f}",
                    "% Conserved": f"{stat.get('pct_conserved', 0.0):.1f}%",
                    "% Variable": f"{stat.get('pct_variable', 0.0):.1f}%",
                })
            st.dataframe(pd.DataFrame(assoc_table_data), use_container_width=True, hide_index=True)

        with assoc_col2:
            assoc_fig = plot_structure_conservation_association(struct_stats)
            st.plotly_chart(assoc_fig, use_container_width=True)

        with st.expander("🔺 Hotspot - Structure Boundary Analysis", expanded=True):
            boundary_results = detect_hotspot_boundary_overlap(
                conservation_result,
                consensus_structure,
                boundary_window=2,
                overlap_threshold=0.5,
            )

            if not boundary_results:
                st.info("No variable hotspots detected or no structure data available.")
            else:
                boundary_hotspots = [b for b in boundary_results if b["is_boundary_hotspot"]]
                if boundary_hotspots:
                    st.warning(
                        f"⚠️ **{len(boundary_hotspots)} Boundary-Associated Hotspot(s)** detected "
                        f"(≥50% positions near structure transitions)"
                    )

                boundary_table_data = []
                for br in boundary_results:
                    positions_str = (
                        ", ".join(str(p) for p in br["overlap_positions"][:10])
                        + ("..." if len(br["overlap_positions"]) > 10 else "")
                    ) if br["overlap_positions"] else "-"
                    label = "✅ Boundary Hotspot" if br["is_boundary_hotspot"] else "No"
                    boundary_table_data.append({
                        "Hotspot": br["region_id"],
                        "Columns": f"{br['start_col']}-{br['end_col']}",
                        "Region Length": br["region_len"],
                        "Overlap Count": br["overlap_count"],
                        "Overlap %": f"{br['overlap_ratio']*100:.1f}%",
                        "Boundary-Associated": label,
                        "Overlap Positions": positions_str,
                    })
                st.dataframe(pd.DataFrame(boundary_table_data), use_container_width=True, hide_index=True)

    st.divider()
    csv_data = generate_regions_csv(conservation_result, consensus_structure)
    st.download_button(
        "⬇️ Export Regions as CSV",
        data=csv_data,
        file_name="conservation_regions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    st.title("🧬 Protein Structure Analysis Toolkit")
    st.markdown(
        "A comprehensive tool for **protein secondary structure prediction** and "
        "**sequence alignment analysis**."
    )

    sidebar_sequence_manager()

    st.sidebar.divider()
    page = st.sidebar.radio(
        "📑 Navigation",
        [
            "📥 Sequence Input",
            "🔮 Structure Prediction",
            "🔗 Pairwise Alignment",
            "📊 Multiple Alignment",
            "🛡️ Conservation Analysis",
            "📐 Batch Evaluation",
        ],
    )

    if page.startswith("📥"):
        page_sequence_input()
    elif page.startswith("🔮"):
        page_prediction()
    elif page.startswith("🔗"):
        page_pairwise_alignment()
    elif page.startswith("📊"):
        page_msa()
    elif page.startswith("🛡️"):
        page_conservation()
    elif page.startswith("📐"):
        page_evaluation()

    st.sidebar.divider()
    st.sidebar.markdown(
        "---\n"
        "💡 **Tips:**\n"
        "- Use UniProt accessions like P01308 (insulin)\n"
        "- Valid amino acids: ACDEFGHIKLMNPQRSTVWY\n"
        "- Click sequence names in sidebar to see details\n"
    )


if __name__ == "__main__":
    main()
