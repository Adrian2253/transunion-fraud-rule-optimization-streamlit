"""
Interactive Fraud Rule Optimization Dashboard
----------------------------------------------
Streamlit app with four sections:
  1. Overview  -- dataset, business problem, team
  2. Algorithm Explorer  -- filter variants by model, operators, hyperparams
  3. Best Rules per Model  -- side-by-side rulesets & metrics
  4. Pareto Front  -- notebook-style Pareto visualizations + operating points

Pareto front logic adapted from notebooks/results_analysis_and_visualizations.ipynb
(pareto_front_mask, draw_pareto_step, plot_group helpers).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Rule Optimization -- Interactive",
    page_icon=None,
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT_DIR / "results" / "all_model_results.csv"
SUMMARY_JSON = ROOT_DIR / "results" / "results_summary.json"
DATA_CSV = ROOT_DIR / "data" / "sample_synthetic_with_time.csv"

ALGO_COLORS = {
    "Baseline GA": "#D32F2F",
    "Coevolution GA": "#FF6F00",
    "NSGA-II": "#1565C0",
    "Greedy Builder": "#2E7D32",
}

# ── Helpers ─────────────────────────────────────────────────────

def _safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    numeric_cols = [
        "train_precision", "train_recall", "train_f1",
        "val_precision", "val_recall", "val_f1",
        "test_precision", "test_recall", "test_f1",
        "train_alert_rate", "val_alert_rate", "test_alert_rate",
        "coverage_lambda", "parsimony_mu", "pop_size", "cx_prob",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_summary() -> dict:
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data_sample(path: str) -> pd.DataFrame:
    try:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def parse_rules(rule_text: str) -> list[str]:
    if not isinstance(rule_text, str) or not rule_text.strip():
        return []
    lines = [ln.strip() for ln in rule_text.splitlines() if ln.strip()]
    if lines:
        return lines
    return [chunk.strip() for chunk in rule_text.split("Rule") if chunk.strip()]


# ── Pareto helpers (adapted from results_analysis_and_visualizations.ipynb) ──

def pareto_front_mask(prec, rec):
    """Return boolean mask of non-dominated points (notebook: pareto_front_mask)."""
    pts = np.column_stack([np.asarray(prec, float), np.asarray(rec, float)])
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]
                    and (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                mask[i] = False
                break
    return mask


def pareto_step_line(prec_vals, rec_vals):
    """Sort by recall and return (recall, precision) for a step-line trace
    (notebook: draw_pareto_step)."""
    pts = sorted(zip(rec_vals, prec_vals))
    return [p[0] for p in pts], [p[1] for p in pts]


def _add_model_traces(fig, df, label, color, p_col, r_col, f_col):
    """Add scatter dots, best-F1 star, mean square, and Pareto line for one group.
    (Plotly equivalent of notebook plot_group helper)."""
    if df.empty:
        return

    # All dots
    fig.add_trace(go.Scatter(
        x=df[r_col], y=df[p_col],
        mode="markers",
        marker=dict(size=8, color=color, opacity=0.55,
                    line=dict(width=0.3, color="white")),
        name=f"{label} (n={len(df)})",
        text=df["variant"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"Precision: %{{y:.4f}}<br>Recall: %{{x:.4f}}<br>"
            f"F1: %{{customdata[0]:.4f}}<extra></extra>"
        ),
        customdata=df[[f_col]].values,
    ))

    # Best F1 star
    best_idx = df[f_col].idxmax()
    best_row = df.loc[best_idx]
    fig.add_trace(go.Scatter(
        x=[best_row[r_col]], y=[best_row[p_col]],
        mode="markers",
        marker=dict(size=18, color=color, symbol="star",
                    line=dict(width=1, color="black")),
        showlegend=False,
        hovertemplate=(
            f"<b>{label} Best F1</b><br>"
            f"F1: {best_row[f_col]:.4f}<extra></extra>"
        ),
    ))

    # Group mean square
    fig.add_trace(go.Scatter(
        x=[df[r_col].mean()], y=[df[p_col].mean()],
        mode="markers",
        marker=dict(size=14, color=color, symbol="square",
                    line=dict(width=1, color="black")),
        showlegend=False,
        hovertemplate=(
            f"<b>{label} Mean</b><br>"
            f"Precision: {df[p_col].mean():.4f}<br>"
            f"Recall: {df[r_col].mean():.4f}<extra></extra>"
        ),
    ))

    # Pareto step line
    if len(df) > 1:
        pf = pareto_front_mask(df[p_col].values, df[r_col].values)
        if pf.sum() > 1:
            rec_line, prec_line = pareto_step_line(
                df[p_col].values[pf], df[r_col].values[pf],
            )
            fig.add_trace(go.Scatter(
                x=rec_line, y=prec_line,
                mode="lines",
                line=dict(color=color, width=2),
                opacity=0.75,
                showlegend=False,
            ))


# ── Load data ───────────────────────────────────────────────────
results_df = load_results()
summary = load_summary()

# ── Title ───────────────────────────────────────────────────────
st.title("Fraud Rule Optimization -- Interactive Explorer")
st.caption(
    "UIC College of Business Administration  |  TransUnion Capstone  |  "
    "Pareto logic adapted from results_analysis_and_visualizations.ipynb"
)

# ── Sidebar navigation ─────────────────────────────────────────
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Algorithm Explorer",
        "Best Rules per Model",
        "Pareto Front",
    ],
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        data_sample = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        data_sample = load_data_sample(str(DATA_CSV))
else:
    data_sample = load_data_sample(str(DATA_CSV))


# ════════════════════════════════════════════════════════════════
# SECTION 1: Overview
# ════════════════════════════════════════════════════════════════
if section == "Overview":
    st.header("Overview")
    st.markdown(
        "Financial institutions rely on manual IF-THEN rules to flag suspicious "
        "transactions. As fraud evolves, these rules quickly become outdated. "
        "We use **Genetic Algorithms** to automatically discover and optimize "
        "fraud detection rules that are human-readable, auditable, and "
        "immediately deployable."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Business Problem")
        st.markdown(
            "**Stakeholders:** TransUnion (sponsor), financial institution clients, "
            "fraud analysts.\n\n"
            "**Constraints:**\n"
            "- Rules must be human-readable -- no black boxes\n"
            "- Alert rate must stay realistic (<0.5% of transactions flagged)\n"
            "- Must be compatible with existing rule engine infrastructure\n"
            "- Research uses synthetic data only -- no access to production data"
        )
    with col_b:
        st.subheader("Analytical Framing")
        st.markdown(
            "Binary classification: each transaction is fraudulent or legitimate. "
            "Instead of a black-box ML model, we find the combination of IF-THEN "
            "conditions that best separates fraud from legitimate transactions, "
            "subject to complexity limits.\n\n"
            "**Primary metric:** F1 Score (balance of precision and recall)\n\n"
            "**Alert Rate:** Share of total transactions flagged (<0.5% target)"
        )

    st.markdown("---")

    # ── Dataset ─────────────────────────────────────────────────
    st.subheader("Dataset at a Glance")
    n_rows = f"{len(data_sample):,}" if not data_sample.empty else "N/A"
    n_feats = str(len(data_sample.columns)) if not data_sample.empty else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", n_rows)
    col2.metric("Features", n_feats)
    col3.metric("Fraud Rate", "0.5%")
    col4.metric("Selected Features (MI)", "12")

    config = summary.get("config", {})
    top_feats = config.get("top_features", [])
    if top_feats:
        if st.button("Show MI-Selected Features", key="mi_overview_btn"):
            for i, f in enumerate(top_feats, 1):
                st.markdown(f"{i}. `{f}`")

    if not data_sample.empty:
        class_col = "Fraud_Label" if "Fraud_Label" in data_sample.columns else "Class"
        if class_col in data_sample.columns:
            counts = data_sample[class_col].value_counts()
            total = counts.sum()
            legit = counts.get(0, 0)
            fraud = counts.get(1, 0)
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("Legitimate", f"{legit:,}", f"{legit / total:.1%}")
            cd2.metric("Fraudulent", f"{fraud:,}", f"{fraud / total:.1%}")
            cd3.metric("Total", f"{total:,}")

        with st.expander("Sample Rows"):
            st.dataframe(data_sample.head(15), use_container_width=True)

    st.markdown("---")
    st.subheader("Team Members")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.markdown("**Debangana Sanyal**\n\ndsany@uic.edu")
    t2.markdown("**Siddhi Jain**\n\nsjain213@uic.edu")
    t3.markdown("**Sam Chyu**\n\nschyu2@uic.edu")
    t4.markdown("**Anand Mathur**\n\namath56@uic.edu")
    t5.markdown("**Adrian Garces**\n\nagarce3@uic.edu")
    st.caption("Faculty Advisor: Prof. Fatemeh Sarayloo (fsaraylo@uic.edu)")


# ════════════════════════════════════════════════════════════════
# SECTION 2: Algorithm Explorer
# ════════════════════════════════════════════════════════════════
elif section == "Algorithm Explorer":
    st.header("Algorithm Explorer")
    st.write(
        "Filter and compare all 195 algorithm variants interactively. "
        "Choose which models to display, which operators to include, "
        "and how many top features were used."
    )

    # ── Model filter ────────────────────────────────────────────
    explorer_models_list = ["Baseline GA", "Coevolution GA", "Greedy Builder"]
    selected_models = st.multiselect(
        "Select algorithms to display",
        explorer_models_list,
        default=explorer_models_list,
        key="explorer_models",
    )

    filtered = results_df[results_df["model"].isin(selected_models)].copy()

    # ── Baseline GA operator filters ────────────────────────────
    if "Baseline GA" in selected_models:
        with st.expander("Baseline GA Operator Filters", expanded=True):
            bl = filtered[filtered["model"] == "Baseline GA"]
            c1, c2, c3 = st.columns(3)
            with c1:
                all_cx = sorted(bl["crossover"].dropna().unique())
                sel_cx = st.multiselect("Crossover", all_cx, default=all_cx,
                                        key="filter_cx")
            with c2:
                all_mut = sorted(bl["mutation"].dropna().unique())
                sel_mut = st.multiselect("Mutation", all_mut, default=all_mut,
                                         key="filter_mut")
            with c3:
                all_sel = sorted(bl["selection"].dropna().unique())
                sel_sel = st.multiselect("Selection", all_sel, default=all_sel,
                                         key="filter_sel")

            bl_mask = (
                (filtered["model"] == "Baseline GA")
                & (filtered["crossover"].isin(sel_cx))
                & (filtered["mutation"].isin(sel_mut))
                & (filtered["selection"].isin(sel_sel))
            )
            other_mask = filtered["model"] != "Baseline GA"
            filtered = filtered[bl_mask | other_mask].copy()

    # ── Coevolution GA hyperparameter filters ───────────────────
    if "Coevolution GA" in selected_models:
        coevo_data = filtered[filtered["model"] == "Coevolution GA"]
        if not coevo_data.empty:
            with st.expander("Coevolution GA Filters"):
                lam_vals = sorted(coevo_data["coverage_lambda"].dropna().unique())
                mu_vals = sorted(coevo_data["parsimony_mu"].dropna().unique())
                c1, c2 = st.columns(2)
                with c1:
                    sel_lam = st.multiselect(
                        "Coverage Lambda", lam_vals, default=lam_vals,
                        key="filter_lam",
                    )
                with c2:
                    sel_mu = st.multiselect(
                        "Parsimony Mu", mu_vals, default=mu_vals,
                        key="filter_mu",
                    )
                coevo_mask = (
                    (filtered["model"] == "Coevolution GA")
                    & (filtered["coverage_lambda"].isin(sel_lam))
                    & (filtered["parsimony_mu"].isin(sel_mu))
                )
                other_mask = filtered["model"] != "Coevolution GA"
                filtered = filtered[coevo_mask | other_mask].copy()

    st.markdown(f"**Showing {len(filtered)} of {len(results_df)} variants**")

    # ── Best metrics bar chart (Validation + Test) ────────────────
    st.subheader("Best Variant Outcomes -- Validation and Test Sets")

    bar_rows = []
    for model_name in selected_models:
        model_sub = filtered[filtered["model"] == model_name]
        if model_sub.empty:
            continue
        best = model_sub.sort_values("test_f1", ascending=False).iloc[0]
        for split_name, prefix in [("Validation", "val"), ("Test", "test")]:
            bar_rows.append({
                "Algorithm": model_name,
                "Split": split_name,
                "Precision": best[f"{prefix}_precision"],
                "Recall": best[f"{prefix}_recall"],
                "F1": best[f"{prefix}_f1"],
            })

    if bar_rows:
        bar_df = pd.DataFrame(bar_rows)
        for metric_name in ["Precision", "Recall", "F1"]:
            fig_bar = px.bar(
                bar_df, x="Algorithm", y=metric_name,
                color="Split", barmode="group",
                color_discrete_map={"Validation": "#1565C0", "Test": "#FF6F00"},
                text_auto=".2%",
                title=f"Best {metric_name} per Algorithm (Validation vs Test)",
            )
            fig_bar.update_layout(height=380, yaxis_tickformat=".0%")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No data available for the current filters.")

    # ── Convergence line chart ──────────────────────────────────
    st.subheader("Convergence Over Generations / Cycles")
    st.write(
        "Simulated convergence curves showing how each algorithm's best F1 "
        "improves over generations (Baseline GA) or cycles (Coevolution GA, "
        "Greedy Builder)."
    )

    conv_traces = []

    if "Baseline GA" in selected_models:
        bl_sub = filtered[filtered["model"] == "Baseline GA"]
        if not bl_sub.empty:
            best_bl_f1 = bl_sub["test_f1"].max()
            gens = list(range(31))
            bl_curve = [0.05 + (best_bl_f1 - 0.05) * (1 - np.exp(-g / 6))
                        for g in gens]
            for g, f in zip(gens, bl_curve):
                conv_traces.append({
                    "Step": g, "F1": f, "Algorithm": "Baseline GA",
                })

    if "Coevolution GA" in selected_models:
        coevo_sub = filtered[filtered["model"] == "Coevolution GA"]
        if not coevo_sub.empty:
            best_coevo_f1 = coevo_sub["test_f1"].max()
            cycles = list(range(0, 81, 2))
            coevo_curve = [0.15 + (best_coevo_f1 - 0.15) * (1 - np.exp(-c / 18))
                           for c in cycles]
            for c, f in zip(cycles, coevo_curve):
                conv_traces.append({
                    "Step": c, "F1": f, "Algorithm": "Coevolution GA",
                })

    if "Greedy Builder" in selected_models:
        greedy_sub = filtered[filtered["model"] == "Greedy Builder"]
        if not greedy_sub.empty:
            best_greedy_f1 = greedy_sub["test_f1"].max()
            n_rules = summary.get("greedy_builder", {}).get("n_paths", 7)
            steps = list(range(1, n_rules + 1))
            marginal = [0.30, 0.18, 0.10, 0.06, 0.03, 0.015, 0.008][:n_rules]
            cumulative = [sum(marginal[:i + 1]) for i in range(len(marginal))]
            # Scale so final cumulative matches actual best F1
            scale = best_greedy_f1 / cumulative[-1] if cumulative[-1] > 0 else 1
            for s, f in zip(steps, cumulative):
                conv_traces.append({
                    "Step": s, "F1": f * scale, "Algorithm": "Greedy Builder",
                })

    if conv_traces:
        conv_df = pd.DataFrame(conv_traces)
        fig_conv = px.line(
            conv_df, x="Step", y="F1", color="Algorithm",
            color_discrete_map=ALGO_COLORS,
            title="Convergence: Best F1 Over Generations / Cycles / Rules Added",
            labels={"Step": "Generation / Cycle / Rule #", "F1": "Best F1"},
        )
        fig_conv.update_layout(height=450)

        # Add horizontal lines for final F1 per algorithm
        for model_name in selected_models:
            model_sub = filtered[filtered["model"] == model_name]
            if not model_sub.empty:
                final_f1 = model_sub["test_f1"].max()
                fig_conv.add_hline(
                    y=final_f1, line_dash="dot",
                    line_color=ALGO_COLORS.get(model_name, "#888"),
                    annotation_text=f"{model_name}: {final_f1:.4f}",
                    annotation_position="top left",
                )

        st.plotly_chart(fig_conv, use_container_width=True)
        st.caption(
            "Baseline GA: 30 generations  |  "
            "Coevolution GA: ~80 cycles (multi-rule team building)  |  "
            "Greedy Builder: rules added one at a time (deterministic)"
        )

    # ── Ranked F1 table ─────────────────────────────────────────
    st.subheader("Filtered Variants Ranked by Test F1 Score")
    n_show = st.slider("Number of rows to display", 5, min(100, len(filtered)),
                        value=min(20, len(filtered)), key="n_show")
    ranked = filtered.sort_values("test_f1", ascending=False).head(n_show)
    display_cols = ["model", "variant",
                    "test_precision", "test_recall", "test_f1",
                    "val_precision", "val_recall", "val_f1"]
    optional = ["crossover", "mutation", "selection", "coverage_lambda",
                "parsimony_mu", "pop_size", "cx_prob"]
    for c in optional:
        if c in ranked.columns and ranked[c].notna().any():
            display_cols.append(c)

    st.dataframe(
        ranked[display_cols].reset_index(drop=True).round(4),
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════
# SECTION 3: Best Rules per Model
# ════════════════════════════════════════════════════════════════
elif section == "Best Rules per Model":
    st.header("Best Rules per Model")
    st.write(
        "Side-by-side comparison of the best rules produced by each algorithm. "
        "These results come from the pre-computed pipeline output "
        "(results_summary.json) and are **independent of the Algorithm Explorer** "
        "filters. Select which models to view."
    )

    model_choice = st.multiselect(
        "Select models to compare",
        ["Baseline GA", "Coevolution GA", "NSGA-II", "Greedy Builder"],
        default=["Baseline GA", "Coevolution GA", "Greedy Builder"],
        key="rules_models",
    )

    # ── Summary cards ───────────────────────────────────────────
    summary_map = {
        "Baseline GA": ("baseline_ga", "best_rule"),
        "Coevolution GA": ("coevolution_ga", "best_ruleset"),
        "Greedy Builder": ("greedy_builder", "best_ruleset"),
    }

    cols = st.columns(max(len(model_choice), 1))
    for i, model_name in enumerate(model_choice):
        with cols[i % len(cols)]:
            st.subheader(model_name)

            if model_name == "NSGA-II":
                # NSGA-II: show operating points
                nsga2_s = summary.get("nsga2_ga", {})
                nsga_ops_info = nsga2_s.get("operators", {})
                if nsga_ops_info:
                    st.markdown("**Best Parameters**")
                    st.markdown(
                        f"- Selection: `{nsga_ops_info.get('selection', 'N/A')}`\n"
                        f"- Crossover: `{nsga_ops_info.get('crossover', 'N/A')}`\n"
                        f"- Mutation: `{nsga_ops_info.get('mutation', 'N/A')}`\n"
                        f"- Fitness: `{nsga_ops_info.get('fitness', 'N/A')}`"
                    )
                ops = nsga2_s.get("operating_points", {})
                for pt_name, pt_data in ops.items():
                    tm = pt_data.get("test_metrics", {})
                    st.markdown(f"**{pt_name.title()}**")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Prec", f"{_safe_float(tm.get('precision')):.2%}")
                    mc2.metric("Rec", f"{_safe_float(tm.get('recall')):.2%}")
                    mc3.metric("F1", f"{_safe_float(tm.get('f1')):.2%}")
                    mc4.metric("Alert Rate", f"{_safe_float(tm.get('alert_rate')):.4%}")

                    # Find closest variant
                    nsga_res = results_df[results_df["model"] == "NSGA-II"].copy()
                    if not nsga_res.empty and tm:
                        tp = _safe_float(tm.get("precision", 0))
                        tr = _safe_float(tm.get("recall", 0))
                        nsga_res["_d"] = (
                            (nsga_res["test_precision"] - tp) ** 2
                            + (nsga_res["test_recall"] - tr) ** 2
                        ) ** 0.5
                        closest = nsga_res.sort_values("_d").iloc[0]
                        rule_str = closest.get("best_rule_str", "")
                        st.caption(f"Variant: `{closest['variant']}`")
                        if isinstance(rule_str, str) and rule_str.strip():
                            with st.expander(f"Rules ({pt_name.title()})"):
                                for r in parse_rules(rule_str):
                                    st.code(r, language=None)
                    st.markdown("---")
            elif model_name in summary_map:
                key, rules_key = summary_map[model_name]
                fam = summary.get(key, {})
                test_m = fam.get("test_metrics", {})
                operators = fam.get("operators", {})
                if test_m:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Precision",
                               f"{_safe_float(test_m.get('precision')):.2%}")
                    mc2.metric("Recall",
                               f"{_safe_float(test_m.get('recall')):.2%}")
                    mc3.metric("F1",
                               f"{_safe_float(test_m.get('f1')):.2%}")
                    mc4.metric("Alert Rate",
                               f"{_safe_float(test_m.get('alert_rate')):.4%}")

                if operators:
                    st.markdown("**Best Parameters**")
                    param_lines = []
                    for pkey, pval in operators.items():
                        param_lines.append(f"- {pkey}: `{pval}`")
                    st.markdown("\n".join(param_lines))

                rules = fam.get(rules_key, "")
                parsed = parse_rules(rules)
                if parsed:
                    for j, rule in enumerate(parsed, 1):
                        clean = rule.strip()
                        if clean.startswith(f"Rule {j}:"):
                            clean = clean[len(f"Rule {j}:"):].strip()
                        st.markdown(f"**Rule {j}:** `{clean}`")
                else:
                    st.info(f"No rules found for {model_name}.")

    # ── Side-by-side metrics comparison ─────────────────────────
    st.markdown("---")
    st.subheader("Metrics Comparison")

    comp_rows = []
    for model_name in model_choice:
        subset = results_df[results_df["model"] == model_name]
        if subset.empty:
            continue
        best = subset.sort_values("test_f1", ascending=False).iloc[0]
        comp_rows.append({
            "Algorithm": model_name,
            "Test Precision": best["test_precision"],
            "Test Recall": best["test_recall"],
            "Test F1": best["test_f1"],
            "Variant": best["variant"],
        })
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        fig_comp = go.Figure()
        metric_colors = ["#1565C0", "#FF6F00", "#2E7D32"]
        for metric, color in zip(
            ["Test Precision", "Test Recall", "Test F1"], metric_colors
        ):
            fig_comp.add_trace(go.Bar(
                x=comp_df["Algorithm"], y=comp_df[metric],
                name=metric, marker_color=color,
                text=[f"{v:.2%}" for v in comp_df[metric]],
                textposition="outside",
                textfont=dict(color="white", size=13),
            ))
        fig_comp.update_layout(
            barmode="group", height=450,
            yaxis_tickformat=".0%",
            title="Best Test Metrics by Algorithm",
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# SECTION 4: Pareto Front
# ════════════════════════════════════════════════════════════════
elif section == "Pareto Front":
    st.header("Pareto Front Analysis")
    st.write(
        "Precision-Recall Pareto front visualizations adapted from "
        "**results_analysis_and_visualizations.ipynb**. "
        "Choose which model families to display and which data split to analyze."
    )

    # ── Controls ────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        pareto_models = st.multiselect(
            "Model families",
            ["Baseline GA", "Coevolution GA", "NSGA-II", "Greedy Builder"],
            default=["Coevolution GA", "NSGA-II", "Greedy Builder"],
            key="pareto_models",
        )
    with c2:
        pareto_split = st.radio(
            "Split", ["Test", "Validation"], horizontal=True,
            key="pareto_split",
        )

    sp = {"Test": "test", "Validation": "val"}[pareto_split]
    p_col = f"{sp}_precision"
    r_col = f"{sp}_recall"
    f_col = f"{sp}_f1"

    # ── Baseline GA color-by selector ───────────────────────────
    baseline_color_by = None
    if "Baseline GA" in pareto_models:
        baseline_color_by = st.selectbox(
            "Color Baseline GA dots by",
            ["None (single color)", "Crossover", "Mutation", "Selection"],
            key="bl_color_by",
        )

    # ── Notebook-style Pareto chart ─────────────────────────────
    CX_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2"]
    MUT_PALETTE = ["#17becf", "#bcbd22", "#7f7f7f"]
    SEL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig = go.Figure()

    for model_name in pareto_models:
        model_df = results_df[results_df["model"] == model_name].copy()
        model_df = model_df.dropna(subset=[p_col, r_col, f_col])
        if model_df.empty:
            continue

        # For Baseline GA, optionally color by operator
        if model_name == "Baseline GA" and baseline_color_by and baseline_color_by != "None (single color)":
            col_name = baseline_color_by.lower()
            groups = sorted(model_df[col_name].dropna().unique())
            if col_name == "crossover":
                pal = dict(zip(groups, CX_PALETTE[:len(groups)]))
            elif col_name == "mutation":
                pal = dict(zip(groups, MUT_PALETTE[:len(groups)]))
            else:
                pal = dict(zip(groups, SEL_PALETTE[:len(groups)]))

            for grp in groups:
                grp_df = model_df[model_df[col_name] == grp]
                color = pal.get(grp, "#888888")
                _add_model_traces(fig, grp_df, grp, color, p_col, r_col, f_col)
        else:
            color = ALGO_COLORS.get(model_name, "#888888")
            _add_model_traces(fig, model_df, model_name, color, p_col, r_col, f_col)

    # ── NSGA-II operating points ────────────────────────────────
    if "NSGA-II" in pareto_models:
        nsga2_s = summary.get("nsga2_ga", {})
        ops = nsga2_s.get("operating_points", {})
        op_styles = {
            "conservative": {"symbol": "diamond", "color": "#2E7D32"},
            "balanced": {"symbol": "square", "color": "#FF6F00"},
            "aggressive": {"symbol": "star-diamond", "color": "#D32F2F"},
        }
        for pt_name, style in op_styles.items():
            pt_data = ops.get(pt_name, {}).get("test_metrics", {})
            if pareto_split == "Validation":
                pt_data = ops.get(pt_name, {}).get("val_metrics", {})
                # val_metrics has mixed keys; try test_precision style first
                px_val = _safe_float(pt_data.get("precision",
                         pt_data.get("test_precision")))
                ry_val = _safe_float(pt_data.get("recall",
                         pt_data.get("test_recall")))
            else:
                px_val = _safe_float(pt_data.get("precision"))
                ry_val = _safe_float(pt_data.get("recall"))
            if not (np.isnan(px_val) or np.isnan(ry_val)):
                fig.add_trace(go.Scatter(
                    x=[ry_val], y=[px_val],
                    mode="markers+text",
                    name=f"OP: {pt_name.title()}",
                    marker=dict(size=22, color=style["color"],
                                symbol=style["symbol"],
                                line=dict(width=2, color="white")),
                    text=[pt_name.title()],
                    textposition="top center",
                    textfont=dict(size=12, color=style["color"]),
                ))

    fig.update_layout(
        height=650,
        xaxis_title="Recall",
        yaxis_title="Precision",
        title=f"Precision-Recall Pareto Front ({pareto_split} Set)",
        legend=dict(font=dict(size=10), yanchor="top", y=0.99,
                    xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Star = best F1  |  Square = group mean  |  "
        "Diamond/Square/Star-diamond = NSGA-II operating points  |  "
        "Visualization logic from results_analysis_and_visualizations.ipynb"
    )

    # ── Operating point details ─────────────────────────────────
    if "NSGA-II" in pareto_models:
        st.markdown("---")
        st.subheader("NSGA-II Operating Points")

        nsga2_summary = summary.get("nsga2_ga", {})
        operating_points = nsga2_summary.get("operating_points", {})

        col_c, col_b, col_a = st.columns(3)
        for col_widget, pt_name, description in [
            (col_c, "conservative",
             "Maximize precision, alert rate <= 0.05%. Flag only high-confidence fraud."),
            (col_b, "balanced",
             "Maximize F2 (recall-weighted), alert rate <= 0.20%. Best for general operations."),
            (col_a, "aggressive",
             "Maximize recall, alert rate <= 1.0%, precision >= 10%. Catch as much fraud as possible."),
        ]:
            with col_widget:
                st.markdown(f"#### {pt_name.title()}")
                st.write(description)
                pt = operating_points.get(pt_name, {}).get("test_metrics", {})
                if pt:
                    st.metric("Precision", f"{_safe_float(pt.get('precision')):.2%}")
                    st.metric("Recall", f"{_safe_float(pt.get('recall')):.2%}")
                    st.metric("F1", f"{_safe_float(pt.get('f1')):.4f}")
                    st.metric("Alert Rate", f"{_safe_float(pt.get('alert_rate')):.4%}")
