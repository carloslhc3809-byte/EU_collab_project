import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import umap.umap_ as umap


st.set_page_config(
    page_title="Temporal Co-Patent Link Predictor",
    page_icon="🧬",
    layout="wide",
)

HIGH_CONF_THRESHOLD = 0.9
SCORE_THRESHOLDS = {"Very high": 0.9, "High": 0.75, "Moderate": 0.6}

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "assets" / "dashboard_walkthrough.mp4"
SLIDES_PATH = BASE_DIR / "assets" / "dashboard_slides.pdf"
SLIDES_IMG_DIR = BASE_DIR / "assets" / "slides"

CLUSTER_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
    -1: "#636EFA",
}


@st.cache_data(ttl=0)
def load_data():
    data_dir = BASE_DIR / "data" / "processed"
    if not data_dir.exists():
        data_dir = BASE_DIR / "notebooks" / "data" / "processed"

    df = pd.read_csv(data_dir / "metadata" / "nodes_with_clusters.csv")
    z = np.load(data_dir / "embeddings" / "z_2024.npy")

    with open(data_dir / "predictions" / "top_links.json", "r") as f:
        top_links = json.load(f)

    metrics = None
    metrics_path = data_dir / "predictions" / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    cluster_matrix = None
    cluster_matrix_path = data_dir / "predictions" / "cluster_matrix.npy"
    if cluster_matrix_path.exists():
        cluster_matrix = np.load(cluster_matrix_path)

    prob_matrix = None
    prob_matrix_path = data_dir / "predictions" / "prob_matrix.npy"
    if prob_matrix_path.exists():
        prob_matrix = np.load(prob_matrix_path)

    train_edges = None
    train_edge_path = data_dir / "edges" / "train_edges.npy"
    if train_edge_path.exists():
        train_edges = np.load(train_edge_path)
        if train_edges.ndim == 2 and train_edges.shape[1] != 2 and train_edges.shape[0] == 2:
            train_edges = train_edges.T
        if train_edges.ndim == 2 and train_edges.shape[1] == 2:
            train_edges = train_edges.astype(int)

    test_edges = None
    test_edge_path = data_dir / "edges" / "test_edges.npy"
    if test_edge_path.exists():
        test_edges = np.load(test_edge_path)
        if test_edges.ndim == 2 and test_edges.shape[1] != 2 and test_edges.shape[0] == 2:
            test_edges = test_edges.T
        if test_edges.ndim == 2 and test_edges.shape[1] == 2:
            test_edges = test_edges.astype(int)

    return df, z, top_links, metrics, cluster_matrix, prob_matrix, train_edges, test_edges


@st.cache_data
def compute_umap(z_array):
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.9,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(z_array)


@st.cache_data
def get_slide_images(slides_dir):
    slides_dir = Path(slides_dir)
    if not slides_dir.exists():
        return []

    def slide_sort_key(path):
        try:
            return int(path.stem.split("_")[-1])
        except Exception:
            return float("inf")

    return sorted(slides_dir.glob("slide_*.png"), key=slide_sort_key)


df, z, top_links, metrics, cluster_matrix, prob_matrix, train_edges, test_edges = load_data()

df = df[df["node_id"] < len(z)].copy()
df["applicant_name"] = df["applicant_name"].astype(str)
df.loc[df["applicant_name"].str.lower().isin(["nan", "none", ""]), "applicant_name"] = np.nan
df["applicant_name"] = df["applicant_name"].fillna("Unknown applicant")
df["clean"] = df["applicant_name"].str.strip().str.lower()
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)
df["cluster_color"] = df["cluster"].map(CLUSTER_COLORS).fillna(CLUSTER_COLORS[-1])

with st.spinner("Computing UMAP layout — this may take a moment on first load…"):
    coords = compute_umap(z[df["node_id"].values])

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

nodeid_to_row = {nid: i for i, nid in enumerate(df["node_id"].tolist())}
name_to_nodeid = dict(zip(df["clean"], df["node_id"]))
nodeid_to_name = dict(zip(df["node_id"], df["applicant_name"]))

historical_edges = set()
if train_edges is not None:
    for i, j in train_edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        historical_edges.add((a, b))

test_year_edges = set()
if test_edges is not None:
    for i, j in test_edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        test_year_edges.add((a, b))


def get_row_by_nodeid(node_id):
    return df.iloc[nodeid_to_row[node_id]]


def score_label(score):
    if score >= SCORE_THRESHOLDS["Very high"]:
        return "Very high"
    if score >= SCORE_THRESHOLDS["High"]:
        return "High"
    if score >= SCORE_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Weak"


def edge_status(source, target):
    a, b = (source, target) if source < target else (target, source)
    if (a, b) in historical_edges:
        return "Historical"
    if (a, b) in test_year_edges:
        return "Observed in test year"
    return "Predicted"


def get_links_for_node(node_id):
    raw = top_links.get(str(node_id), top_links.get(node_id, []))
    result = []
    for item in raw:
        tgt = int(item["target"])
        if tgt not in nodeid_to_row or tgt == node_id:
            continue
        status = edge_status(node_id, tgt)
        result.append({
            "source": int(node_id),
            "target": int(tgt),
            "score": float(item["score"]),
            "status": status,
            "observed": status != "Predicted",
            "target_name": nodeid_to_name.get(tgt, f"Node {tgt}"),
        })
    return sorted(result, key=lambda x: x["score"], reverse=True)


def filter_links(link_list, threshold_eff):
    return [l for l in link_list if l["score"] >= threshold_eff]


@st.cache_data
def build_candidate_nodeids(threshold_eff):
    result = []
    for node_id in df["node_id"]:
        node_links = get_links_for_node(node_id)
        if any(l["score"] >= threshold_eff for l in node_links):
            result.append(int(node_id))
    return result


def base_scatter(selected_node_id=None, shown_links=None):
    fig = go.Figure()
    shown_links = shown_links or []
    predicted_neighbors = {l["target"] for l in shown_links}

    for cluster_id in sorted(df["cluster"].unique()):
        df_cluster = df[df["cluster"] == cluster_id]
        if df_cluster.empty:
            continue

        fig.add_trace(go.Scatter(
            x=df_cluster["x"],
            y=df_cluster["y"],
            mode="markers",
            marker=dict(size=4.0, color=df_cluster["cluster_color"].tolist(), opacity=0.30, line=dict(width=0)),
            text=df_cluster["applicant_name"],
            customdata=np.stack([df_cluster["node_id"], df_cluster["cluster"]], axis=1),
            hovertemplate="<b>%{text}</b><br>Node ID: %{customdata[0]}<br>Cluster: %{customdata[1]}<extra></extra>",
            name=f"Cluster {cluster_id}",
            showlegend=True,
        ))

        df_high = df_cluster[
            df_cluster["node_id"].isin(
                predicted_neighbors | ({selected_node_id} if selected_node_id is not None else set())
            )
        ]

        if not df_high.empty:
            high_sizes, high_line_widths, high_line_colors = [], [], []
            for nid in df_high["node_id"]:
                if nid == selected_node_id:
                    high_sizes.append(11)
                    high_line_widths.append(1.8)
                    high_line_colors.append("rgba(10,10,10,0.95)")
                else:
                    high_sizes.append(8.5)
                    high_line_widths.append(1.2)
                    high_line_colors.append("rgba(20,20,20,0.85)")

            fig.add_trace(go.Scatter(
                x=df_high["x"],
                y=df_high["y"],
                mode="markers",
                marker=dict(
                    size=high_sizes,
                    color=df_high["cluster_color"].tolist(),
                    opacity=1.0,
                    line=dict(width=high_line_widths, color=high_line_colors),
                ),
                text=df_high["applicant_name"],
                customdata=np.stack([df_high["node_id"], df_high["cluster"]], axis=1),
                hovertemplate="<b>%{text}</b><br>Node ID: %{customdata[0]}<br>Cluster: %{customdata[1]}<extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def add_ego_edges(fig, node_id, links):
    source_row = get_row_by_nodeid(node_id)
    x0, y0 = source_row["x"], source_row["y"]

    pred_x, pred_y = [], []
    hist_x, hist_y = [], []
    test_x, test_y = [], []

    for link in links:
        tgt = link["target"]
        target_row = get_row_by_nodeid(tgt)
        x1, y1 = target_row["x"], target_row["y"]

        if link["status"] == "Historical":
            hist_x += [x0, x1, None]
            hist_y += [y0, y1, None]
        elif link["status"] == "Observed in test year":
            test_x += [x0, x1, None]
            test_y += [y0, y1, None]
        else:
            pred_x += [x0, x1, None]
            pred_y += [y0, y1, None]

    if pred_x:
        fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode="lines",
            line=dict(width=1, dash="dot", color="#0072CE"), hoverinfo="skip", name="Predicted"))
    if hist_x:
        fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode="lines",
            line=dict(width=1.8, color="#5A5A5A"), hoverinfo="skip", name="Historical"))
    if test_x:
        fig.add_trace(go.Scatter(x=test_x, y=test_y, mode="lines",
            line=dict(width=2.4, color="#00A878"), hoverinfo="skip", name="Observed in test year"))

    return fig


def render_slide_viewer():
    """Analytical guide slide viewer with buttons and slider kept in sync."""
    slide_images = get_slide_images(str(SLIDES_IMG_DIR))

    if SLIDES_PATH.exists():
        with open(SLIDES_PATH, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download slides (PDF)",
            data=pdf_bytes,
            file_name=SLIDES_PATH.name,
            mime="application/pdf",
        )
    else:
        st.caption("PDF deck not found — expected at `assets/dashboard_slides.pdf`.")

    if not slide_images:
        st.caption("No slide images found — expected as `assets/slides/slide_*.png`.")
        return

    total = len(slide_images)

    if "current_slide" not in st.session_state:
        st.session_state.current_slide = 1
    st.session_state.current_slide = max(1, min(st.session_state.current_slide, total))

    nav_left, nav_mid, nav_right = st.columns([1, 4, 1])

    with nav_left:
        prev_clicked = st.button(
            "⬅ Previous",
            disabled=(st.session_state.current_slide == 1),
            use_container_width=True,
            key="slide_prev",
        )

    with nav_right:
        next_clicked = st.button(
            "Next ➡",
            disabled=(st.session_state.current_slide == total),
            use_container_width=True,
            key="slide_next",
        )

    if prev_clicked:
        st.session_state.current_slide -= 1
    if next_clicked:
        st.session_state.current_slide += 1

    with nav_mid:
        st.slider(
            "Slide",
            min_value=1,
            max_value=total,
            key="current_slide",
            label_visibility="collapsed",
        )

    st.image(str(slide_images[st.session_state.current_slide - 1]), use_container_width=True)
    st.caption(f"Slide {st.session_state.current_slide} of {total}")


if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False


if not st.session_state.onboarding_complete:
    st.title("🧬 Temporal Co-Patent Link Predictor")

    video_col, desc_col = st.columns([3, 2], gap="large")

    with video_col:
        if VIDEO_PATH.exists():
            st.video(str(VIDEO_PATH))
        else:
            st.caption("Walkthrough video not found — expected at `assets/dashboard_walkthrough.mp4`.")

    with desc_col:
        st.markdown(
            """
            This dashboard explores model-predicted co-patenting link candidates between
            organisations using a temporal graph neural network trained on historical
            patent collaboration data. The model learns from the evolving structure of
            co-inventor networks — capturing not only who has collaborated, but also how
            collaboration patterns change over time — to identify which pairs of
            organisations emerge as strong candidates for future joint patenting. Each
            prediction is assigned a model score and can be filtered by score threshold or
            explored spatially through an interactive UMAP embedding of the learned
            node representations. The Explorer tab lets you select any organisation and
            inspect its ranked collaboration candidates, while the Model Overview tab
            provides training diagnostics and the full analytical guide.
            """
        )

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Enter dashboard →", type="primary", use_container_width=True):
            st.session_state.onboarding_complete = True
            st.rerun()

    st.stop()


st.title("🧬 Temporal Co-Patent Link Predictor")
st.caption(
    "Explore model-predicted co-patenting link candidates between organizations "
    "based on historical collaboration patterns and temporal graph embeddings."
)

section = st.radio(
    "Section",
    ["Explorer", "Model Overview"],
    horizontal=True,
    label_visibility="collapsed",
)

if section == "Explorer":
    st.sidebar.markdown("### Explorer")

    _prev_threshold = st.session_state.get("_explorer_threshold", 0.0)
    _prev_hco = st.session_state.get("_explorer_hco", False)
    _threshold_eff_init = max(_prev_threshold, HIGH_CONF_THRESHOLD) if _prev_hco else _prev_threshold

    candidate_nodeids = build_candidate_nodeids(_threshold_eff_init)
    valid_names = sorted({
        nodeid_to_name[nid].strip().lower()
        for nid in candidate_nodeids
        if nid in nodeid_to_name
    })

    if not valid_names:
        st.sidebar.warning("No nodes meet the current threshold.")
        selected = None
        top_k = 5
    else:
        selected = st.sidebar.selectbox("Select company", valid_names)

        _node_id_init = name_to_nodeid.get(selected)
        _all_links_init = get_links_for_node(_node_id_init) if _node_id_init is not None else []
        _filtered_init = filter_links(_all_links_init, _threshold_eff_init)
        _max_k_init = max(1, len(_filtered_init))

        top_k = st.sidebar.slider(
            "Number of links displayed",
            min_value=1,
            max_value=_max_k_init,
            value=min(5, _max_k_init),
        )

    threshold = st.sidebar.slider(
        "Prediction score threshold", 0.0, 1.0, _prev_threshold, 0.05
    )

    high_conf_only = st.sidebar.checkbox("High confidence only (≥ 0.9)", value=_prev_hco)

    threshold_eff = max(threshold, HIGH_CONF_THRESHOLD) if high_conf_only else threshold
    st.session_state["_explorer_threshold"] = threshold
    st.session_state["_explorer_hco"] = high_conf_only

    candidate_nodeids = build_candidate_nodeids(threshold_eff)
    valid_names = sorted({
        nodeid_to_name[nid].strip().lower()
        for nid in candidate_nodeids
        if nid in nodeid_to_name
    })

    st.sidebar.metric("Applicants with links", len(valid_names))

    st.sidebar.divider()
    if st.sidebar.button("← Back to home"):
        st.session_state.onboarding_complete = False
        st.rerun()

    if not valid_names:
        st.info("No nodes meet the current threshold. Try lowering the score threshold in the sidebar.")
    else:
        if selected not in valid_names:
            selected = valid_names[0]

        node_id = name_to_nodeid[selected]

        all_links = get_links_for_node(node_id)
        all_links_filtered = filter_links(all_links, threshold_eff)
        max_links_for_selected_node = max(1, len(all_links_filtered))
        top_k = min(top_k, max_links_for_selected_node)
        links = all_links_filtered[:top_k]

        selected_cluster = int(get_row_by_nodeid(node_id)["cluster"])

        cluster_density = None
        if cluster_matrix is not None and 0 <= selected_cluster < cluster_matrix.shape[0]:
            cluster_density = float(cluster_matrix[selected_cluster, selected_cluster])

        all_predicted_clusters = {
            int(get_row_by_nodeid(l["target"])["cluster"])
            for l in all_links_filtered
            if l["target"] in nodeid_to_row
        }
        cluster_diversity_total = len(all_predicted_clusters)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            "Displayed candidates", len(links),
            help="Number of candidate links shown based on top-k and threshold settings.",
        )
        kpi2.metric(
            "Eligible predicted candidates", len(all_links_filtered),
            help="Total candidates for this company that pass the score threshold.",
        )
        kpi3.metric(
            "Cluster cohesion",
            f"{cluster_density:.3f}" if cluster_density is not None else "n/a",
            help="Intra-cluster link density for the selected company's cluster. Higher = more tightly knit group.",
        )
        kpi4.metric(
            "Cluster diversity", cluster_diversity_total,
            help="Number of distinct clusters represented among all eligible candidates.",
        )

        fig = base_scatter(node_id, links)
        if links:
            fig = add_ego_edges(fig, node_id, links)
        st.plotly_chart(fig, use_container_width=True)
        

        st.markdown("#### Ranked collaboration candidates")
        st.caption(
            "Scores reflect the model’s relative ranking of collaboration candidates "
            "based on historical network patterns and temporal dynamics."
        )

        if links:
            table_df = pd.DataFrame({
                "Target": [l["target_name"] for l in links],
                "Score": [round(l["score"], 4) for l in links],
                "Confidence": [score_label(l["score"]) for l in links],
                "Cluster": [int(get_row_by_nodeid(l["target"])["cluster"]) for l in links],
            })
            st.dataframe(table_df, use_container_width=True)
        else:
            st.info("No links satisfy the current settings.")

else:
    st.sidebar.markdown("### Model summary")

    if metrics is not None:
        train_years = metrics.get("train_years", [])
        test_year = metrics.get("test_year", None)
        model_cfg = metrics.get("model_config", {})

        st.sidebar.metric(
            "Final AUC", f'{metrics.get("final_auc", float("nan")):.3f}',
            help="Area Under the ROC Curve on the held-out test year.",
        )
        st.sidebar.metric(
            "Final AP", f'{metrics.get("final_ap", float("nan")):.3f}',
            help="Average Precision on the held-out test year.",
        )
        if train_years:
            st.sidebar.markdown(f"**Train period:** {min(train_years)}–{max(train_years)}")
        if test_year is not None:
            st.sidebar.markdown(f"**Test year:** {test_year}")
        if model_cfg:
            st.sidebar.markdown(f"**Predictor:** {model_cfg.get('predictor', 'n/a')}")
            st.sidebar.markdown(f"**Temporal aggregation:** {model_cfg.get('temporal', 'n/a')}")
    else:
        st.sidebar.caption("No metrics file found.")

    st.markdown("#### Analytical guide")
    st.caption(
        "Covers the project context, data construction process, modeling architecture, "
        "dashboard functionality, and interpretation of outputs."
    )
    render_slide_viewer()

    st.divider()

    if metrics is None:
        st.info("No training metrics file found at `data/processed/predictions/training_metrics.json`.")
    else:
        model_cfg = metrics.get("model_config", {})

        if metrics.get("losses"):
            st.markdown("#### Training loss")
            loss_df = pd.DataFrame({
                "epoch": np.arange(len(metrics["losses"])),
                "loss": metrics["losses"],
            })
            st.line_chart(loss_df.set_index("epoch"))

        chart_cols = st.columns(2)

        if metrics.get("auc_history"):
            auc_df = pd.DataFrame(metrics["auc_history"])
            if not auc_df.empty:
                with chart_cols[0]:
                    st.markdown("#### Validation AUC over training")
                    st.line_chart(auc_df.set_index("epoch"))

        if metrics.get("ap_history"):
            ap_df = pd.DataFrame(metrics["ap_history"])
            if not ap_df.empty:
                with chart_cols[1]:
                    st.markdown("#### Validation AP over training")
                    st.line_chart(ap_df.set_index("epoch"))

        if model_cfg:
            st.divider()
            st.markdown("#### Full model configuration")
            st.json(model_cfg)
