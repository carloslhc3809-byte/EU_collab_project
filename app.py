import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import umap.umap_ as umap

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Temporal Co-Patent Link Predictor",
    page_icon="🧬",
    layout="wide",
)

HIGH_CONF_THRESHOLD = 0.9

st.title("EU Collaboration Dashboard")

st.info("New to the dashboard? Watch this short walkthrough to understand its features and how to use it")

with st.expander("Open video walkthrough"):
    st.video("assets/dashboard_walkthrough.mp4")

# Fixed cluster color mapping
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

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
@st.cache_data(ttl=0)
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "notebooks", "data", "processed")

    df = pd.read_csv(os.path.join(data_dir, "metadata", "nodes_with_clusters.csv"))
    z = np.load(os.path.join(data_dir, "embeddings", "z_2024.npy"))

    with open(os.path.join(data_dir, "predictions", "top_links.json")) as f:
        top_links = json.load(f)

    metrics = None
    metrics_path = os.path.join(data_dir, "predictions", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    cluster_matrix = None
    cluster_matrix_path = os.path.join(data_dir, "predictions", "cluster_matrix.npy")
    if os.path.exists(cluster_matrix_path):
        cluster_matrix = np.load(cluster_matrix_path)

    prob_matrix = None
    prob_matrix_path = os.path.join(data_dir, "predictions", "prob_matrix.npy")
    if os.path.exists(prob_matrix_path):
        prob_matrix = np.load(prob_matrix_path)

    train_edges = None
    train_edge_path = os.path.join(data_dir, "edges", "train_edges.npy")
    if os.path.exists(train_edge_path):
        train_edges = np.load(train_edge_path)
        if train_edges.ndim == 2 and train_edges.shape[1] != 2 and train_edges.shape[0] == 2:
            train_edges = train_edges.T
        if train_edges.ndim == 2 and train_edges.shape[1] == 2:
            train_edges = train_edges.astype(int)

    test_edges = None
    test_edge_path = os.path.join(data_dir, "edges", "test_edges.npy")
    if os.path.exists(test_edge_path):
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
        random_state=42
    )
    return reducer.fit_transform(z_array)


# ─────────────────────────────────────────
# PREPARE DATA
# ─────────────────────────────────────────
df, z, top_links, metrics, cluster_matrix, prob_matrix, train_edges, test_edges = load_data()

df = df[df["node_id"] < len(z)].copy()

df["applicant_name"] = df["applicant_name"].astype(str)
df.loc[df["applicant_name"].str.lower().isin(["nan", "none", ""]), "applicant_name"] = np.nan
df["applicant_name"] = df["applicant_name"].fillna("Unknown applicant")
df["clean"] = df["applicant_name"].str.strip().str.lower()

df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)
df["cluster_color"] = df["cluster"].map(CLUSTER_COLORS).fillna(CLUSTER_COLORS[-1])

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

known_edges = historical_edges | test_year_edges


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_row_by_nodeid(node_id):
    return df.iloc[nodeid_to_row[node_id]]

def score_label(score):
    if score >= 0.9:
        return "Very high"
    if score >= 0.75:
        return "High"
    if score >= 0.6:
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

        if tgt not in nodeid_to_row:
            continue
        if tgt == node_id:
            continue

        status = edge_status(node_id, tgt)

        result.append({
            "source": int(node_id),
            "target": int(tgt),
            "score": float(item["score"]),
            "status": status,
            "observed": status != "Predicted",
            "target_name": nodeid_to_name.get(tgt, f"Node {tgt}")
        })

    return sorted(result, key=lambda x: x["score"], reverse=True)

def filter_links(link_list, threshold_eff):
    return [l for l in link_list if l["score"] >= threshold_eff]


# ─────────────────────────────────────────
# SCATTER
# ─────────────────────────────────────────
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
            marker=dict(
                size=4.0,
                color=df_cluster["cluster_color"].tolist(),
                opacity=0.30,
                line=dict(width=0)
            ),
            text=df_cluster["applicant_name"],
            customdata=np.stack([df_cluster["node_id"], df_cluster["cluster"]], axis=1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Node ID: %{customdata[0]}<br>"
                "Cluster: %{customdata[1]}<extra></extra>"
            ),
            name=f"Cluster {cluster_id}",
            showlegend=True
        ))

        df_high = df_cluster[
            df_cluster["node_id"].isin(
                predicted_neighbors | ({selected_node_id} if selected_node_id is not None else set())
            )
        ]

        if not df_high.empty:
            high_sizes = []
            high_line_widths = []
            high_line_colors = []

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
                    line=dict(width=high_line_widths, color=high_line_colors)
                ),
                text=df_high["applicant_name"],
                customdata=np.stack([df_high["node_id"], df_high["cluster"]], axis=1),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Node ID: %{customdata[0]}<br>"
                    "Cluster: %{customdata[1]}<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )

    return fig


# ─────────────────────────────────────────
# EDGES
# ─────────────────────────────────────────
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
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            mode="lines",
            line=dict(width=0.5, dash="dot", color="#0072CE"),
            hoverinfo="skip",
            name="Predicted"
        ))
    
    if hist_x:
        fig.add_trace(go.Scatter(
            x=hist_x,
            y=hist_y,
            mode="lines",
            line=dict(width=1.8, color="#5A5A5A"),
            hoverinfo="skip",
            name="Historical"
        ))

    if test_x:
        fig.add_trace(go.Scatter(
            x=test_x,
            y=test_y,
            mode="lines",
            line=dict(width=2.4, color="#00A878"),
            hoverinfo="skip",
            name="Observed in test year"
        ))

    return fig


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🧬 Temporal Co-Patent Link Predictor")
st.caption("Applicant embeddings are derived from heterogeneous metapaths and then modeled over time with a GRU to rank candidate future co-patenting links.")


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### Explorer")

    # temporary placeholder; real max is updated after node selection
    top_k_placeholder = st.empty()

    threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.05)
    high_conf_only = st.checkbox("High confidence only (≥ 0.9)")

    if metrics is not None:
        st.markdown("### Model quality")
        st.metric("Final AUC", f'{metrics.get("final_auc", np.nan):.3f}')
        st.metric("Final AP", f'{metrics.get("final_ap", np.nan):.3f}')

        train_years = metrics.get("train_years", [])
        test_year = metrics.get("test_year", None)
        model_cfg = metrics.get("model_config", {})

        if train_years:
            st.write(f"**Train period:** {min(train_years)}–{max(train_years)}")
        if test_year is not None:
            st.write(f"**Test year:** {test_year}")
        if model_cfg:
            st.write(f"**Predictor:** {model_cfg.get('predictor', 'n/a')}")
            st.write(f"**Temporal aggregation:** {model_cfg.get('temporal', 'n/a')}")
            st.write(f"**Activation:** {model_cfg.get('activation', 'n/a')}")
            st.write(f"**Dropout:** {model_cfg.get('dropout', 'n/a')}")

# ─────────────────────────────────────────
# GLOBAL FILTERS
# ─────────────────────────────────────────
threshold_eff = max(threshold, HIGH_CONF_THRESHOLD) if high_conf_only else threshold

candidate_nodeids = []
for node_id in df["node_id"]:
    node_links = get_links_for_node(node_id)
    filtered_node_links = filter_links(node_links, threshold_eff)
    if len(filtered_node_links) > 0:
        candidate_nodeids.append(int(node_id))

valid_names = sorted({
    nodeid_to_name[nid].strip().lower()
    for nid in candidate_nodeids
    if nid in nodeid_to_name
})

st.sidebar.metric("Applicants with links", len(valid_names))

if not valid_names:
    st.warning("No nodes meet the current threshold.")
    st.stop()

selected = st.selectbox("Select company", valid_names)
node_id = name_to_nodeid[selected]


# ─────────────────────────────────────────
# LINKS
# ─────────────────────────────────────────
all_links = get_links_for_node(node_id)
all_links_filtered = filter_links(all_links, threshold_eff)

max_links_for_selected_node = max(1, len(all_links_filtered))

top_k = top_k_placeholder.slider(
    "Top-K",
    min_value=1,
    max_value=max_links_for_selected_node,
    value=min(5, max_links_for_selected_node)
)

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Shown links", len(links))
col2.metric("Total predicted links", len(all_links_filtered))
if cluster_density is not None:
    col3.metric("Cluster cohesion", f"{cluster_density:.3f}")
else:
    col3.metric("Cluster cohesion", "n/a")
col4.metric("Cluster diversity (all predicted)", cluster_diversity_total)


# ─────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────
fig = base_scatter(node_id, links)
if links:
    fig = add_ego_edges(fig, node_id, links)

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# LINK TABLE
# ─────────────────────────────────────────
if links:
    table_df = pd.DataFrame({
        "target_node_id": [l["target"] for l in links],
        "target_name": [l["target_name"] for l in links],
        "score": [round(l["score"], 4) for l in links],
        "confidence": [score_label(l["score"]) for l in links],
        "status": [l["status"] for l in links],
        "target_cluster": [int(get_row_by_nodeid(l["target"])["cluster"]) for l in links]
    })

    st.markdown("### Ranked collaboration candidates")
    st.caption(
        "Scores indicate model-estimated collaboration potential between organizations based on historical network patterns and temporal dynamics. "
        "Higher scores suggest a stronger predicted likelihood of future collaboration."
    )
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("No links satisfy the current settings.")


# ─────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────
if metrics is not None and metrics.get("losses"):
    st.markdown("### Training loss")
    loss_df = pd.DataFrame({
        "epoch": np.arange(len(metrics["losses"])),
        "loss": metrics["losses"]
    })
    st.line_chart(loss_df.set_index("epoch"))

if metrics is not None and metrics.get("auc_history"):
    auc_df = pd.DataFrame(metrics["auc_history"])
    if not auc_df.empty:
        st.markdown("### Validation AUC over training")
        st.line_chart(auc_df.set_index("epoch"))

if metrics is not None and metrics.get("ap_history"):
    ap_df = pd.DataFrame(metrics["ap_history"])
    if not ap_df.empty:
        st.markdown("### Validation AP over training")
        st.line_chart(ap_df.set_index("epoch"))
