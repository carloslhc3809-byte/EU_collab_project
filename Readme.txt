# EU Collaboration Project

This project predicts potential future collaborations in EU co-patenting networks using graph-based machine learning. It combines heterogeneous patent-network structure, metapath-based feature engineering, and a temporal Graph Neural Network (GNN) to estimate which organizations are most likely to collaborate in the future.

The project is focused on EU Life Sciences patenting activity, with particular attention to collaboration patterns in genetic engineering and mutation-related technologies.

## Project outputs

- **Interactive dashboard:** `app.py`
- **Prefect-orchestrated ML pipeline:** `run_pipeline.py`
- **Project report:** `Project_report.pdf`

## Overview

The project studies innovation collaboration through patent co-application networks in the EU Life Sciences domain. The main objective is to identify promising future organization-to-organization collaborations based on historical patenting activity, network structure, and temporal evolution.

The workflow uses a heterogeneous graph representation linking applicants, patents, inventors, and CPC classes. From this structure, metapath-based embeddings are generated for applicants and used as inputs to a temporal link prediction model.

The final outputs are exported for interactive exploration through a Streamlit dashboard.

## Methodological approach

The project follows four main methodological steps:

1. **Data preparation**  
   Patent data are cleaned, harmonized, and transformed into graph-ready structures.

2. **Heterogeneous graph construction**  
   Yearly graph snapshots are built by connecting applicants, patents, inventors, and CPC classes.

3. **Metapath-based feature engineering**  
   Applicant representations are learned from heterogeneous relational paths, capturing structural proximity and complementarity across the patent collaboration network.

4. **Temporal link prediction**  
   A snapshot-based temporal GNN models how applicant collaboration patterns evolve over time and estimates the likelihood of future co-patenting links.

## Selected model

The predictive model is an **applicant-level temporal GNN for link prediction**. In this architecture:

- a **GCN** encodes applicant graph structure at each yearly snapshot
- a **GRU** captures temporal evolution across snapshots
- an **attention-based temporal aggregation layer** emphasizes the most informative time periods
- an **MLP decoder** produces collaboration scores between pairs of organizations

This design was chosen to balance predictive performance, interpretability, and computational feasibility for an evolving collaboration network.

## Orchestrated ML workflow

The project includes a Prefect-orchestrated pipeline in `run_pipeline.py` to make the end-to-end graph machine learning workflow more reproducible and observable.

The pipeline structures the workflow into explicit stages:

1. Configuration, directory setup, and device detection
2. Patent dataset loading and applicant-name preparation
3. Temporal embedding generation
4. Temporal tensor construction
5. Temporal GNN training
6. Export of model artifacts and dashboard-ready outputs

This separates the model production workflow from the Streamlit dashboard. The pipeline generates the analytical outputs, while `app.py` provides the interactive interpretation layer.

In practical terms:

```text
run_pipeline.py → produces model artifacts and dashboard-ready outputs
app.py          → visualizes and explores those outputs