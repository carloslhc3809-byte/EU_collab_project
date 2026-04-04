# EU Collaboration Project

This project predicts potential future collaborations in EU co-patenting networks using graph-based machine learning. It combines heterogeneous patent-network structure, metapath-based feature engineering, and a temporal Graph Neural Network (GNN) to estimate which organizations are most likely to collaborate in the future.

## Project outputs

- **Interactive dashboard:** `app.py`
- **Pipeline entry point:** `run_pipeline.py`
- **Project report:** `report.pdf`

## Overview

The project studies innovation collaboration through patent co-application networks in the EU Life Sciences domain. The main objective is to identify promising future organization-to-organization collaborations based on historical patenting activity, network structure, and temporal evolution.

The workflow uses a heterogeneous graph representation linking applicants, patents, inventors, and CPC classes. From this structure, metapath-based embeddings are generated for applicants and used as inputs to a temporal link prediction model.

## Methodological approach

The project follows four main steps:

1. **Data preparation**  
   Patent data are cleaned, harmonized, and transformed into graph-ready structures.

2. **Heterogeneous graph construction**  
   Yearly graph snapshots are built connecting applicants, patents, inventors, and CPC classes.

3. **Metapath-based feature engineering**  
   Applicant representations are learned from heterogeneous relational paths.

4. **Temporal link prediction**  
   A snapshot-based temporal GNN models how applicant collaboration patterns evolve over time and estimates the likelihood of future co-patenting links.

## Selected model

The predictive model is an **applicant-level temporal GNN for link prediction**. In this architecture:

- a **GCN** encodes applicant graph structure at each yearly snapshot
- a **GRU** captures temporal evolution across snapshots
- an **attention-based temporal aggregation layer** emphasizes the most informative time periods
- an **MLP decoder** produces collaboration scores between pairs of organizations

This design was chosen to balance predictive performance, interpretability, and computational feasibility for an evolving collaboration network.

## Dashboard

The dashboard in `app.py` provides an interactive interface to explore model outputs. It allows users to inspect:

- predicted collaboration scores between organizations
- applicant positions in the embedding space
- cluster membership and local network context
- model outputs in a more accessible visual format

The dashboard is intended as a practical interpretation layer on top of the modeling pipeline.

## Repository structure

```text
EU_collaboration_project/
├─ app.py
├─ run_pipeline.py
├─ README.md
├─ report.pdf
├─ requirements.txt
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ export.py
│  ├─ features.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ notebooks/
│  └─ Notebook_final.ipynb
└─ data/
   ├─ raw/
   └─ processed/