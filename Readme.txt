{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # EU Collaboration Project\
\
This project explores potential future collaborations in EU co-patenting networks using a heterogeneous temporal graph approach and Graph Neural Networks (GNNs). The goal is to estimate which organizations are more likely to collaborate based on past patenting activity, network structure, and temporal evolution.\
\
## Project structure\
\
### Notebooks\
The `notebooks/` folder contains the main analytical workflow of the project, including:\
\
- **data processing**: cleaning and structuring the patent data\
- **heterogeneous temporal graph construction**: building yearly graph snapshots connecting applicants, patents, inventors, and CPC classes\
- **metapath-based feature engineering**: generating applicant representations from heterogeneous graph relations\
- **clustering analysis**: identifying groups of organizations with similar structural or embedding patterns\
- **ablation study**: comparing alternative model configurations and architectural choices\
- **model development and evaluation**: training and assessing the selected temporal GNN for link prediction\
\
### app.py\
The `app.py` file contains an interactive dashboard that visualizes the model outputs and supports better informed decision-making. Rather than presenting only technical metrics, the dashboard helps users interpret predicted collaboration potential across organizations in a more accessible way.\
\
## Approach\
\
The project models innovation collaboration as a **heterogeneous temporal graph**, where different node types \'97 such as applicants, patents, inventors, and CPC classes \'97 are connected across yearly snapshots. To capture meaningful structural information, the workflow uses **metapath-based feature engineering**, which derives applicant embeddings from indirect relations in the heterogeneous graph.\
\
These features are then used in a **snapshot-based temporal GNN** to learn how collaboration structures evolve over time and to estimate the likelihood of future applicant-to-applicant links.\
\
## Selected model\
\
The selected model is a **snapshot-based temporal GNN for link prediction**. In this architecture:\
\
- a **GCN** encodes applicant graph structure at each yearly snapshot\
- a **GRU** captures the temporal evolution of node representations across years\
- an **attention-based temporal aggregation** layer weights the most informative time periods\
- an **MLP decoder** produces predicted collaboration scores between pairs of organizations\
\
This model was selected because it combines structural and temporal learning and performed best in the ablation study, making it the most suitable approach for predicting collaboration potential in an evolving innovation network.\
\
## Dashboard\
\
The dashboard in `app.py` presents the model results through an interactive network interface. It allows users to explore:\
\
- **predicted collaboration scores** between organizations, reflecting model-estimated collaboration potential\
- **network clusters**\
- **organizational positions in the embedding space**\
- **local and broader connection patterns**\
- **selected model outputs and graph-based insights** in a more interpretable visual format\
\
Overall, the dashboard complements the notebooks by translating model results into a visual and exploratory tool that can support interpretation and strategic analysis.}