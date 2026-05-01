import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from src.data_loader import load_and_preprocess_data
from src.model import ClinicalTwinGNN
from src.explain import explain_prediction
import os

# Page Config
st.set_page_config(page_title="OncoGraph X", layout="wide", page_icon="🧬")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    /* Subheaders and Titles */
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    data, df, feature_names = load_and_preprocess_data('data')
    model = ClinicalTwinGNN(in_channels=data.num_features, hidden_channels=64)
    if os.path.exists('models/clinical_twin_gnn.pth'):
        model.load_state_dict(torch.load('models/clinical_twin_gnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return data, df, feature_names, model

def create_gauge(value, title, color="#00d4ff"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        # Remove title from inside the figure to avoid visibility issues
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#3e4150"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#3e4150",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 212, 255, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        font={'family': "Arial", 'size': 18}, # Let theme handle color
        margin=dict(t=0, b=0, l=10, r=10)
    )
    return fig

def main():
    st.title("OncoGraph X")
    st.markdown("### Explainable AI for Cancer Recurrence & Survival Prediction")
    
    try:
        data, df, feature_names, model = load_resources()
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure CSV files are in the 'data' folder and 'src' scripts are correct.")
        return

    # Sidebar
    st.sidebar.header("Patient Selection")
    patient_ids = df['patient_id'].tolist()
    selected_id = st.sidebar.selectbox("Select Patient ID", patient_ids)
    node_idx = df[df['patient_id'] == selected_id].index[0]

    # Run Inference
    with torch.no_grad():
        out_rec, out_sur = model(data.x, data.edge_index)
        rec_prob = torch.softmax(out_rec[node_idx], dim=-1)[1].item()
        sur_rate = out_sur[node_idx].item()

    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Recurrence Probability")
        st.plotly_chart(create_gauge(rec_prob, "Recurrence Probability", color="#ff4b4b"), use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Estimated Survival Rate")
        st.plotly_chart(create_gauge(sur_rate, "Estimated Survival Rate", color="#00d4ff"), use_container_width=True)

    # Explanation Section
    st.markdown("---")
    st.header("🔍 Clinical Interpretation")
    
    expl_col, graph_col = st.columns([1, 2])
    
    with st.spinner("Generating explanations..."):
        feat_imp, edge_imp, edge_idx = explain_prediction(model, data, node_idx)

    with expl_col:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feat_imp
        }).sort_values(by='Importance', ascending=False).head(10)
        
        fig_feat = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                          color='Importance', color_continuous_scale='Viridis')
        fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_feat, use_container_width=True)

    with graph_col:
        st.subheader("Clinical Twin Map")
        st.info("Showing the most influential 'Twins' (similar patients) for this prediction.")
        
        # Build Pyvis Graph
        net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white", notebook=False)
        
        # Add target node
        net.add_node(int(node_idx), label=f"Patient {selected_id}", color="#ff4b4b", size=30)
        
        # Find top 5 influential edges
        connected_edges = (edge_idx[0] == node_idx) | (edge_idx[1] == node_idx)
        relevant_edges = edge_idx[:, connected_edges]
        relevant_weights = edge_imp[connected_edges]
        
        top_indices = np.argsort(relevant_weights)[-5:]
        
        for idx in top_indices:
            u, v = relevant_edges[:, idx]
            neighbor_idx = v if u == node_idx else u
            neighbor_idx = int(neighbor_idx)
            neighbor_id = df.iloc[neighbor_idx]['patient_id']
            neighbor_survival = df.iloc[neighbor_idx]['survival_status']
            
            color = "#00d4ff" if neighbor_survival == 'living' else "#7f8c8d"
            net.add_node(neighbor_idx, label=f"Twin {neighbor_id}", color=color, size=20)
            net.add_edge(int(node_idx), neighbor_idx, value=float(relevant_weights[idx]), title=f"Weight: {relevant_weights[idx]:.2f}")

        net.save_graph("clinical_twin_net.html")
        HtmlFile = open("clinical_twin_net.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=450)
        
        # Legend
        st.markdown("""
        **Legend:**
        - <span style='color:#ff4b4b'>●</span> **Target Patient**
        - <span style='color:#00d4ff'>●</span> **Clinical Twin (Historical Survival: Living)**
        - <span style='color:#7f8c8d'>●</span> **Clinical Twin (Historical Survival: Deceased)**
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Why the Clinical Twin Logic?
    Traditional oncology models often act as 'Black Boxes'. The **Clinical Twin** approach provides transparency by:
    1. **Similarity-Driven Inference**: Predictions are not just numbers; they are based on how similar patients with matched tumor sites and stages responded historically.
    2. **Graph-Based Evidence**: By visualizing the 'Twins' in the network, clinicians can see the specific historical cases that the AI is using as reference.
    3. **Feature Specificity**: Understanding if blood work (Leukocytes) or pathology (Infiltration depth) is driving the risk allows for more targeted clinical interventions.
    """)

if __name__ == "__main__":
    main()
