import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}

.main {
    background-color: #0f172a;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid #334155;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 500;
    padding: 10px 20px;
    background-color: transparent;
    border-radius: 6px 6px 0 0;
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    background-color: #1e293b !important;
    border-color: #334155 #334155 transparent #334155 !important;
}

.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.prediction-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.prediction-normal {
    background-color: rgba(16, 185, 129, 0.08);
    border: 1px solid #10b981;
}

.prediction-ddos {
    background-color: rgba(239, 68, 68, 0.08);
    border: 1px solid #ef4444;
}

.prediction-portscan {
    background-color: rgba(14, 165, 233, 0.08);
    border: 1px solid #0ea5e9;
}

.prediction-bruteforce {
    background-color: rgba(245, 158, 11, 0.08);
    border: 1px solid #f59f0b;
}

.metric-container {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 15px;
}

.metric-box {
    text-align: center;
    background: #1e293b;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #334155;
    flex: 1 1 150px;
}

.metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
    margin-top: 2px;
}

.metric-label {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.custom-title {
    font-size: 32px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 4px;
}

.custom-desc {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'rf_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    le_path = os.path.join('models', 'le.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        
    return model, scaler, le

@st.cache_data
def load_dataset():
    import kagglehub
    path = kagglehub.dataset_download("juanschafle/cyber-attack-detection-using-network-traffic")
    csv_file = os.path.join(path, 'cyber_attack_dataset_100000.csv')
    df = pd.read_csv(csv_file)
    # Prevent division by zero for derived features
    df['throughput'] = (df['src_bytes'] + df['dst_bytes']) / df['duration'].replace(0, 0.001)
    df['bytes_per_packet'] = df['src_bytes'] / df['packet_count'].replace(0, 1)
    df['asymmetry_ratio'] = df['src_bytes'] / (df['src_bytes'] + df['dst_bytes']).replace(0, 1)
    return df

try:
    model, scaler, le = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    error_msg = str(e)

st.markdown('<div class="custom-title">Network Intrusion Detection & Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-desc">Real-time intrusion detection and network security analytics powered by Random Forest.</div>', unsafe_allow_html=True)

if not assets_loaded:
    st.error(f"Error loading model objects. Make sure the 'models' directory exists and contains the serialized files. Details: {error_msg}")
else:
    df_raw = load_dataset()
    class_averages = {
        'DDoS': {'throughput': 3082.15, 'packet_size': 16.58, 'asymmetry': 0.96},
        'Normal': {'throughput': 120.09, 'packet_size': 80.77, 'asymmetry': 0.57},
        'PortScan': {'throughput': 113.20, 'packet_size': 1.98, 'asymmetry': 0.69},
        'BruteForce': {'throughput': 38.36, 'packet_size': 14.19, 'asymmetry': 0.65}
    }
    attack_colors = {
        'Normal': '#10b981',
        'DDoS': '#ef4444',
        'PortScan': '#0ea5e9',
        'BruteForce': '#f59f0b'
    }

    tab_pred, tab_eda, tab_model = st.tabs(["Prediction Classifier", "Exploratory Data Analysis", "Model Evaluation"])

    with tab_pred:
        st.sidebar.header("Network Connection Input")
        
        duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=15)
        src_bytes = st.sidebar.slider("Source Bytes", min_value=50, max_value=10000, value=1500)
        dst_bytes = st.sidebar.slider("Destination Bytes", min_value=20, max_value=2000, value=300)
        packet_count = st.sidebar.slider("Packet Count", min_value=5, max_value=1000, value=150)
        protocol = st.sidebar.selectbox("Protocol", options=["TCP", "UDP"])
        failed_logins = st.sidebar.slider("Failed Logins", min_value=0, max_value=10, value=0)
        
        throughput = (src_bytes + dst_bytes) / max(duration, 0.001)
        bytes_per_packet = src_bytes / max(packet_count, 1)
        asymmetry_ratio = src_bytes / max(src_bytes + dst_bytes, 1)
        protocol_num = 1 if protocol == "TCP" else 0
        
        input_data = pd.DataFrame([{
            'duration': duration,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'packet_count': packet_count,
            'protocol': protocol_num,
            'failed_logins': failed_logins,
            'throughput': throughput,
            'bytes_per_packet': bytes_per_packet,
            'asymmetry_ratio': asymmetry_ratio
        }])

        input_scaled = scaler.transform(input_data)
        
        pred_encoded = model.predict(input_scaled)[0]
        pred_label = le.classes_[pred_encoded]
        
        pred_probs = model.predict_proba(input_scaled)[0]

        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("Classifier Prediction")
            
            if pred_label == 'Normal':
                st.markdown("""
                <div class="card prediction-normal">
                    <div class="prediction-title" style="color: #10b981;">Normal Traffic Detected</div>
                    <p style="color: #94a3b8; margin: 0; font-size: 15px;">The connection behaves normally without anomalous payload patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == 'DDoS':
                st.markdown("""
                <div class="card prediction-ddos">
                    <div class="prediction-title" style="color: #ef4444;">DDoS Attack Detected</div>
                    <p style="color: #94a3b8; margin: 0; font-size: 15px;">High-volume data transfer asymmetry and low-duration request flooding.</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == 'PortScan':
                st.markdown("""
                <div class="card prediction-portscan">
                    <div class="prediction-title" style="color: #0ea5e9;">PortScan Probe Detected</div>
                    <p style="color: #94a3b8; margin: 0; font-size: 15px;">High packet count with extremely small bytes per packet (SYN probes).</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == 'BruteForce':
                st.markdown("""
                <div class="card prediction-bruteforce">
                    <div class="prediction-title" style="color: #f59f0b;">BruteForce Attack Detected</div>
                    <p style="color: #94a3b8; margin: 0; font-size: 15px;">Repeated failed authentication attempts detected.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Threat Severity Index (Gauge Chart)
            risk_scores = {
                'Normal': 10,
                'PortScan': 45,
                'BruteForce': 75,
                'DDoS': 95
            }
            risk_val = risk_scores.get(pred_label, 0)
            risk_color = attack_colors.get(pred_label, '#94a3b8')
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Threat Severity Index", 'font': {'color': '#f8fafc', 'size': 14, 'family': 'Inter'}},
                number={'font': {'color': risk_color, 'size': 32, 'family': 'Inter'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155', 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': risk_color},
                    'bgcolor': '#1e293b',
                    'borderwidth': 1,
                    'bordercolor': '#334155',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [30, 60], 'color': 'rgba(14, 165, 233, 0.15)'},
                        {'range': [60, 85], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=180,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
                
            st.subheader("Derived Features")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value">{throughput:.2f} B/s</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Bytes/Packet</div>
                    <div class="metric-value">{bytes_per_packet:.2f} B</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Asymmetry Ratio</div>
                    <div class="metric-value">{asymmetry_ratio:.2%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Prediction Confidence")
            
            probs_df = pd.DataFrame({
                'Attack Type': le.classes_,
                'Probability': pred_probs
            })
            
            fig_probs = px.bar(
                probs_df,
                x='Attack Type',
                y='Probability',
                color='Attack Type',
                color_discrete_map=attack_colors,
                text='Probability'
            )
            
            fig_probs.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                height=300,
                xaxis=dict(
                    title="",
                    tickfont=dict(color='#94a3b8', size=11),
                    gridcolor='#1e293b'
                ),
                yaxis=dict(
                    title=dict(text="Probability", font=dict(color='#94a3b8', size=11)),
                    tickfont=dict(color='#94a3b8', size=11),
                    gridcolor='#1e293b',
                    range=[0, 1.05]
                ),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_probs.update_traces(
                texttemplate='%{y:.1%}',
                textposition='outside',
                cliponaxis=False
            )
            st.plotly_chart(fig_probs, use_container_width=True)

        with col2:
            st.subheader("Current Session vs. Attack Profiles")
            
            metric_to_compare = st.selectbox(
                "Select Metric for Comparison", 
                options=["Throughput (Bytes/Second)", "Average Packet Size (Bytes)", "Data Asymmetry Ratio"]
            )
            
            compare_data = []
            
            if metric_to_compare == "Throughput (Bytes/Second)":
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['throughput'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': throughput, 'Source': 'Current Input'})
                y_label = "Throughput (B/s)"
                log_scale = True
            elif metric_to_compare == "Average Packet Size (Bytes)":
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['packet_size'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': bytes_per_packet, 'Source': 'Current Input'})
                y_label = "Bytes / Packet"
                log_scale = False
            else:
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['asymmetry'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': asymmetry_ratio, 'Source': 'Current Input'})
                y_label = "Asymmetry Ratio"
                log_scale = False
                
            compare_df = pd.DataFrame(compare_data)
            
            fig_comp = px.bar(
                compare_df,
                x='Type',
                y='Value',
                color='Source',
                barmode='group',
                color_discrete_map={'Dataset Average': '#475569', 'Current Input': '#38bdf8'},
                text='Value'
            )
            
            fig_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                height=350,
                xaxis=dict(
                    title=dict(text="Connection Type", font=dict(color='#94a3b8', size=11)),
                    tickfont=dict(color='#94a3b8', size=11),
                    gridcolor='#1e293b'
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color='#94a3b8', size=11)),
                    tickfont=dict(color='#94a3b8', size=11),
                    gridcolor='#1e293b',
                    type='log' if log_scale else 'linear'
                ),
                legend=dict(
                    title="",
                    font=dict(color='#94a3b8', size=10),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_comp.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                cliponaxis=False
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; margin-top: 20px;">
                <strong style="color: #f8fafc;">Security Insight Context:</strong>
                <ul style="color: #94a3b8; margin: 8px 0 0 0; padding-left: 20px; font-size: 14px;">
                    <li><strong>DDoS</strong> is marked by extremely high throughput (log scale) and near-100% asymmetry.</li>
                    <li><strong>Normal</strong> traffic features larger payloads (Bytes/Packet) and balanced asymmetry (~50%).</li>
                    <li><strong>PortScan</strong> utilizes micro-packets (low Bytes/Packet) to bypass detection.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_eda:
        st.subheader("Exploratory Data Analysis (EDA) Insights")
        
        eda_view = st.selectbox(
            "Select Analysis Topic",
            options=[
                "1. Distribution of Numerical Variables",
                "2. Distribution of Variables by Attack Type",
                "3. Attack Types by Network Protocol",
                "4. Multivariate Pairplot by Attack Type",
                "5. Protocol Traffic & Attack Comparison",
                "6. Correlation Heatmap",
                "7. Throughput by Attack Type",
                "8. Average Packet Size by Attack Type",
                "9. Data Asymmetry Ratio by Attack Type"
            ]
        )
        
        if eda_view == "1. Distribution of Numerical Variables":
            st.write("#### Distribution Analysis of Numerical Variables")
            selected_var = st.selectbox("Select Variable to Plot Distribution", options=['src_bytes', 'dst_bytes', 'packet_count', 'duration'])
            
            fig = px.histogram(
                df_raw, 
                x=selected_var, 
                nbins=50, 
                marginal="box",
                title=f"Distribution of {selected_var.replace('_', ' ').title()}",
                color_discrete_sequence=['#38bdf8']
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                title_font=dict(size=14, color='#f8fafc')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            * Numeric variables (`src_bytes`, `dst_bytes`, and `packet_count`) show a similar distribution pattern, namely **positively skewed (right-skewed)** with a mean value that is consistently higher than the median.
            * This indicates that most of the connections in the dataset have relatively low to moderate network activity, while there are a small number of connections with very high values that form the long tail of the distribution.
            """)
            
        elif eda_view == "2. Distribution of Variables by Attack Type":
            st.write("#### Distribution of Numerical Variables by Attack Type")
            selected_var = st.selectbox("Select Variable for Distribution Comparison", options=['src_bytes', 'dst_bytes', 'packet_count', 'duration'])
            
            fig = px.violin(
                df_raw, 
                y=selected_var, 
                x='attack_type', 
                color='attack_type',
                box=True, 
                points=False,
                color_discrete_map=attack_colors,
                title=f"Comparison of {selected_var.replace('_', ' ').title()} by Attack Type"
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                xaxis=dict(title="", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                title_font=dict(size=14, color='#f8fafc')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_view == "3. Attack Types by Network Protocol":
            st.write("#### Distribution of Attack Types by Network Protocol")
            
            fig = px.histogram(
                df_raw, 
                x='attack_type', 
                color='protocol', 
                barmode='group',
                color_discrete_map={'TCP': '#38bdf8', 'UDP': '#f59f0b'},
                text_auto='.0f',
                title="Attack Types by Network Protocol"
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(title="Attack Type", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                yaxis=dict(title="Number of Connections", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                title_font=dict(size=14, color='#f8fafc')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_view == "4. Multivariate Pairplot by Attack Type":
            st.write("#### Multivariate Pairplot by Attack Type")
            pairplot_path = os.path.join('img', 'Multivariate Pairplot by Attack Type.png')
            if os.path.exists(pairplot_path):
                st.image(pairplot_path, use_container_width=True)
            else:
                st.info("Multivariate pairplot image not found. Please run the notebook cells to generate and save it.")
            st.markdown("""
            **Pairplot Insights:**
            1. **BruteForce Attack Profile (Failed Logins as a Key Discriminator)**:
               * **BruteForce** is the **only** class that exhibits `failed_logins` greater than zero (ranging from 1 to 10).
               * For all other classes, the failed login count is strictly zero. This makes `failed_logins` a perfect single-feature predictor for BruteForce attacks.
            2. **DDoS Attack Profile (High Volume, Low Latency)**:
               * **DDoS** is clearly clustered with high `src_bytes` (3,000 to 10,000 bytes) and high `packet_count` (200 to 1,000 packets).
               * Despite the massive data volume, DDoS traffic has a very short `duration` (mostly < 5 seconds).
            """)
            
        elif eda_view == "5. Protocol Traffic & Attack Comparison":
            st.write("#### Traffic and Attack Comparison by Protocol")
            
            protocol_counts = pd.crosstab(df_raw['protocol'], df_raw['attack_type']).reset_index()
            protocol_counts_melted = protocol_counts.melt(id_vars='protocol', var_name='Attack Type', value_name='Count')
            
            total_by_proto = protocol_counts_melted.groupby('protocol')['Count'].transform('sum')
            protocol_counts_melted['Percentage'] = (protocol_counts_melted['Count'] / total_by_proto) * 100
            
            col_abs, col_rel = st.columns(2)
            
            with col_abs:
                fig_abs = px.bar(
                    protocol_counts_melted, 
                    x='protocol', 
                    y='Count', 
                    color='Attack Type',
                    color_discrete_map=attack_colors,
                    title="Absolute Traffic and Attack Counts",
                    text_auto='.0f'
                )
                fig_abs.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    xaxis=dict(title="Protocol", tickfont=dict(color='#94a3b8')),
                    yaxis=dict(title="Number of Connections", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                    legend=dict(font=dict(color='#94a3b8'))
                )
                st.plotly_chart(fig_abs, use_container_width=True)
                
            with col_rel:
                fig_rel = px.bar(
                    protocol_counts_melted, 
                    x='protocol', 
                    y='Percentage', 
                    color='Attack Type',
                    color_discrete_map=attack_colors,
                    title="Relative Traffic and Attack Percentages",
                    text_auto='.1f'
                )
                fig_rel.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    xaxis=dict(title="Protocol", tickfont=dict(color='#94a3b8')),
                    yaxis=dict(title="Percentage (%)", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                    legend=dict(font=dict(color='#94a3b8'))
                )
                st.plotly_chart(fig_rel, use_container_width=True)
            
        elif eda_view == "6. Correlation Heatmap":
            st.write("#### Correlation Heatmap of Numerical Features")
            corr_matrix = df_raw[['duration', 'src_bytes', 'dst_bytes', 'packet_count', 'failed_logins']].corr()
            
            fig = px.imshow(
                corr_matrix, 
                text_auto='.2f', 
                color_continuous_scale='RdBu_r', 
                zmin=-1, 
                zmax=1,
                title="Correlation Heatmap"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(tickfont=dict(color='#94a3b8')),
                yaxis=dict(tickfont=dict(color='#94a3b8')),
                coloraxis_colorbar=dict(tickfont=dict(color='#94a3b8'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_view == "7. Throughput by Attack Type":
            st.write("#### Throughput by Attack Type")
            df_grouped = df_raw.groupby('attack_type')['throughput'].mean().reset_index()
            
            fig = px.bar(
                    df_grouped,
                    x='attack_type',
                    y='throughput',
                    color='attack_type',
                    color_discrete_map=attack_colors,
                    text_auto='.2f',
                    title="Average Throughput by Attack Type"
                )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                xaxis=dict(title="", tickfont=dict(color='#94a3b8')),
                yaxis=dict(title="Throughput (Bytes/Second, Log Scale)", gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), type='log')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_view == "8. Average Packet Size by Attack Type":
            st.write("#### Average Packet Size by Attack Type")
            df_grouped = df_raw.groupby('attack_type')['bytes_per_packet'].mean().reset_index()
            
            fig = px.bar(
                df_grouped,
                x='attack_type',
                y='bytes_per_packet',
                color='attack_type',
                color_discrete_map=attack_colors,
                text_auto='.2f',
                title="Average Packet Size by Attack Type"
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                xaxis=dict(title="", tickfont=dict(color='#94a3b8')),
                yaxis=dict(title="Packet Size (Bytes)", gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.write("#### Data Asymmetry Ratio by Attack Type")
            df_grouped = df_raw.groupby('attack_type')['asymmetry_ratio'].mean().reset_index()
            
            fig = px.bar(
                df_grouped,
                x='attack_type',
                y='asymmetry_ratio',
                color='attack_type',
                color_discrete_map=attack_colors,
                text_auto='.2%',
                title="Data Asymmetry Ratio by Attack Type"
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                xaxis=dict(title="", tickfont=dict(color='#94a3b8')),
                yaxis=dict(title="Asymmetry Ratio (Src / Total)", gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), range=[0, 1.1])
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_model:
        st.subheader("Model Training & Evaluation Results")
        
        # Model Comparison Cards
        st.markdown("""
        <div style="display: flex; gap: 16px; margin-bottom: 24px;">
            <div class="card" style="flex: 1; text-align: center; margin-bottom: 0;">
                <div style="font-size: 13px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Decision Tree Classifier</div>
                <div style="font-size: 32px; font-weight: 800; color: #38bdf8; margin: 4px 0;">99.99%</div>
                <div style="font-size: 12px; color: #64748b;">Testing Accuracy (20k Samples)</div>
            </div>
            <div class="card" style="flex: 1; text-align: center; margin-bottom: 0; border-color: #10b981; background-color: rgba(16, 185, 129, 0.02)">
                <div style="font-size: 13px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Random Forest Classifier</div>
                <div style="font-size: 32px; font-weight: 800; color: #10b981; margin: 4px 0;">100.00%</div>
                <div style="font-size: 12px; color: #64748b;">Testing Accuracy (20k Samples)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.write("#### Confusion Matrix")
            cm_path = os.path.join('img', 'Confusion Matrix of Classification Models.png')
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Confusion matrix plot not found. Run Chapter 5 code in the notebook to generate and save it.")
                
        with col_m2:
            st.write("#### Feature Importance (Dynamic)")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = ['duration', 'src_bytes', 'dst_bytes', 'packet_count', 'protocol', 'failed_logins', 'throughput', 'bytes_per_packet', 'asymmetry_ratio']
                clean_names = [f.replace('_', ' ').title() for f in feature_names]
                
                fi_df = pd.DataFrame({
                    'Feature': clean_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig_fi = px.bar(
                    fi_df,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    title='Random Forest Classifier Feature Importance',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    coloraxis_showscale=False,
                    xaxis=dict(title="Importance Score", gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                    yaxis=dict(title="", tickfont=dict(color='#94a3b8')),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance plot cannot be dynamically generated from the loaded model.")
