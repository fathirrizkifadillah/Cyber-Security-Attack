import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    path = r"C:\Users\FATHIR\.cache\kagglehub\datasets\juanschafle\cyber-attack-detection-using-network-traffic\versions\1"
    csv_file = os.path.join(path, 'cyber_attack_dataset_100000.csv')
    df = pd.read_csv(csv_file)
    df['throughput'] = ((df['src_bytes'] + df['dst_bytes']) / df['duration'])
    df['bytes_per_packet'] = df['src_bytes'] / df['packet_count']
    df['asymmetry_ratio'] = df['src_bytes'] / (df['src_bytes'] + df['dst_bytes'])
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

    tab_pred, tab_eda, tab_model = st.tabs(["Prediction Classifier", "Exploratory Data Analysis", "Model Evaluation"])

    with tab_pred:
        st.sidebar.header("Network Connection Input")
        
        duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=15)
        src_bytes = st.sidebar.slider("Source Bytes", min_value=50, max_value=10000, value=1500)
        dst_bytes = st.sidebar.slider("Destination Bytes", min_value=20, max_value=2000, value=300)
        packet_count = st.sidebar.slider("Packet Count", min_value=5, max_value=1000, value=150)
        protocol = st.sidebar.selectbox("Protocol", options=["TCP", "UDP"])
        failed_logins = st.sidebar.slider("Failed Logins", min_value=0, max_value=10, value=0)
        
        throughput = (src_bytes + dst_bytes) / duration
        bytes_per_packet = src_bytes / packet_count
        asymmetry_ratio = src_bytes / (src_bytes + dst_bytes)
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
            st.bar_chart(data=probs_df, x='Attack Type', y='Probability', use_container_width=True)

        with col2:
            st.subheader("Current Session vs. Attack Profiles")
            
            metric_to_compare = st.selectbox(
                "Select Metric for Comparison", 
                options=["Throughput (Bytes/Second)", "Average Packet Size (Bytes)", "Data Asymmetry Ratio"]
            )
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--')
            
            compare_data = []
            
            if metric_to_compare == "Throughput (Bytes/Second)":
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['throughput'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': throughput, 'Source': 'Current Input'})
                palette = {'Dataset Average': '#475569', 'Current Input': '#38bdf8'}
                y_label = "Throughput (B/s)"
            elif metric_to_compare == "Average Packet Size (Bytes)":
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['packet_size'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': bytes_per_packet, 'Source': 'Current Input'})
                palette = {'Dataset Average': '#475569', 'Current Input': '#38bdf8'}
                y_label = "Bytes / Packet"
            else:
                for label, vals in class_averages.items():
                    compare_data.append({'Type': label, 'Value': vals['asymmetry'], 'Source': 'Dataset Average'})
                compare_data.append({'Type': 'Current Input', 'Value': asymmetry_ratio, 'Source': 'Current Input'})
                palette = {'Dataset Average': '#475569', 'Current Input': '#38bdf8'}
                y_label = "Asymmetry Ratio"
                
            compare_df = pd.DataFrame(compare_data)
            sns.barplot(data=compare_df, x='Type', y='Value', hue='Source', palette=palette, errorbar=None, ax=ax)
            
            if metric_to_compare == "Throughput (Bytes/Second)":
                ax.set_yscale('log')
                
            ax.set_xlabel("Connection Type", color='#94a3b8')
            ax.set_ylabel(y_label, color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            ax.set_title(f"Comparison of {metric_to_compare}", color='#f8fafc', fontweight='bold')
            
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
                                
            st.pyplot(fig)
            
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
        
        plt.style.use('dark_background')
        
        if eda_view == "1. Distribution of Numerical Variables":
            st.write("#### Distribution Analysis of Numerical Variables")
            target_columns = ['src_bytes', 'dst_bytes', 'packet_count']
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
            fig.patch.set_facecolor('#1e293b')
            for ax, col in zip(axes, target_columns):
                ax.patch.set_facecolor('#1e293b')
                ax.grid(True, color='#334155', linestyle='--')
                sns.histplot(data=df_raw, x=col, bins=50, kde=True, ax=ax)
                ax.tick_params(colors='#94a3b8')
                ax.set_title(f'{col.replace("_", " ").title()}', color='#f8fafc', fontweight='bold')
                ax.set_xlabel(col, color='#94a3b8')
                ax.set_ylabel('Count', color='#94a3b8')
            st.pyplot(fig)
            st.markdown("""
            * Numeric variables (`src_bytes`, `dst_bytes`, and `packet_count`) show a similar distribution pattern, namely **positively skewed (right-skewed)** with a mean value that is consistently higher than the median.
            * This indicates that most of the connections in the dataset have relatively low to moderate network activity, while there are a small number of connections with very high values that form the long tail of the distribution.
            """)
            
        elif eda_view == "2. Distribution of Variables by Attack Type":
            st.write("#### Distribution of Numerical Variables by Attack Type")
            target_columns = ['src_bytes', 'dst_bytes', 'packet_count']
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
            fig.patch.set_facecolor('#1e293b')
            palette = {'Normal': '#10b981', 'DDoS': '#ef4444', 'PortScan': '#0ea5e9', 'BruteForce': '#f59f0b'}
            for ax, col in zip(axes, target_columns):
                ax.patch.set_facecolor('#1e293b')
                ax.grid(True, color='#334155', linestyle='--')
                sns.kdeplot(data=df_raw, x=col, hue='attack_type', fill=True, common_norm=False, palette=palette, alpha=0.4, linewidth=2, ax=ax)
                ax.tick_params(colors='#94a3b8')
                ax.set_title(f'Distribution of {col.replace("_", " ").title()} by Attack Type', color='#f8fafc', fontweight='bold')
                ax.set_xlabel(col, color='#94a3b8')
                ax.set_ylabel('Density', color='#94a3b8')
            st.pyplot(fig)
            
        elif eda_view == "3. Attack Types by Network Protocol":
            st.write("#### Distribution of Attack Types by Network Protocol")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--')
            palette = {'TCP': '#475569', 'UDP': '#cbd5e1'}
            sns.countplot(data=df_raw, x='attack_type', hue='protocol', palette=palette, edgecolor='black', alpha=0.9, ax=ax)
            ax.tick_params(colors='#94a3b8')
            ax.set_xlabel('Attack Type', color='#94a3b8')
            ax.set_ylabel('Number of Connections', color='#94a3b8')
            ax.set_title('Distribution of Attack Types by Network Protocol', color='#f8fafc', fontweight='bold')
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{height:,.0f}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig)
            
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
            protocol_counts = pd.crosstab(df_raw['protocol'], df_raw['attack_type'])
            protocol_pct = pd.crosstab(df_raw['protocol'], df_raw['attack_type'], normalize='index') * 100
            colors = sns.color_palette('Set2', len(df_raw['attack_type'].unique()))
            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
            fig.patch.set_facecolor('#1e293b')
            for ax in axes:
                ax.patch.set_facecolor('#1e293b')
                ax.grid(True, color='#334155', linestyle='--')
                ax.tick_params(colors='#94a3b8')
                
            protocol_counts.plot(kind='bar', stacked=True, color=colors, ax=axes[0])
            axes[0].set_title('Absolute Traffic and Attack Counts', color='#f8fafc', fontweight='bold')
            axes[0].set_xlabel('Protocol', color='#94a3b8')
            axes[0].set_ylabel('Number of Connections', color='#94a3b8')
            axes[0].set_xticklabels(protocol_counts.index, rotation=0)
            axes[0].get_legend().remove()
            for p in axes[0].patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                if height > 0:
                    axes[0].annotate(f'{int(height):,}', (x + width/2, y + height/2), 
                                     ha='center', va='center', color='white', fontweight='bold', fontsize=9)
                                     
            protocol_pct.plot(kind='bar', stacked=True, color=colors, ax=axes[1])
            axes[1].set_title('Relative Traffic and Attack Percentages', color='#f8fafc', fontweight='bold')
            axes[1].set_xlabel('Protocol', color='#94a3b8')
            axes[1].set_ylabel('Percentage (%)', color='#94a3b8')
            axes[1].set_xticklabels(protocol_pct.index, rotation=0)
            axes[1].legend(title='Traffic Type', bbox_to_anchor=(1.02, 1), loc='upper left')
            for p in axes[1].patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                if height > 0:
                    axes[1].annotate(f'{height:.1f}%', (x + width/2, y + height/2), 
                                     ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            st.pyplot(fig)
            
        elif eda_view == "6. Correlation Heatmap":
            st.write("#### Correlation Heatmap of Numerical Features")
            corr_matrix = df_raw[['duration', 'src_bytes', 'dst_bytes', 'packet_count', 'failed_logins']].corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
            ax.tick_params(colors='#94a3b8')
            ax.set_title('Correlation Heatmap', color='#f8fafc', fontweight='bold')
            st.pyplot(fig)
            
        elif eda_view == "7. Throughput by Attack Type":
            st.write("#### Throughput by Attack Type")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--')
            
            sns.barplot(data=df_raw, x='attack_type', y='throughput', palette='Set2', errorbar=None, ax=ax)
            ax.set_yscale('log')
            ax.set_ylabel("Average Throughput (Bytes/Second)", color='#94a3b8')
            ax.set_xlabel("Attack Type", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            ax.set_title('Throughput by Attack Type', color='#f8fafc', fontweight='bold')
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height * 1.15),
                                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig)
            
        elif eda_view == "8. Average Packet Size by Attack Type":
            st.write("#### Average Packet Size by Attack Type")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--')
            
            sns.barplot(data=df_raw, x='attack_type', y='bytes_per_packet', palette='Set2', errorbar=None, ax=ax)
            ax.set_ylabel("Average Packet Size (Bytes/Packet)", color='#94a3b8')
            ax.set_xlabel("Attack Type", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            ax.set_title('Average Packet Size by Attack Type', color='#f8fafc', fontweight='bold')
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 1),
                                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig)
            
        else:
            st.write("#### Data Asymmetry Ratio by Attack Type")
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.patch.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--')
            
            sns.barplot(data=df_raw, x='attack_type', y='asymmetry_ratio', palette='Set2', errorbar=None, ax=ax)
            ax.set_ylabel("Average Asymmetry Ratio (Src / Total)", color='#94a3b8')
            ax.set_xlabel("Attack Type", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            ax.set_title('Data Asymmetry Ratio by Attack Type', color='#f8fafc', fontweight='bold')
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig)

    with tab_model:
        st.subheader("Model Training & Evaluation Results")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.write("#### Confusion Matrix Comparison")
            cm_path = os.path.join('img', 'Confusion Matrix of Classification Models.png')
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Confusion matrix plot not found. Run Chapter 5 code to generate and save it.")
                
        with col_m2:
            st.write("#### Feature Importance Comparison")
            fi_path = os.path.join('img', 'Feature Importance Comparison.png')
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
            else:
                st.info("Feature importance plot not found. Run Chapter 5 code to generate and save it.")
