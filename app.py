"""
🌞 SOLAR PANEL FAULT DETECTION & MAINTENANCE SYSTEM
Web Application untuk monitoring dan maintenance recommendations
"""

import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from inference_helper import SolarPanelInference
from maintenance_guide import (
    MAINTENANCE_GUIDE, 
    get_maintenance_info, 
    get_priority_color,
    get_urgency_priority
)

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="☀️ Solar Panel Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM STYLING ==================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Status indicators */
    .status-good { color: #27ae60; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-critical { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================== INITIALIZE SESSION ==================
if 'inference_model' not in st.session_state:
    st.session_state.inference_model = None
    st.session_state.prediction_history = []

# Load model
if st.session_state.inference_model is None:
    with st.spinner("🔄 Loading YOLO Model..."):
        st.session_state.inference_model = SolarPanelInference()

model = st.session_state.inference_model

# ================== MAIN APP ==================
with st.sidebar:
    st.image("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect fill='%23FFD700' width='100' height='100'/%3E%3Ccircle cx='50' cy='30' r='25' fill='%23FFA500'/%3E%3Crect x='20' y='55' width='15' height='35' fill='%23333'/%3E%3Crect x='65' y='55' width='15' height='35' fill='%23333'/%3E%3C/svg%3E", width=100)
    
    st.markdown("### 🌞 Solar Panel Monitor")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Menu",
        options=["🔍 Predict", "📊 Dashboard", "📚 Maintenance", "ℹ️ About"],
        icons=["search", "bar-chart", "book", "info-circle"],
        menu_icon="menu-button-wide",
        default_index=0
    )
    
    st.markdown("---")
    st.markdown("### Model Info")
    if model and model.model:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "98.06%")
        with col2:
            st.metric("Classes", "6")
        st.caption("YOLOv8s-Classification")
        st.caption(f"Device: {model.device}")

# ================== PAGE 1: PREDICTION ==================
if selected == "🔍 Predict":
    st.markdown('<div class="header-title">🔍 Panel Condition Detection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Select panel image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload photo of solar panel untuk diagnosis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🚀 Analyze Panel", use_container_width=True, type="primary"):
                with st.spinner("🔬 Analyzing panel..."):
                    result = model.predict(image)
                
                if result.get('success'):
                    predicted_class = result['class']
                    confidence = result['confidence']
                    
                    # Save to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'class': predicted_class,
                        'confidence': confidence,
                        'image': image
                    })
                    
                    # Store result for right column
                    st.session_state.last_result = result
    
    # ================== RESULT DISPLAY ==================
    with col2:
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            predicted_class = result['class']
            confidence = result['confidence']
            
            st.markdown("### 📋 Analysis Result")
            
            # Status indicator
            urgency = get_urgency_priority(predicted_class)
            color = get_priority_color(predicted_class)
            
            if urgency >= 4:
                st.markdown(f'<div class="metric-card" style="border-left: 5px solid {color}"><h3 class="status-critical">{predicted_class}</h3></div>', 
                          unsafe_allow_html=True)
            elif urgency >= 2:
                st.markdown(f'<div class="metric-card" style="border-left: 5px solid {color}"><h3 class="status-warning">{predicted_class}</h3></div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card" style="border-left: 5px solid {color}"><h3 class="status-good">{predicted_class}</h3></div>', 
                          unsafe_allow_html=True)
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                title={'text': "Confidence Score"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 5 predictions
            with st.expander("📊 Top 5 Predictions", expanded=True):
                top5_data = pd.DataFrame(
                    result['top5'],
                    columns=['Class', 'Confidence']
                )
                top5_data['Confidence %'] = top5_data['Confidence'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(top5_data[['Class', 'Confidence %']], use_container_width=True, hide_index=True)
                
                # Visualization
                fig_top5 = px.bar(
                    x=[c[1]*100 for c in result['top5']],
                    y=[c[0] for c in result['top5']],
                    orientation='h',
                    labels={'x': 'Confidence %', 'y': 'Class'},
                    color=[c[1]*100 for c in result['top5']],
                    color_continuous_scale='RdYlGn'
                )
                fig_top5.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_top5, use_container_width=True)

# ================== PAGE 2: DASHBOARD ==================
elif selected == "📊 Dashboard":
    st.markdown('<div class="header-title">📊 System Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.prediction_history:
        history = st.session_state.prediction_history
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scans", len(history))
        
        with col2:
            healthy = sum(1 for h in history if h['class'] == 'Clean')
            st.metric("Healthy Panels", healthy)
        
        with col3:
            issues = sum(1 for h in history if get_urgency_priority(h['class']) >= 3)
            st.metric("Issues Found", issues)
        
        with col4:
            avg_conf = np.mean([h['confidence'] for h in history])
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        
        st.markdown("---")
        
        # Classification distribution
        col1, col2 = st.columns(2)
        
        with col1:
            class_counts = pd.Series([h['class'] for h in history]).value_counts()
            
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                labels={'x': 'Panel Condition', 'y': 'Count'},
                color=class_counts.index,
                color_discrete_map={cls: get_priority_color(cls) for cls in class_counts.index}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                color=class_counts.index,
                color_discrete_map={cls: get_priority_color(cls) for cls in class_counts.index}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        st.markdown("### 📈 Scan Timeline")
        timeline_data = pd.DataFrame({
            'Time': [h['timestamp'] for h in history],
            'Class': [h['class'] for h in history],
            'Confidence': [h['confidence'] for h in history]
        })
        
        fig = px.scatter(
            timeline_data,
            x='Time',
            y='Confidence',
            color='Class',
            color_discrete_map={cls: get_priority_color(cls) for cls in timeline_data['Class'].unique()},
            size_max=15,
            hover_data=['Class', 'Confidence']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 No prediction history yet. Go to 'Predict' tab to start analyzing panels!")

# ================== PAGE 3: MAINTENANCE GUIDE ==================
elif selected == "📚 Maintenance":
    st.markdown('<div class="header-title">📚 Maintenance & Care Guide</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_class = st.selectbox(
            "Select Panel Condition",
            options=list(MAINTENANCE_GUIDE.keys()),
            help="Choose a condition to view maintenance recommendations"
        )
    
    with col2:
        urgency = get_urgency_priority(selected_class)
        urgency_text = ["Low", "Low", "Medium", "High", "High", "CRITICAL"][min(urgency, 5)]
        color = get_priority_color(selected_class)
        st.markdown(f'<h4 style="color: {color}">Urgency: {urgency_text}</h4>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get maintenance info
    maintenance = get_maintenance_info(selected_class)
    
    if maintenance:
        # Status & Description
        col1, col2 = st.columns([1, 2])
        
        with col1:
            status = maintenance.get('status', 'Unknown')
            st.markdown(f"### {status}")
        
        with col2:
            description = maintenance.get('description', '')
            st.markdown(f"**{description}**")
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Schedule", maintenance.get('maintenance_schedule', 'N/A'))
        
        with col2:
            st.metric("Efficiency Loss", maintenance.get('estimated_efficiency_loss', 'N/A'))
        
        with col3:
            if len(maintenance.get('recommended_actions', [])) > 0:
                st.metric("Actions", len(maintenance.get('recommended_actions', [])))
        
        st.markdown("---")
        
        # Recommended actions
        st.markdown("### 🔧 Recommended Actions")
        
        for i, action_info in enumerate(maintenance.get('recommended_actions', []), 1):
            with st.expander(f"**{i}. {action_info['action']}**", expanded=i==1):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**⏰ Frequency:**\n{action_info['frequency']}")
                
                with col2:
                    st.markdown(f"**💰 Cost:**\n{action_info['cost']}")
                
                with col3:
                    st.markdown(f"**📋 Details:**")
                
                # Details
                details = action_info.get('details', [])
                if isinstance(details, list):
                    for detail in details:
                        st.markdown(f"• {detail}")
                else:
                    st.markdown(details)
        
        # Quick action items
        st.markdown("---")
        st.markdown("### ✅ Quick Checklist")
        
        actions_list = maintenance.get('actions', [])
        for action in actions_list:
            st.markdown(f"- {action}")

# ================== PAGE 4: ABOUT ==================
elif selected == "ℹ️ About":
    st.markdown('<div class="header-title">ℹ️ About This System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌞 Solar Panel Fault Detection System
        
        Sistem berbasis AI untuk mendeteksi kondisi panel surya dan memberikan rekomendasi perawatan.
        
        #### 🤖 Technology
        - **Model**: YOLOv8s-Classification
        - **Accuracy**: 98.06%
        - **Framework**: PyTorch + Ultralytics
        - **Interface**: Streamlit
        
        #### 📊 Features
        ✅ Real-time panel analysis  
        ✅ 6-class fault detection  
        ✅ Automated maintenance recommendations  
        ✅ Prediction history & analytics  
        ✅ Confidence scoring  
        
        #### 🔬 Classification Categories
        1. **Clean** - Panel dalam kondisi sempurna
        2. **Dusty** - Panel tertutup debu
        3. **Bird-drop** - Kotoran burung
        4. **Snow-Covered** - Tertutup salju
        5. **Electrical-damage** - Kerusakan listrik
        6. **Physical-Damage** - Kerusakan fisik
        
        #### 💡 How to Use
        1. Upload foto panel surya (JPG/PNG)
        2. Sistem akan menganalisis kondisi panel
        3. Lihat hasil prediksi dan confidence score
        4. Baca maintenance recommendations untuk tindakan selanjutnya
        5. Track history di Dashboard
        
        #### 🎯 Use Cases
        - Field inspection oleh teknisi
        - Predictive maintenance planning
        - Asset monitoring & reporting
        - Training & quality assurance
        """)
    
    with col2:
        st.markdown("### 📈 Model Performance")
        
        metrics = {
            'Metric': ['Top-1 Accuracy', 'Top-5 Accuracy', 'Model Size', 'Classes'],
            'Value': ['98.06%', '100%', '10.2 MB', '6']
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        if model and model.model:
            info = model.get_model_info()
            for key, value in info.items():
                st.markdown(f"**{key}:** `{value}`")

# ================== FOOTER ==================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("© 2025 Solar Panel Monitoring System")
with col2:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col3:
    st.caption("Version 1.0")
