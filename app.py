"""
FreshScanAI - Ultra Premium Apple-Inspired UI
Designed for trust, elegance, and emotional engagement
"""

import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import time
import base64

from predict import FreshScanAIPredictor
from config import CLASS_CONFIG

# Page config
st.set_page_config(
    page_title="FreshScanAI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra Premium CSS - Apple Inspired
st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header, .stDeployButton {display: none !important;}
    
    /* Main container */
    .main {
        background: linear-gradient(180deg, #FAFAFA 0%, #FFFFFF 100%);
        padding: 0;
    }
    
    .block-container {
        max-width: 1200px;
        padding: 4rem 2rem;
        margin: 0 auto;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 6rem 0 4rem 0;
        animation: fadeInUp 1.2s ease-out;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 300;
        letter-spacing: -0.04em;
        color: #1D1D1F;
        margin: 0;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 300;
        color: #86868B;
        margin-top: 1rem;
        line-height: 1.5;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        padding: 3rem;
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Upload Section */
    .upload-zone {
        background: rgba(255, 255, 255, 0.5);
        border: 2px dashed rgba(0, 0, 0, 0.1);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .upload-zone:hover {
        background: rgba(255, 255, 255, 0.8);
        border-color: rgba(52, 199, 89, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
    }
    
    .upload-text {
        font-size: 1.25rem;
        font-weight: 400;
        color: #1D1D1F;
        margin: 0;
    }
    
    .upload-hint {
        font-size: 0.95rem;
        color: #86868B;
        margin-top: 0.5rem;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        backdrop-filter: blur(30px);
        border-radius: 28px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.12);
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: 1px solid rgba(255, 255, 255, 0.9);
    }
    
    .result-status {
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #86868B;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 3.5rem;
        font-weight: 300;
        letter-spacing: -0.03em;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .result-fresh {
        color: #34C759;
    }
    
    .result-spoiled {
        color: #FF9500;
    }
    
    .result-rotten {
        color: #FF3B30;
    }
    
    .result-message {
        font-size: 1.125rem;
        font-weight: 400;
        color: #1D1D1F;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    /* Info Panel */
    .info-panel {
        background: rgba(248, 248, 248, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .info-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #86868B;
        margin-bottom: 0.75rem;
    }
    
    .info-content {
        font-size: 1rem;
        font-weight: 400;
        color: #1D1D1F;
        line-height: 1.7;
    }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: #1D1D1F;
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.0625rem;
        font-weight: 500;
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.01em;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:hover {
        background: #2D2D2F;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Loading State */
    .loading-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .loading-text {
        font-size: 1.25rem;
        font-weight: 300;
        color: #1D1D1F;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Divider */
    .elegant-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent);
        margin: 4rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Confidence Ring */
    .confidence-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .stat-item:hover {
        transform: translateY(-4px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1D1D1F;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Disclaimer */
    .disclaimer {
        text-align: center;
        font-size: 0.8125rem;
        color: #86868B;
        padding: 2rem;
        margin-top: 4rem;
    }
    
    /* Image preview */
    img {
        border-radius: 20px;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide file uploader label */
    .stFileUploader label {
        display: none;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'loading_stage' not in st.session_state:
    st.session_state.loading_stage = 0

@st.cache_resource
def load_model():
    """Load model once"""
    try:
        return FreshScanAIPredictor()
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None

def create_animated_confidence_ring(confidence, class_name):
    """Create elegant animated confidence ring"""
    color_map = {
        'Fresh': '#34C759',
        'Slightly_Spoiled': '#FF9500',
        'Rotten': '#FF3B30'
    }
    color = color_map.get(class_name, '#34C759')
    
    fig = go.Figure()
    
    # Outer ring (background)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={
            'suffix': "%",
            'font': {'size': 56, 'color': '#1D1D1F', 'family': 'Inter'},
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'tickcolor': "white",
                'visible': False
            },
            'bar': {'color': color, 'thickness': 0.15},
            'bgcolor': "rgba(0,0,0,0.03)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': 'rgba(0,0,0,0.03)'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, -apple-system, sans-serif'}
    )
    
    return fig

def create_elegant_probability_bars(probabilities):
    """Create soft, elegant probability visualization"""
    colors = {
        'Fresh': '#34C759',
        'Slightly_Spoiled': '#FF9500',
        'Rotten': '#FF3B30'
    }
    
    classes = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in classes]
    bar_colors = [colors.get(k, '#86868B') for k in classes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[c.replace('_', ' ') for c in classes],
        x=values,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(width=0)
        ),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=14, color='#1D1D1F', family='Inter'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=80, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, 105],
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=14, color='#1D1D1F', family='Inter'),
            fixedrange=True
        ),
        showlegend=False,
        font={'family': 'Inter, -apple-system, sans-serif'}
    )
    
    return fig

def show_hero_section():
    """Hero section - Apple style"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">FreshScanAI</h1>
        <p class="hero-subtitle">AI-powered food quality analysis for health and safety</p>
    </div>
    """, unsafe_allow_html=True)

def show_loading_animation():
    """Cinematic loading states"""
    loading_messages = [
        "Analyzing freshness…",
        "Evaluating spoilage patterns…",
        "Finalizing safety assessment…"
    ]
    
    placeholder = st.empty()
    
    for msg in loading_messages:
        with placeholder.container():
            st.markdown(f"""
            <div class="loading-container">
                <p class="loading-text">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
        time.sleep(0.8)
    
    placeholder.empty()

def get_result_message(class_name, confidence):
    """Calm, human messages"""
    messages = {
        'Fresh': "This food appears fresh and safe to consume.",
        'Slightly_Spoiled': "This food may no longer be safe to eat. Inspect carefully before consuming.",
        'Rotten': "This food is not safe to consume. Please discard it."
    }
    return messages.get(class_name, "Analysis complete.")

def show_result_section(result):
    """Ceremonial result presentation"""
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Result card with animation
    color_class = {
        'Fresh': 'result-fresh',
        'Slightly_Spoiled': 'result-spoiled',
        'Rotten': 'result-rotten'
    }.get(predicted_class, 'result-fresh')
    
    st.markdown(f"""
    <div class="result-card">
        <div class="result-status">Assessment Complete</div>
        <div class="result-value {color_class}">{predicted_class.replace('_', ' ')}</div>
        <div class="result-message">{get_result_message(predicted_class, confidence)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confidence ring
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(
            create_animated_confidence_ring(confidence, predicted_class),
            use_column_width=True,
            config={'displayModeBar': False, 'staticPlot': True}
        )
    
    st.markdown("<div class='elegant-divider'></div>", unsafe_allow_html=True)
    
    # Probability breakdown
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <p style="font-size: 0.875rem; font-weight: 500; color: #86868B; text-transform: uppercase; letter-spacing: 0.1em;">
            Classification Breakdown
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(
        create_elegant_probability_bars(result['all_probabilities']),
        use_column_width=True,
        config={'displayModeBar': False, 'staticPlot': True}
    )
    
    # Safety information
    st.markdown("<div class='elegant-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_levels = {
            'Fresh': 'Low',
            'Slightly_Spoiled': 'Medium',
            'Rotten': 'High'
        }
        risk = risk_levels.get(predicted_class, 'Low')
        
        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">Risk Level</div>
            <div class="info-content">{risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        shelf_life = {
            'Fresh': '3-5 days with proper storage',
            'Slightly_Spoiled': 'Use within 24 hours if consumed',
            'Rotten': 'Discard immediately'
        }
        shelf = shelf_life.get(predicted_class, '3-5 days')
        
        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">Recommendation</div>
            <div class="info-content">{shelf}</div>
        </div>
        """, unsafe_allow_html=True)

def show_analytics_section():
    """Elegant analytics"""
    history = st.session_state.prediction_history
    
    if len(history) == 0:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <p style="font-size: 1.125rem; color: #86868B;">No predictions yet</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <p style="font-size: 0.875rem; font-weight: 500; color: #86868B; text-transform: uppercase; letter-spacing: 0.1em;">
            Usage Statistics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    fresh_count = sum(1 for p in history if p['predicted_class'] == 'Fresh')
    spoiled_count = sum(1 for p in history if p['predicted_class'] == 'Slightly_Spoiled')
    rotten_count = sum(1 for p in history if p['predicted_class'] == 'Rotten')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Total</div>
            <div class="stat-number">{len(history)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Fresh</div>
            <div class="stat-number" style="color: #34C759;">{fresh_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Spoiled</div>
            <div class="stat-number" style="color: #FF9500;">{spoiled_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Rotten</div>
            <div class="stat-number" style="color: #FF3B30;">{rotten_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("Clear History", key="clear_btn"):
        st.session_state.prediction_history = []
        if 'current_result' in st.session_state:
            del st.session_state['current_result']
        st.experimental_rerun()

def main():
    """Main application"""
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.error("Failed to load model. Please check installation.")
        st.stop()
    
    # Hero
    show_hero_section()
    
    # Navigation
    tab1, tab2 = st.tabs(["  Analyze  ", "  Statistics  "])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            uploaded_file = st.file_uploader(
                "upload",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is None:
                st.markdown("""
                <div class="upload-zone">
                    <p class="upload-text">Upload an image</p>
                    <p class="upload-hint">JPG or PNG • Best results with well-lit, clear photos</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show image
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Analyze button
                if st.button("Analyze", key="analyze_btn"):
                    # Cinematic loading
                    show_loading_animation()
                    
                    # Predict
                    temp_path = Path("temp_upload.jpg")
                    image.save(temp_path)
                    result = predictor.predict(temp_path)
                    temp_path.unlink(missing_ok=True)
                    
                    if result:
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.prediction_history.append(result)
                        st.session_state['current_result'] = result
                        st.experimental_rerun()
        
        # Show result if exists
        if 'current_result' in st.session_state:
            st.markdown("<div class='elegant-divider'></div>", unsafe_allow_html=True)
            show_result_section(st.session_state['current_result'])
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_analytics_section()
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        This tool assists food safety awareness and is not a substitute for professional inspection or medical advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
