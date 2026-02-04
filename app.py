"""
Streamlit Dashboard for Manufacturing Defect Detection

Features:
1. Single image classification
2. Batch processing
3. Defect localization with Grad-CAM
4. Real-time analytics
"""

import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import io

from config import (
    APP_TITLE, APP_LAYOUT, CLASSES, CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD, TRAIN_DIR, TEST_DIR
)
from models.preprocessing import (
    generate_synthetic_dataset, generate_defect_image, 
    generate_good_image, preprocess_image
)
from models.classifier import DefectDetector, SimpleClassifier
from models.gradcam import DefectLocalizer, create_heatmap_overlay
from evaluation.metrics import evaluate_classifier, DefectEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=APP_TITLE,
    layout=APP_LAYOUT,
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e74c3c 0%, #8e44ad 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .defect-badge {
        background-color: #e74c3c;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .good-badge {
        background-color: #27ae60;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .confidence-high {
        color: #27ae60;
    }
    .confidence-low {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def get_detector():
    """Initialize and cache the defect detector."""
    detector = DefectDetector(model_name="resnet18", pretrained=True)
    return detector


@st.cache_resource
def get_localizer(_detector):
    """Initialize and cache the defect localizer."""
    return DefectLocalizer(_detector)


# Check if data exists
def ensure_data_exists():
    """Generate synthetic data if not present."""
    train_good = TRAIN_DIR / "good"
    if not train_good.exists() or len(list(train_good.glob("*.png"))) == 0:
        with st.spinner("Generating synthetic dataset (first run only)..."):
            generate_synthetic_dataset()
        st.success("Dataset generated!")
        return True
    return False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Confidence threshold
    confidence_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence to classify as defect"
    )
    
    # Show heatmap
    show_heatmap = st.checkbox("Show Defect Heatmap", value=True)
    
    st.markdown("---")
    
    # Demo section
    st.markdown("### 🎲 Generate Demo Images")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Good Sample"):
            st.session_state['demo_image'] = generate_good_image()
            st.session_state['demo_type'] = 'good'
    with col2:
        if st.button("Defect Sample"):
            st.session_state['demo_image'] = generate_defect_image()
            st.session_state['demo_type'] = 'defect'
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("- **Model**: ResNet-18 (pretrained)")
    st.markdown("- **Input**: 224x224 RGB")
    st.markdown("- **Classes**: Good, Defect")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<p class="main-header">🔍 Manufacturing Defect Detection</p>', unsafe_allow_html=True)
st.markdown("AI-powered visual inspection for quality control")

# Ensure data exists
ensure_data_exists()

# Initialize detector
detector = get_detector()

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Single Image",
    "📁 Batch Processing", 
    "📊 Analytics",
    "🔬 Model Evaluation"
])


# ============================================================================
# TAB 1: SINGLE IMAGE CLASSIFICATION
# ============================================================================

with tab1:
    st.markdown("### Upload or Generate Image for Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            help="Upload a product image for defect detection"
        )
        
        # Use demo image if available
        if 'demo_image' in st.session_state:
            image = st.session_state['demo_image']
            st.info(f"Using generated {st.session_state.get('demo_type', 'demo')} sample")
        elif uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
        else:
            image = None
        
        if image is not None:
            st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        if image is not None:
            # Run prediction
            with st.spinner("Analyzing..."):
                prediction, confidence = detector.predict(image)
            
            # Display result
            st.markdown("### Result")
            
            if prediction == "defect":
                st.markdown(f'<span class="defect-badge">⚠️ DEFECT DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="good-badge">✅ GOOD</span>', unsafe_allow_html=True)
            
            # Confidence
            conf_class = "confidence-high" if confidence > HIGH_CONFIDENCE_THRESHOLD else "confidence-low"
            st.markdown(f'**Confidence:** <span class="{conf_class}">{confidence:.1%}</span>', unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Heatmap
            if show_heatmap and prediction == "defect":
                st.markdown("### Defect Localization")
                try:
                    localizer = get_localizer(detector)
                    heatmap, overlay = localizer.localize(image)
                    st.image(overlay, caption="Defect Heatmap (Grad-CAM)", use_container_width=True)
                except Exception as e:
                    st.warning(f"Heatmap generation failed: {e}")
        else:
            st.info("👈 Upload an image or generate a demo sample from the sidebar")


# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================

with tab2:
    st.markdown("### Batch Defect Analysis")
    
    # Use test directory
    test_images = []
    for label in ["good", "defect"]:
        label_dir = TEST_DIR / label
        if label_dir.exists():
            for img_path in label_dir.glob("*.png"):
                test_images.append((str(img_path), label))
    
    if not test_images:
        st.warning("No test images found. Generate dataset first.")
    else:
        st.info(f"Found {len(test_images)} test images")
        
        if st.button("Run Batch Analysis", type="primary"):
            results = []
            progress = st.progress(0)
            
            for i, (img_path, true_label) in enumerate(test_images):
                img = Image.open(img_path).convert('RGB')
                pred_label, confidence = detector.predict(img)
                
                results.append({
                    'Image': Path(img_path).name,
                    'True Label': true_label,
                    'Predicted': pred_label,
                    'Confidence': f"{confidence:.1%}",
                    'Correct': '✅' if pred_label == true_label else '❌'
                })
                
                progress.progress((i + 1) / len(test_images))
            
            # Display results
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Summary
            correct = sum(1 for r in results if r['Correct'] == '✅')
            accuracy = correct / len(results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.1%}")
            with col2:
                st.metric("Total Images", len(results))
            with col3:
                st.metric("Errors", len(results) - correct)
            
            st.dataframe(df, use_container_width=True)


# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================

with tab3:
    st.markdown("### Defect Analytics Dashboard")
    
    # Simulated production data
    np.random.seed(42)
    n_items = 1000
    defect_rate = 0.05  # 5% defect rate
    
    # Generate production timeline
    defects = np.random.binomial(1, defect_rate, n_items)
    confidence_scores = np.random.beta(8, 2, n_items) * 0.5 + 0.5  # Mostly high confidence
    
    # Defect type distribution
    defect_types = ['Scratch', 'Crack', 'Contamination', 'Dent']
    defect_counts = [45, 20, 25, 10]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Inspected", n_items)
    with col2:
        st.metric("Defects Found", sum(defects))
    with col3:
        st.metric("Defect Rate", f"{sum(defects)/n_items:.1%}")
    with col4:
        st.metric("Avg Confidence", f"{np.mean(confidence_scores):.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Defect Type Distribution")
        fig = px.pie(
            values=defect_counts,
            names=defect_types,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Confidence Distribution")
        fig = px.histogram(
            x=confidence_scores,
            nbins=20,
            labels={'x': 'Confidence', 'y': 'Count'},
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Defect trend
    st.markdown("#### Defect Rate Over Time (Simulated)")
    hours = list(range(24))
    hourly_rates = [0.03, 0.02, 0.02, 0.04, 0.05, 0.06, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03,
                   0.04, 0.05, 0.06, 0.07, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=hourly_rates,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#e74c3c')
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="gray", annotation_text="Target: 5%")
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Defect Rate",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 4: MODEL EVALUATION
# ============================================================================

with tab4:
    st.markdown("### Model Performance Evaluation")
    
    if st.button("Run Full Evaluation on Test Set"):
        evaluator = DefectEvaluator()
        
        # Load test data
        test_images = []
        for label_idx, label in enumerate(["good", "defect"]):
            label_dir = TEST_DIR / label
            if label_dir.exists():
                for img_path in label_dir.glob("*.png"):
                    test_images.append((str(img_path), label_idx))
        
        if not test_images:
            st.error("No test images found!")
        else:
            progress = st.progress(0)
            
            for i, (img_path, true_label) in enumerate(test_images):
                img = Image.open(img_path).convert('RGB')
                pred_label, _ = detector.predict(img)
                pred_idx = CLASSES.index(pred_label)
                
                evaluator.add_prediction(true_label, pred_idx)
                progress.progress((i + 1) / len(test_images))
            
            # Get results
            metrics = evaluator.evaluate()
            cm = evaluator.get_confusion_matrix()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
            with col2:
                st.metric("Precision", f"{metrics['Precision']:.1%}")
            with col3:
                st.metric("Recall", f"{metrics['Recall']:.1%}")
            with col4:
                st.metric("F1 Score", f"{metrics['F1 Score']:.1%}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=['Good', 'Defect'],
                y=['Good', 'Defect'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("#### Interpretation")
            st.markdown(f"""
            - **True Negatives (TN)**: {cm[0,0]} - Correctly identified good items
            - **False Positives (FP)**: {cm[0,1]} - Good items wrongly flagged as defects
            - **False Negatives (FN)**: {cm[1,0]} - **Missed defects** (critical!)
            - **True Positives (TP)**: {cm[1,1]} - Correctly identified defects
            """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🔍 Manufacturing Defect Detection | Powered by CNN + Grad-CAM<br>
    Solving the $3 trillion manufacturing quality crisis
</div>
""", unsafe_allow_html=True)
