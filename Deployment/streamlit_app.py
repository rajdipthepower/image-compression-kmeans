import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import io
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.image_utils import prepare, compress_image
from src.metrics import mse, PSNR, compression_ratio, inertia

st.set_page_config(
    page_title="K-Means Image Compression",
    page_icon="🎨",
    layout="wide"
)

# --- THEME & STYLING TOGGLE ---
title_col, mode_col = st.columns([5, 1], vertical_alignment="center")
with title_col:
    st.title("🎨 Image Compression using K-Means Clustering")
with mode_col:
    night_mode = st.toggle("🌙 Night Mode", value=False)

# Dynamic CSS styling for background and clean card containers
if night_mode:
    bg_color = "#0e1117"
    text_color = "#ffffff"
    card_bg = "#1f2937"
    border_color = "#374151"
else:
    bg_color = "#fdfdfd"
    text_color = "#000000"
    card_bg = "#ffffff"
    border_color = "#e5e7eb"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    main p,
    main label,
    main [data-testid="stText"] *,
    main [data-testid="stMetricLabel"],
    main [data-testid="stMetricValue"],
    main [data-testid="stMetricDelta"] {{
        color: {text_color} !important;
    }}
    [data-testid="stDownloadButton"] button {{
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stDownloadButton"] button * {{
        color: {text_color} !important;
    }}
    .custom-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }}
    .app-footer {{
        color: {text_color} !important;
        text-align: right;
        margin-top: 2rem;
        padding: 0.75rem 0;
    }}
    .app-footer span {{
        color: #e11d48 !important;
    }}
    /* Force file uploader text and button text to black */
    [data-testid="stFileUploaderDropzone"] div, 
    [data-testid="stFileUploaderDropzone"] span, 
    [data-testid="stFileUploaderDropzone"] small, 
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {{
        color: #000000 !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("Explore color quantization using custom K-Means clustering implemented from scratch in NumPy.")

# Navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", ["Live Compression App", "Exploratory Experiment Results"])

if app_mode == "Live Compression App":
    st.header("Interactive K-Means Image Compressor")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("1. Input Image")
        default_path = ROOT_DIR / "data" / "input" / "bird.png"
        
        run_default = st.checkbox("Use default bird.png", value=True)
        
        original_pil = None
        img_path = None
        uploaded_file = None
        original_file_size_kb = None
        
        if run_default:
            if default_path.exists():
                original_pil = Image.open(default_path).convert('RGB')
                img_path = default_path
                original_file_size_kb = default_path.stat().st_size / 1024
                st.image(original_pil, caption="Default Image (bird.png)", width='stretch')
            else:
                st.warning(f"Default image not found at: {default_path}")
        else:
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                original_pil = Image.open(uploaded_file).convert('RGB')
                original_file_size_kb = uploaded_file.size / 1024
                st.image(original_pil, caption="Uploaded Image", width='stretch')
            else:
                st.info("Please upload an image file to proceed.")

    with col2:
        st.subheader("2. Compression Settings")
        k_clusters = st.slider("Number of Colors / Clusters (K)", min_value=2, max_value=64, value=8, step=2)
        max_iter = st.slider("Max Iterations", min_value=5, max_value=50, value=20)
        tolerance = st.select_slider("Tolerance", options=[1e-4, 1e-5, 1e-6], value=1e-5)
        seed_val = st.number_input("Random Seed", value=42, step=1)
        
        run_btn = st.button("🚀 Run K-Means Compression", type="primary", use_container_width=True)

    if run_btn:
        if original_pil is not None and (img_path is not None or uploaded_file is not None):
            with st.spinner("Crunching pixels and running K-Means clustering..."):
                temp_path = None
                try:
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(
                            suffix=Path(uploaded_file.name).suffix, delete=False
                        ) as temp_file:
                            temp_path = Path(temp_file.name)
                            temp_file.write(uploaded_file.getvalue())
                        processing_path = temp_path
                    else:
                        processing_path = img_path

                    assignments, centroids, reconstructed = compress_image(
                        processing_path, k_clusters, max_iter, tolerance, seed=int(seed_val)
                    )

                    old_uint8 = np.array(original_pil.convert("RGB"))
                    comp_ratio = compression_ratio(reconstructed, old_uint8, centroids)
                    psnr_val = PSNR(reconstructed, old_uint8)
                    mse_val = mse(reconstructed, old_uint8)

                    reconstructed_pil = Image.fromarray(
                        np.round(reconstructed * 255.0).astype(np.uint8)
                    )

                    palette = np.clip(np.round(centroids * 255), 0, 255).astype(np.uint8)
                    height, width = reconstructed.shape[:2]
                    idx_img = Image.fromarray(
                        assignments.reshape(height, width).astype(np.uint8), mode="P"
                    )
                    idx_img.putpalette(palette.flatten().tolist())

                    buf = io.BytesIO()
                    idx_img.save(buf, format="PNG", optimize=True)
                    compressed_bytes = buf.getvalue()
                    compressed_size_kb = len(compressed_bytes) / 1024
                    savings_pct = (
                        (original_file_size_kb - compressed_size_kb)
                        / original_file_size_kb
                        * 100
                    )
                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()
                
            st.success("Compression Complete!")
            
            st.markdown("---")
            st.subheader("3. Results & File Size Comparison")
            st.caption("Original file vs K-color palette PNG")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.image(original_pil, caption="Original Image", width='stretch')
                if original_file_size_kb > 1024:
                    st.text(f"Original Size: {original_file_size_kb / 1024:.2f} MB")
                else:
                    st.text(f"Original Size: {original_file_size_kb:.2f} KB")
                    
            with res_col2:
                st.image(reconstructed_pil, caption=f"Compressed Image (K={k_clusters})", width='stretch')
                if compressed_size_kb > 1024:
                    st.text(f"Compressed Size: {compressed_size_kb / 1024:.2f} MB")
                else:
                    st.text(f"Compressed Size: {compressed_size_kb:.2f} KB")
            
            # Simple metrics row for naive users
            m1, m2, m3, m4, m5 = st.columns(5)
            size_change_label = (
                f"{abs(savings_pct):.1f}% larger"
                if savings_pct < 0
                else f"{savings_pct:.1f}% smaller"
            )
            m1.metric("File Size Reduction", size_change_label)
            m2.metric("PSNR", f"{psnr_val:.2f} dB")
            m3.metric("MSE", f"{mse_val:.5f}")
            m4.metric("Theoretical Ratio", f"{comp_ratio:.2f}x")
            m4.caption("index + palette bits vs raw 24-bit RGB")
            m5.metric("Color Palette", f"{k_clusters} Colors")
            
            # Download Button
            st.markdown("### 📥 Download Your Compressed Image")
            st.download_button(
                label="Download Compressed Image (PNG)",
                data=compressed_bytes,
                file_name=f"compressed_image_K{k_clusters}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.warning("Please make sure an image is selected or uploaded first.")

elif app_mode == "Exploratory Experiment Results":
    st.header("📊 Experiment Analysis Dashboard")
    st.markdown("Review the pre-computed benchmark metrics across different cluster sizes ($K$).")
    
    csv_path = ROOT_DIR / "data" / "output" / "experiment_results.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Performance Plots")
        plot_files = [
            ("Compression Ratio vs Clusters.png", "Compression Ratio"),
            ("Inertia vs Clusters.png", "Inertia"),
            ("Mean Squared Error vs Clusters.png", "Mean Squared Error"),
            ("Memory Usage vs Number of Clusters.png", "Memory Usage"),
            ("PSNR vs Clusters.png", "PSNR"),
            ("Time taken vs Clusters.png", "Time Taken")
        ]
        
        plot_col1, plot_col2 = st.columns(2)
        for idx, (filename, title) in enumerate(plot_files):
            plot_path = ROOT_DIR / "data" / "output" / "plots" / filename
            if plot_path.exists():
                target_col = plot_col1 if idx % 2 == 0 else plot_col2
                with target_col:
                    st.image(str(plot_path), caption=title, width='stretch')
            else:
                st.warning(f"Plot missing: {filename}")
    else:
        st.info(f"No experiment results CSV found at `{csv_path}`. Run `main.py` locally to generate them.")

st.markdown(
    f'<div class="app-footer">Made by Rajdip <span>❤️❤️</span></div>',
    unsafe_allow_html=True,
)