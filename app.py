import streamlit as st
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# -----------------------------------
# Import functions from apputil
# -----------------------------------
# You may need to adjust the import depending on folder structure
from apputil import kmeans, kmeans_diamonds, kmeans_timer

st.set_page_config(page_title="K-Means Diamonds App", layout="wide")

st.title("💎 K-Means Clustering on Diamonds Dataset")
st.write("This Streamlit app uses your *apputil.py* functions to run k-means clustering on the diamonds dataset.")

# Sidebar options
st.sidebar.header("⚙️ Settings")

n = st.sidebar.number_input(
    "Number of Rows (n)", min_value=10, max_value=50000, value=1000, step=100
)

k = st.sidebar.number_input(
    "Number of Clusters (k)", min_value=1, max_value=20, value=5, step=1
)

n_iter = st.sidebar.number_input(
    "Timer Iterations (for Exercise 3)", min_value=1, max_value=20, value=5
)

# Load diamonds numeric dataset
st.subheader("📊 Diamonds Data (first 10 rows)")
diamonds = sns.load_dataset("diamonds")
numeric_diamonds = diamonds.select_dtypes(include=[np.number])
st.dataframe(numeric_diamonds.head(10))

# Run clustering
if st.button("Run K-Means Clustering"):
    st.subheader("🔍 Running kmeans_diamonds...")
    centroids, labels = kmeans_diamonds(n, k)

    st.write("### Cluster Centroids:")
    st.dataframe(centroids)

    st.write("### Cluster Labels (first 50 shown):")
    st.write(labels[:50])

# Timer
if st.button("Run Timer Test"):
    st.subheader("⏱️ Running kmeans_timer...")
    avg_time = kmeans_timer(n, k, n_iter)
    st.success(f"Average Runtime over {n_iter} runs: {avg_time:.5f} seconds")
