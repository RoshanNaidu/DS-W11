# Exercise 1

import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, k):
    """
    Perform k-means clustering on a numerical NumPy array X.

    Parameters:
        X (np.ndarray): Data array of shape (n_samples, n_features)
        k (int): Number of clusters

    Returns:
        (centroids, labels)
            centroids: (k, n_features)
            labels:    (n_samples,)
    """
    
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X)
    
    centroids = model.cluster_centers_
    labels = model.labels_
    
    return centroids, labels


# --- Given in the exercise ---
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

centroids, labels = kmeans(X, k=3)


# Exercise 2

import numpy as np
import seaborn as sns

# ---------------------------------------------------------
# 1. Load the diamonds dataset
# ---------------------------------------------------------
diamonds = sns.load_dataset("diamonds")

# ---------------------------------------------------------
# 2. Keep only the 7 numerical columns
# ---------------------------------------------------------
numeric_cols = diamonds.select_dtypes(include=[np.number])
numeric_diamonds = numeric_cols.copy()   # global variable
# This contains: carat, depth, table, price, x, y, z

# ---------------------------------------------------------
# 3. Function kmeans_diamonds(n, k)
# ---------------------------------------------------------
def kmeans_diamonds(n, k):
    """
    Runs k-means clustering on the first n rows of the numeric
    diamonds dataset using the kmeans() function from Exercise 1.
    
    Parameters:
        n (int): number of rows to use
        k (int): number of clusters
        
    Returns:
        (centroids, labels)
    """
    # Get first n samples
    X = numeric_diamonds.iloc[:n].to_numpy()
    
    # Use the kmeans() function from Exercise 1
    centroids, labels = kmeans(X, k)
    
    return centroids, labels


# Exercise 3

from time import time

def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds(n, k) n_iter times and returns
    the average runtime in seconds.

    Parameters:
        n (int): number of rows to use in diamonds dataset
        k (int): number of clusters
        n_iter (int): number of repetitions

    Returns:
        float: average runtime in seconds
    """
    
    runtimes = []

    for _ in range(n_iter):
        start = time()                 # start timer
        _ = kmeans_diamonds(n, k)      # run clustering
        elapsed = time() - start       # time for this run
        runtimes.append(elapsed)

    # Return the average runtime
    return sum(runtimes) / n_iter

