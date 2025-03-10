# **Clustering Algorithms Project**

## **Overview**
This project explores clustering techniques to uncover patterns in a given dataset. We apply three different clustering algorithms:

1. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
2. **Gaussian Mixture Model (GMM)**
3. **K-Means Clustering**

Each method has its strengths and weaknesses, and we compare their performance based on cluster quality, efficiency, and handling of outliers.

---

## **1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

### **How DBSCAN Works**
DBSCAN is a density-based clustering algorithm that identifies clusters based on the density of data points. It classifies points as:
- **Core Points**: Points with at least `min_samples` neighbors within a given radius `eps`.
- **Border Points**: Points within `eps` of a core point but with fewer than `min_samples` neighbors.
- **Noise Points**: Points that do not belong to any cluster.

### **Advantages**
No need to specify the number of clusters in advance.  
Can identify clusters of arbitrary shapes.  
Effectively detects outliers as noise points.  

### **Challenges**
Sensitive to parameter selection (`eps` and `min_samples`).  
Struggles with clusters of varying densities.  

### **Implementation**
We determine `eps` using the k-distance graph and set `min_samples = dimensions + 1`. The DBSCAN algorithm is applied as follows:
```python
from sklearn.cluster import DBSCAN

# Apply DBSCAN with optimal parameters
dbscan = DBSCAN(eps=0.25, min_samples=5)
clusters = dbscan.fit_predict(data)
```

---

## **2. Gaussian Mixture Model (GMM)**

### **How GMM Works**
Gaussian Mixture Models assume that data points are generated from a mixture of several Gaussian distributions. It uses the **Expectation-Maximization (EM) algorithm** to estimate:
- The probability that each point belongs to a cluster.
- The mean and covariance of each Gaussian component.

### **Advantages**
Soft clustering allows uncertainty (points belong to multiple clusters with probabilities).  
Works well when clusters have elliptical shapes.  
Can model complex distributions.

### **Challenges**
Requires specifying the number of clusters (`k`).  
Computationally expensive compared to K-Means.  

### **Implementation**
We use the Bayesian Information Criterion (BIC) to select the optimal number of Gaussian components:
```python
from sklearn.mixture import GaussianMixture

# Fit GMM with 3 components
gmm = GaussianMixture(n_components=3, covariance_type='full')
clusters = gmm.fit_predict(data)
```

---

## **3. K-Means Clustering**

### **How K-Means Works**
K-Means is a centroid-based clustering algorithm that partitions data into `k` clusters by minimizing the variance within each cluster.

**Steps:**
1. Select `k` cluster centers randomly.
2. Assign each point to the nearest cluster center.
3. Compute new cluster centers by averaging assigned points.
4. Repeat until convergence.

### **Advantages**
Fast and scalable.  
Works well for spherical clusters.  
Easy to implement.

### **Challenges**
Requires specifying `k` in advance.  
Sensitive to outliers and cluster shape.  
Can converge to local optima.

### **Implementation**
We use the **Elbow Method** to determine the optimal `k`:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)
```

---

## **Comparison & Conclusion**
| Algorithm | Handles Outliers | Detects Arbitrary Shapes | Requires k | Soft Clustering |

- **If the dataset has noise and complex shapes, DBSCAN is the best choice.**
- **If the clusters are elliptical and `k` is known, GMM performs well.**
- **If speed is a priority and clusters are spherical, K-Means is ideal.**

---

## **Getting Started**
### **Installation**
Install the required dependencies using:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### **Run the Code**
```bash
python clustering_project.py
```

---

## **Future Improvements**
- Experiment with different distance metrics for DBSCAN.
- Use silhouette scores to evaluate clustering quality.
- Implement hierarchical clustering for comparison.

---

## **Author**
Developed by **Neela Ropp**, a data scientist passionate about machine learning and clustering analysis.

