SCENARIO 1: Customer Segmentation using K-Means

Dataset: Mall Customer Segmentation Dataset (Kaggle – Public)
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Objective:
To group customers into distinct segments based on their income, spending behavior, and age using K-Means clustering.

Tasks Performed:

Imported required Python libraries.
Loaded the dataset into a Pandas DataFrame.
Inspected the dataset using head(), info(), and describe().
Checked and handled missing values.
Performed feature scaling for better clustering performance.
Selected relevant features such as Annual Income, Spending Score, and Age.
Used the Elbow Method to determine the optimal number of clusters (K).
Applied K-Means clustering algorithm.
Assigned cluster labels to each customer.
Visualized clusters using scatter plots and centroids.

Visualized:
Elbow Curve (K vs Inertia).
Customer clusters using scatter plots.
Cluster centroids.

Outcome:

This analysis helped in identifying different customer segments such as high income–high spending and low income–low spending groups. It also demonstrated how K-Means effectively forms well-separated clusters and provides insights for targeted marketing strategies.

SCENARIO 2: Customer Segmentation using GMM

Dataset: Mall Customer Segmentation Dataset (Kaggle – Public)
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Objective:
To cluster customers using Gaussian Mixture Models (GMM) and analyze probabilistic cluster membership and overlapping clusters.

Tasks Performed:

Loaded the dataset into Pandas.
Performed preprocessing and feature scaling.
Selected relevant features such as Annual Income, Spending Score, and Age.
Applied Gaussian Mixture Model (GMM).
Selected the number of components (clusters).
Fit the model using the Expectation-Maximization (EM) algorithm.
Predicted cluster probabilities for each customer.
Assigned clusters based on highest probability.
Compared clustering results with K-Means.

Visualized:
Cluster probability distributions.
GMM cluster plots and contour visualizations.
Comparison between K-Means and GMM clustering results.

Outcome:

The analysis showed that GMM provides more flexible clustering by allowing overlapping clusters and probabilistic membership. It revealed deeper insights into customer behavior compared to K-Means, especially when clusters are not clearly separable.
