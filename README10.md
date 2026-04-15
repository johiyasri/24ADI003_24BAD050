SCENARIO 1 – SVD-Based Recommendation
🔹 Objective

To use SVD to predict unseen movie ratings and generate recommendations.

Dataset Link:
https://www.kaggle.com/datasets/abhikjha/movielens-100k

🔹 Steps Performed
Imported libraries
Loaded dataset
Preprocessed data
Created user-item matrix
Applied normalization
Performed SVD
Selected latent factors (k)
Reconstructed matrix
Predicted missing ratings
Generated recommendations

🔹 Evaluation Metrics
RMSE
MAE

🔹 Analysis
Increasing k reduces RMSE
Small k → underfitting
Large k → overfitting
Improves prediction accuracy

🔹 Visualizations
Heatmap (original vs reconstructed)
RMSE vs latent factors
Top recommended movies

🔹 Outcome (SVD)
Successfully predicted missing ratings
Achieved low RMSE and MAE values
Generated accurate Top-N movie recommendations
Demonstrated that increasing latent factors improves performance
Showed effectiveness of dimensionality reduction in recommendation systems

SCENARIO 2 – NMF-Based Recommendation
🔹 Objective
To use NMF for generating interpretable recommendations.

Dataset Link:
https://www.kaggle.com/datasets/abhikjha/movielens-100k

🔹 Steps Performed
Loaded dataset
Created user-item matrix
Handled missing values
Applied NMF
Factorized matrices
Reconstructed ratings
Predicted values
Generated recommendations

🔹 Evaluation Metrics
RMSE
Precision@K
Recall@K

🔹 Analysis
Produces meaningful latent features
Handles sparsity well
Slightly less accurate than SVD
Easier to interpret

🔹 Visualizations
Latent feature graph
Reconstruction heatmap
Recommendation chart

🔹 Outcome (NMF)
Successfully generated movie recommendations
Achieved reasonable RMSE values
Provided interpretable latent features
Demonstrated effective handling of sparse data
Generated Top-N recommendations with good Precision@K and Recall@K
