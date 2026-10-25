# Multicollinearity 

In machine learning occurs when two or more independent features in a dataset are highly correlated with each other.
This means that one feature can be accurately predicted from the others, making them redundant. For example, if you have features for Birth Year and Age, or Loan Amount and Property Value, they move so closely together that they essentially provide the same information to the model.

## Why It Matters (By Model Type)

The impact of multicollinearity depends entirely on the type of machine learning algorithm you are using:

### 1. Linear & Logistic Regression (Severe Impact)

In linear models, multicollinearity is a major problem because it ruins feature interpretability and model stability.
- Unstable Coefficients: The model struggle to determine which feature is actually driving the target variable. As a result, coefficient weights can swing wildly, flip signs (e.g., showing a positive relationship when it should be negative), or become highly sensitive to tiny changes in training data.
- Inflated P-values: It inflates the standard errors of the coefficients, making statistically important variables appear insignificant.

### 2. Tree-Based Models (XGBoost, Random Forest) (Mild Impact)

Tree-based models do not suffer from mathematical instability due to multicollinearity, but it degrades your ability to understand the model:
- Diluted Feature Importance: If Feature A and Feature B are identical, a tree might split on Feature A in one tree and Feature B in another. Their individual "feature importance" scores will be cut in half, hiding their true impact.
- Unreliable SHAP/LIME Values: Explainable AI tools will struggle to distribute credit properly between the correlated features.

### 3. Distance-Based Models (KNN, SVM, Clustering) (Moderate Impact)

- Implicit Weighting: If you have three highly correlated features representing the same underlying concept, that concept gets "triple-counted" when calculating distances between data points, distorting the model's geometric logic.

## How to Detect It

1. Correlation Matrix: A simple heatmap checking for high Pearson or Spearman correlation coefficients (usually |ρ| > 0.7 or 0.8).
2. Variance Inflation Factor (VIF): A metric that runs a mini-regression of one feature against all other features.
	- VIF = 1: No correlation.
	- VIF > 5 to 10: High multicollinearity; the feature should likely be removed or transformed.

## How to Fix It

- Drop Redundant Features: Keep the feature with the highest correlation to the target variable and drop the others.
- Feature Transformation: Combine them into ratios (like the Loan-to-Income ratio) or differences.
- Dimensionality Reduction: Use techniques like Principal Component Analysis (PCA) to project correlated features into orthogonal (uncorrelated) components.
- Regularization (L1/L2): Use Ridge (L2) regression to shrink coefficients of correlated features together, or Lasso (L1) regression to automatically select one and zero out the others.
Are you currently experiencing an issue where feature importance scores seem artificially low, or are you getting unstable coefficients in a linear/logistic regression model? Let me know which algorithm you are working with so we can pinpoint the right diagnostic step.
