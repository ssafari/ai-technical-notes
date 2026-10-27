# Linear regression evaluation metrics 

They quantify prediction errors and measure how well a line fits continuous data. For applying these metrics by using tools from `scikit-learn 1.9.0`. 

## Key Evaluation Metrics

- **Mean Absolute Error (MAE)**: The average of the absolute differences between actual and predicted values. It treats all errors equally and is robust to outliers. 
- **Mean Squared Error (MSE)**: The average of the squared differences between actual and predicted values. It heavily penalizes large errors because the differences are squared. 
- **Root Mean Squared Error (RMSE)**: The square root of the MSE. It brings the error scale back to the original units of the target variable while keeping a heavy penalty for large errors. 
- **R-squared (R² Score)**: The proportion of variance in the dependent variable explained by the independent variables, ranging from 0 to 1 (though it can be negative if the model performs worse than a horizontal mean line). 
- **Adjusted R-squared**: A modified version of `R²` that penalizes the addition of useless predictor variables, making it more reliable for multiple linear regression. 

## Quick Comparison Table


|Metric|	Sensitivity to Outliers|	Unit Scale|	Best Used For|
|------|---------------------------|--------------|--------------|
|**MAE**|	Low (treats all errors evenly)|	Same as target variable|	General interpretability with outliers|
|**MSE**|	High (squares large errors)|	Squared target variable|	Optimization models (differentiable)|
|**RMSE**|	High (squares large errors)|	Same as target variable|	Measuring average distance in original units|
|**R²**|	Moderate|	Unitless (0 to 1)|	Assessing proportion of explained variance|

