# Classification Metrics

`Accuracy`, `precision`, `recall`, and the `F1 score` are standard evaluation metrics used to measure the performance of classification machine learning models. They rely on the components of a confusion matrix: 
- TP (True Positive): Correctly predicted positive cases.
- TN (True Negative): Correctly predicted negative cases.
- FP (False Positive): Incorrectly predicted positive cases (false alarm).
- FN (False Negative): Missed positive cases. 

## 1. Accuracy

- What it measures: The overall proportion of correct predictions (both positive and negative) out of all predictions made.
- Formula:

>>>>>>>>>>>> $\text{Accuracy}=\frac{\text{TP}+\text{TN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}$

- When to use: Best for balanced datasets where every class has a similar number of samples.
- Limitation: Can be dangerously misleading on imbalanced datasets (e.g., a fraud detection dataset with 99% non-fraud cases where a model predicts "not fraud" for everything and gets 99% accuracy while catching zero fraud). 

## 2. Precision (Positive Predictive Value)

- What it measures: Out of all the instances the model predicted as positive, how many were actually positive?
- Formula:

>>>>>>>>>>>> $\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}$

- When to prioritize: When false positives are costly. For example, in spam email filtering, you want high precision so a legitimate email is rarely flagged as spam (avoiding false positives). 

## 3. Recall (Sensitivity / True Positive Rate)

- What it measures: Out of all the actual positive instances in the dataset, how many did the model manage to find?
- Formula:

>>>>>>>>>>>> $\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}$

- When to prioritize: When false negatives are costly or dangerous. For example, in medical disease screening, you want high recall so you rarely miss an actual sick patient (avoiding false negatives). 

## 4. F1 Score

- What it measures: The harmonic mean of precision and recall, combining them into a single balanced metric.
- Formula:

>>>>>>>>>>>> $\text{F1\ Score}=2\times \frac{\text{Precision}\times \text{Recall}}{\text{Precision}+\text{Recall}}$

- Range: Between 0 (worst) and 1 (perfect).
- When to use: When you need a single score that balances both precision and recall, especially when dealing with skewed or imbalanced class distributions where accuracy fails.

### Summary Comparison


|Metric|Focus|Key Question Answered|Primary Weakness|
|------|-----|---------------------|----------------|
|Accuracy|Overall correctness|How many total predictions were right?|Fails with class imbalance|
|Precision|	Quality of positive picks|	Of predicted positives, how many are real?|	Ignores missed actual positives (FN)|
|Recall|	Quantity of actual positives found|	Of actual positives, how many did we catch?|	Ignores false alarms (FP)|
|F1 Score|	Harmonic balance|	How well do precision and recall trade off?|	Can mask individual flaws if averaged evenly|

