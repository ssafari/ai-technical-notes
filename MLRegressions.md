# Linear Regression

The algorithm finds the line of best fit through your data points by minimizing the distance between the actual values and the predicted line.

### 1. Mathematical Equation

$(\hat{y}=\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+\dots +\beta_{n}x_{n})$

- $(\hat{y})$  is the predicted continuous output.

- $(\beta _{0})$ is the intercept (y-axis crossing point).

- $(\beta_1, \beta_2 \dots \beta_n)$ are the feature weights (coefficients).

- $(x_1, x_2 \dots x_n)$ are the input features.

### 2. Loss Function (What it minimizes)

Ordinary Least Squares (OLS) or Mean Squared Error (MSE). It squares the errors (residuals) to penalize larger mistakes heavily:

$J(\beta )=\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-\hat{y}^{(i)})^{2}$

### 3. Optimization Algorithm (How it learns)

- Normal Equation (Closed-form solution): Calculates the exact optimal weights mathematically in one step using matrix inversion: $\beta = (X^T_X)^{-1}X^T_y$.
- Gradient Descent: Iteratively tweaks the weights (\(\beta \)) to slide down the error curve until finding the lowest possible MSE.

# Logistic Regression

The algorithm calculates a probability score using a linear relationship, then bends that score into a curve bound strictly between 0 and 1.

### 1. Mathematical Equation

First, it calculates a linear combination (\(z\)), just like linear regression:

$(z=\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+\dots +\beta_{n}x_{n})$

Then, it maps $(z)$ into the Sigmoid (Logistic) Function to calculate the final probability $(p)$:

$p=\sigma (z)=\frac{1}{1+e^{-z}}$

### 2. Loss Function (What it minimizes)

Binary Cross-Entropy Loss (also called Log Loss). MSE cannot be used here because the Sigmoid curve creates a wavy, non-convex shape with many local minimums. Log Loss provides a smooth, bowl-shaped (convex) surface:

$(J(\beta )=-\frac{1}{m}\sum _{i=1}^{m}\left[y^{(i)}\log (p^{(i)})+(1-y^{(i)})\log (1-p^{(i)})\right])$

- If the true label $y = 1$ and the model predicts a probability $p$ close to 0, the penalty approaches infinity.

### 3. Optimization Algorithm (How it learns)

Unlike linear regression, logistic regression has no closed-form algebraic solution (you can't solve it in a single step with a formula). It must rely on iterative optimization algorithms to minimize the Log Loss:
- Gradient Descent (or variants like Stochastic Gradient Descent/Adam).
- Advanced Solvers: Optimization algorithms like L-BFGS, Newton-Raphson, or Liblinear are commonly used in libraries like scikit-learn to converge on optimal weights quickly.

## Comparison of the Mechanics

|Attribute|	Linear Regression Algorithm|	Logistic Regression Algorithm|
|---------|----------------------------|---------------------------------|
|Output Type|	Continuous real number $(\mathbb{R})$	|Probability score bounded between $[0, 1]$|
|Geometric Shape|	Straight line, plane, or hyperplane	|`S-shaped curve (Sigmoid curve)`|
|Solvable in 1 Step?|	Yes (via Normal Equation matrix math)	|No (requires iterative optimization)|
|Standard Solver|	`Ordinary Least Squares (OLS)`|	`Gradient Descent`, `L-BFGS`, or `Liblinear`|

