# Loss or Cost funtions

Loss or Cost funtions

Author

Benedict Thekkel

## Loss Functions in Brief

The key takeaway is that the loss function is a measurable way to gauge the performance and accuracy of a machine learning model. In this case, the loss function acts as a guide for the learning process within a model or machine learning algorithm.

The role of the loss function is crucial in the training of machine learning models and includes the following:

- Performance measurement: Loss functions offer a clear metric to evaluate a model’s performance by quantifying the difference between predictions and actual results.
- Direction for improvement: Loss functions guide model improvement by directing the algorithm to adjust parameters(weights) iteratively to reduce loss and improve predictions.
- Balancing bias and variance: Effective loss functions help balance model bias (oversimplification) and variance (overfitting), essential for the model’s generalization to new data.
- Influencing model behavior: Certain loss functions can affect the model’s behavior, such as being more robust against data outliers or prioritizing specific types of errors.

Let’s explore the roles of particular loss functions in later sections and build a detailed intuition and understanding of the loss function.

## Applicability to Classification

### Binary Cross-Entropy Loss / Log Loss

To understand Binary Cross-Entropy Loss, sometimes called Log Loss, it is helpful to discuss the components of the terms. - Loss: This is a mathematical quantification of the margin/difference between the prediction of a machine learning algorithm and the actual target value. - Entropy: A simple definition of entropy is that it is a calculation of the degree of randomness or disorder within a system - Cross Entropy: This is a term commonly utilised in information theory, and it measures the differences between two probability distributions that can be used to identify an observation. - Binary: This is an expression of numerical digits using either of two states, 0 or 1. This is extended to the definition of Binary Classification where we ditingus=ish two classes(A and B) using binary representation, where class A is assigned the numerical representation of 0 and class B is assigned 1.

$L(y , f(x)) = -[y * log(f(x)) + (1 - y) * log(1 - f(x))]$

Where:

    L represents the Binary Cross-Entropy Loss function
    y is the true binary label (0 or 1)
    f(x) is the predicted probability of the positive class (between 0 and 1)

### Categorical Cross-Entropy Loss

### Hinge Loss

> Hinge Loss is a loss function utilized within machine learning to train classifiers that optimize to increase the margin between data points and the decision boundary. Hence, it is mainly used for maximum margin classifications. To ensure the maximum margin between the data points and boundaries, hinge loss penalizes predictions from the machine learning model that are wrongly classified, which are predictions that fall on the wrong side of the margin boundary and also predictions that are correctly classified but are within close proximity to the decision boundary.

$L(y - f(x)) = \max(0, 1 - y * f(x))$ Where:

    L represents the Hinge Loss
    y is the true label or target value (-1 or 1)
    f(x) is the predicted value or decision function output

### Log Loss

## Applicability to Regression

### Mean Square Error (MSE) / L2 Loss

> MSE is a standard loss function utilized in most regression tasks since it directs the model to optimize to minimize the squared differences between the predicted and target values.

$MSE  = \dfrac{1}{n} * \sum(y_i -\bar{y})^2$

Where:

    n is the number of samples in the dataset
    yᵢ is the predicted value for the i-th sample
    ȳ is the target value for the i-th sample

### Mean Absolute Error (MAE) / L1 Loss

> A scenario where MAE is an applicable loss function is one where we don’t want to penalize outliers considerably or at all, for example, predicting delivery times for a food delivery company.

$MAE  = \dfrac{1}{n} * \sum(y_i -\bar{y})$

Where:

    n is the number of samples in the dataset
    yᵢ is the predicted value for the i-th sample
    ȳ is the target value for the i-th sample

### Huber Loss / Smooth Mean Absolute Error

> The Huber Loss function effectively combines two components for handling errors differently, with the transition point between these components determined by the threshold δ:\
> Quadratic Component for Small Errors: For errors smaller than δ, it uses the quadratic component (1/2) \* (f(x) - y)^2\
> Linear Component for Large Errors: For errors larger than δ, it applies the linear component δ \* \|f(x) - y\| - (1/2) \* δ^2

$L(\delta, y , f(x)) = \dfrac{1}{2} * (f(x) - y)^2 \quad if \quad |f(x) - y| <= \delta$\
$\quad \quad \quad \quad = \delta * |f(x) - y| - \dfrac{1}{2} * \delta^2 \quad if \quad |f(x) - y| > \delta$

Where:

    L represents the Huber Loss function
    δ is the delta parameter, which determines the threshold for switching between the quadratic and linear components of the loss function
    y is the true value or target value
    f(x) is the predicted value

Back to top
