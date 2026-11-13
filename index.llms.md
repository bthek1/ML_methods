# ML Methods

Classical machine learning worked through one algorithm at a time: what it assumes, where it breaks, and a runnable example for each.

Author

Benedict Thekkel

This is the pre-deep-learning half of machine learning, the part that still solves most problems that arrive as a table of numbers. Each page takes one algorithm, states the intuition in a line, lists what it is good and bad at, then works an example in scikit-learn.

Several algorithms get two pages: one for the method and one marked `TUT` applying it to a real dataset. Read them as a pair.

For neural networks see [DL Methods](https://bthek1.github.io/DL_methods/), and for forecasting see [Time Series](https://bthek1.github.io/ML_Timeseries/).

------------------------------------------------------------------------

## Foundations

The parts that apply no matter which model you pick.

| Page | Covers |
|----|----|
| [Training and Testing](train_test_split.llms.md) | Why you hold data back at all, and how `train_test_split` does it |
| [Loss and Cost Functions](loss_or_cost_funtions.llms.md) | The common losses, and which ones suit classification against regression |
| [L1 and L2 Regularization](l1_and_l2_regularization.llms.md) | Overfitting shown on a plain regression, then Lasso and Ridge as the fix |
| [Hyper-parameter Optimization](hyperparameteroptimization.llms.md) | Three escalating approaches: manual trial and error over a split, then K-fold cross validation, then `GridSearchCV` |

------------------------------------------------------------------------

## Regression

Predicting a continuous value.

| Page | Covers |
|----|----|
| [Linear Regression](linearregression.llms.md) | The single-variable case, ending in generated house price predictions |
| [Multi-Linear Regression](multi-linearregression.llms.md) | The same with several input variables |
| [Polynomial Regression](polynomialregression.llms.md) | Fitting curves, second order and higher |
| [Kernel Ridge Regression](kernalsridgeregression.llms.md) | Ridge regression combined with the kernel trick, for non-linear relationships |
| [Support Vector Regression](svm.llms.md) | SVM applied to a continuous target |

------------------------------------------------------------------------

## Classification

Predicting a category.

| Page | Covers |
|----|----|
| [K-NN](k-nn.llms.md) | Take any point, it is what it is close to. Distance metrics, choosing k, then a first model and a `GridSearchCV` search for the optimum |
| [Decision Tree](decisiontree.llms.md) | If-statements placed where entropy drops fastest, plus the trade-offs of CART |
| [Logistic Regression](logisticregression.llms.md) | Binary and multiclass, and the classification types each suits |
| [Naive Bayes](bayesian.llms.md) | The Bayesian approach and Gaussian processes |
| [Naive Bayes Tutorial](naive_bayes_tut.llms.md) | The same method built into an email spam filter |
| [Random Forest](random_forest.llms.md) | Many trees voting |

------------------------------------------------------------------------

## Ensembles

| Page | Covers |
|----|----|
| [Ensemble Methods](ensemble.llms.md) | Bagging against boosting, and when each helps |
| [Ensemble Tutorial](ensemble_tut.llms.md) | Bagging applied to heart disease prediction, then boosting |
| [Bagging: Diabetes Prediction](bagging_diabetes_prediction.llms.md) | A second worked bagging example |

------------------------------------------------------------------------

## Unsupervised Learning

Finding structure without labels.

| Page | Covers |
|----|----|
| [Principal Component Analysis](pca.llms.md) | Dimensionality reduction |
| [PCA Tutorial](pca_tut.llms.md) | PCA applied to heart disease prediction |
| [Gaussian Mixture Models](gaussian_mixture_models.llms.md) | Modelling data as a mixture of Gaussians, the expectation-maximization algorithm behind it, and its use for clustering, density estimation, and anomaly detection |
| [K-Means Clustering](k_mean_clustering.llms.md) | The algorithm step by step, choosing K, and the alternatives when it does not fit |

------------------------------------------------------------------------

## GPU Programming

| Page | Covers |
|----|----|
| [CUDA](cuda.llms.md) | The odd one out, and the longest page here. Getting a CUDA toolchain working, then writing kernels: RGB to greyscale, then matrix multiplication. Useful background for why the deep learning libraries are shaped the way they are |

------------------------------------------------------------------------

## Not Covered Yet

- **Three pages are placeholders** marked “coming soon” with no content: [CNNs](cnns.llms.md), [Generative Models](generative.llms.md), and [Back Propagation](backpropagation.llms.md). All three are really deep learning topics and are covered properly in [DL Methods](https://bthek1.github.io/DL_methods/).
- **Nothing on gradient boosting libraries.** No XGBoost, LightGBM, or CatBoost page, despite them being the usual winners on tabular data. XGBoost is covered from the forecasting angle in [Time Series](https://bthek1.github.io/ML_Timeseries/).
- **No feature engineering page**, and no treatment of missing data, categorical encoding, or scaling as topics in their own right.
- **No model interpretability**: nothing on SHAP, permutation importance, or partial dependence.
- Evaluation is spread through the tutorials rather than collected anywhere, so there is no single page on precision, recall, ROC, or calibration.

------------------------------------------------------------------------

Back to top
