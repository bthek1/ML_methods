# Ensemble Methods

Bagging and Boosting

Author

Benedict Thekkel

## Bagging

``` python
import pandas as pd

df = pd.read_csv("Data/diabetes.csv")
df.head()
```

|  | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|----|----|----|----|----|----|----|----|----|----|
| 0 | 6 | 148 | 72 | 35 | 0 | 33.6 | 0.627 | 50 | 1 |
| 1 | 1 | 85 | 66 | 29 | 0 | 26.6 | 0.351 | 31 | 0 |
| 2 | 8 | 183 | 64 | 0 | 0 | 23.3 | 0.672 | 32 | 1 |
| 3 | 1 | 89 | 66 | 23 | 94 | 28.1 | 0.167 | 21 | 0 |
| 4 | 0 | 137 | 40 | 35 | 168 | 43.1 | 2.288 | 33 | 1 |

``` python
df.isnull().sum()
```

    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64

``` python
df.describe()
```

|  | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|----|----|----|----|----|----|----|----|----|----|
| count | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 |
| mean | 3.845052 | 120.894531 | 69.105469 | 20.536458 | 79.799479 | 31.992578 | 0.471876 | 33.240885 | 0.348958 |
| std | 3.369578 | 31.972618 | 19.355807 | 15.952218 | 115.244002 | 7.884160 | 0.331329 | 11.760232 | 0.476951 |
| min | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.078000 | 21.000000 | 0.000000 |
| 25% | 1.000000 | 99.000000 | 62.000000 | 0.000000 | 0.000000 | 27.300000 | 0.243750 | 24.000000 | 0.000000 |
| 50% | 3.000000 | 117.000000 | 72.000000 | 23.000000 | 30.500000 | 32.000000 | 0.372500 | 29.000000 | 0.000000 |
| 75% | 6.000000 | 140.250000 | 80.000000 | 32.000000 | 127.250000 | 36.600000 | 0.626250 | 41.000000 | 1.000000 |
| max | 17.000000 | 199.000000 | 122.000000 | 99.000000 | 846.000000 | 67.100000 | 2.420000 | 81.000000 | 1.000000 |

``` python
df.Outcome.value_counts()
```

    0    500
    1    268
    Name: Outcome, dtype: int64

There is slight imbalance in our dataset but since it is not major we will not worry about it!

### Train test split

``` python
X = df.drop("Outcome",axis="columns")
y = df.Outcome
```

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]
```

    array([[ 0.63994726,  0.84832379,  0.14964075,  0.90726993, -0.69289057,
             0.20401277,  0.46849198,  1.4259954 ],
           [-0.84488505, -1.12339636, -0.16054575,  0.53090156, -0.69289057,
            -0.68442195, -0.36506078, -0.19067191],
           [ 1.23388019,  1.94372388, -0.26394125, -1.28821221, -0.69289057,
            -1.10325546,  0.60439732, -0.10558415]])

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
```

``` python
X_train.shape
```

    (576, 8)

``` python
X_test.shape
```

    (192, 8)

``` python
y_train.value_counts()
```

    0    375
    1    201
    Name: Outcome, dtype: int64

``` python
201/375
```

    0.536

``` python
y_test.value_counts()
```

    0    125
    1     67
    Name: Outcome, dtype: int64

``` python
67/125
```

    0.536

### Train using stand alone model

``` python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores
```

    array([0.68831169, 0.68181818, 0.68831169, 0.78431373, 0.73202614])

``` python
scores.mean()
```

    0.7149562855445208

### Train using Bagging

``` python
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
   estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
bag_model.oob_score_
```

    0.7534722222222222

``` python
bag_model.score(X_test, y_test)
```

    0.7760416666666666

``` python
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, X, y, cv=5)
scores
```

    array([0.75324675, 0.72727273, 0.74675325, 0.82352941, 0.73856209])

``` python
scores.mean()
```

    0.7578728461081402

We can see some improvement in test score with bagging classifier as compared to a standalone classifier

### Train using Random Forest

``` python
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(n_estimators=100), X, y, cv=5)
scores.mean()
```

    0.772192513368984

## Boosting

- **Gradient Boosting** – It is a boosting technique that builds a final model from the sum of several weak learning algorithms that were trained on the same dataset. It operates on the idea of stagewise addition. The first weak learner in the gradient boosting algorithm will not be trained on the dataset; instead, it will simply return the mean of the relevant column. The residual for the first weak learner algorithm’s output will then be calculated and used as the output column or target column for the next weak learning algorithm that will be trained. The second weak learner will be trained using the same methodology, and the residuals will be computed and utilized as an output column once more for the third weak learner, and so on until we achieve zero residuals. The dataset for gradient boosting must be in the form of numerical or categorical data, and the loss function used to generate the residuals must be differential at all times.
- **XGBoost** – In addition to the gradient boosting technique, XGBoost is another boosting machine learning approach. The full name of the XGBoost algorithm is the eXtreme Gradient Boosting algorithm, which is an extreme variation of the previous gradient boosting technique. The key distinction between XGBoost and GradientBoosting is that XGBoost applies a regularisation approach. It is a regularised version of the current gradient-boosting technique. Because of this, XGBoost outperforms a standard gradient boosting method, which explains why it is also faster than that. Additionally, it works better when the dataset contains both numerical and categorical variables.
- **Adaboost** – AdaBoost is a boosting algorithm that also works on the principle of the stagewise addition method where multiple weak learners are used for getting strong learners. The value of the alpha parameter, in this case, will be indirectly proportional to the error of the weak learner, Unlike Gradient Boosting in XGBoost, the alpha parameter calculated is related to the errors of the weak learner, here the value of the alpha parameter will be indirectly proportional to the error of the weak learner.
- **CatBoost** – The growth of decision trees inside CatBoost is the primary distinction that sets it apart from and improves upon competitors. The decision trees that are created in CatBoost are symmetric. As there is a unique sort of approach for handling categorical datasets, CatBoost works very well on categorical datasets compared to any other algorithm in the field of machine learning. The categorical features in CatBoost are encoded based on the output columns. As a result, the output column’s weight will be taken into account while training or encoding the categorical features, increasing its accuracy on categorical datasets.

``` python
# for classification
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

    GradientBoostingClassifier()

**In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.**

GradientBoostingClassifier

    GradientBoostingClassifier()

``` python
model.score(X_test, y_test)
```

    0.796875

``` python
from sklearn.ensemble import AdaBoostClassifier
```

``` python
clf = AdaBoostClassifier(
    n_estimators=100,
    random_state=0,
    algorithm='SAMME')
```

``` python
clf.fit(X_train, y_train)
```

    AdaBoostClassifier(algorithm='SAMME', n_estimators=100, random_state=0)

**In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.**

AdaBoostClassifier

    AdaBoostClassifier(algorithm='SAMME', n_estimators=100, random_state=0)

``` python
clf.score(X_test, y_test)
```

    0.796875

Back to top
