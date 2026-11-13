# The Naive Bayes Approach

The Bayesian Approach and Gaussian Processes

Author

Benedict Thekkel

$P(queen/diamond) = \dfrac{P(diamond/queen) * P(queen)}{P(diamond)}$

\$P(queen/diamond) = \$

$P(diamond/queen) = 1/4$\
$P(queen) = 1/13$\
$P(diamond) = 1/4$

$P(queen/diamond) = \dfrac{1/4 * 1/13}{1/4} = 1/13$

``` python
import pandas as pd
```

``` python
df = pd.read_csv("Data/titanic.csv")
df.head()
```

|  | PassengerId | Name | Pclass | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked | Survived |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1 | Braund, Mr. Owen Harris | 3 | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S | 0 |
| 1 | 2 | Cumings, Mrs. John Bradley (Florence Briggs Th... | 1 | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C | 1 |
| 2 | 3 | Heikkinen, Miss. Laina | 3 | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN | S | 1 |
| 3 | 4 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | 1 | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S | 1 |
| 4 | 5 | Allen, Mr. William Henry | 3 | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S | 0 |

``` python
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()
```

|     | Pclass | Sex    | Age  | Fare    | Survived |
|-----|--------|--------|------|---------|----------|
| 0   | 3      | male   | 22.0 | 7.2500  | 0        |
| 1   | 1      | female | 38.0 | 71.2833 | 1        |
| 2   | 3      | female | 26.0 | 7.9250  | 1        |
| 3   | 1      | female | 35.0 | 53.1000 | 1        |
| 4   | 3      | male   | 35.0 | 8.0500  | 0        |

``` python
inputs = df.drop('Survived',axis='columns')
target = df.Survived
```

``` python
#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
```

``` python
dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)
```

|     | female | male |
|-----|--------|------|
| 0   | 0      | 1    |
| 1   | 1      | 0    |
| 2   | 1      | 0    |

``` python
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)
```

|     | Pclass | Sex    | Age  | Fare    | female | male |
|-----|--------|--------|------|---------|--------|------|
| 0   | 3      | male   | 22.0 | 7.2500  | 0      | 1    |
| 1   | 1      | female | 38.0 | 71.2833 | 1      | 0    |
| 2   | 3      | female | 26.0 | 7.9250  | 1      | 0    |

**I am dropping male column as well because of dummy variable trap theory. One column is enough to repressent male vs female**

``` python
inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)
```

|     | Pclass | Age  | Fare    | female |
|-----|--------|------|---------|--------|
| 0   | 3      | 22.0 | 7.2500  | 0      |
| 1   | 1      | 38.0 | 71.2833 | 1      |
| 2   | 3      | 26.0 | 7.9250  | 1      |

``` python
inputs.columns[inputs.isna().any()]
```

    Index(['Age'], dtype='object')

``` python
inputs.Age[:10]
```

    0    22.0
    1    38.0
    2    26.0
    3    35.0
    4    35.0
    5     NaN
    6    54.0
    7     2.0
    8    27.0
    9    14.0
    Name: Age, dtype: float64

``` python
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()
```

|     | Pclass | Age  | Fare    | female |
|-----|--------|------|---------|--------|
| 0   | 3      | 22.0 | 7.2500  | 0      |
| 1   | 1      | 38.0 | 71.2833 | 1      |
| 2   | 3      | 26.0 | 7.9250  | 1      |
| 3   | 1      | 35.0 | 53.1000 | 1      |
| 4   | 3      | 35.0 | 8.0500  | 0      |

``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
```

``` python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```

``` python
model.fit(X_train,y_train)
```

    GaussianNB()

**In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.**

GaussianNB

    GaussianNB()

``` python
model.score(X_test,y_test)
```

    0.7910447761194029

``` python
X_test[0:10]
```

|     | Pclass | Age       | Fare     | female |
|-----|--------|-----------|----------|--------|
| 509 | 3      | 26.000000 | 56.4958  | 0      |
| 325 | 1      | 36.000000 | 135.6333 | 1      |
| 248 | 1      | 37.000000 | 52.5542  | 0      |
| 391 | 3      | 21.000000 | 7.7958   | 0      |
| 411 | 3      | 29.699118 | 6.8583   | 0      |
| 688 | 3      | 18.000000 | 7.7958   | 0      |
| 183 | 2      | 1.000000  | 39.0000  | 0      |
| 14  | 3      | 14.000000 | 7.8542   | 1      |
| 763 | 1      | 36.000000 | 120.0000 | 1      |
| 383 | 1      | 35.000000 | 52.0000  | 1      |

``` python
y_test[0:10]
```

    509    1
    325    1
    248    1
    391    1
    411    0
    688    0
    183    1
    14     0
    763    1
    383    1
    Name: Survived, dtype: int64

``` python
model.predict(X_test[0:10])
```

    array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1])

``` python
model.predict_proba(X_test[:10])
```

    array([[9.22826078e-01, 7.71739224e-02],
           [1.90547332e-04, 9.99809453e-01],
           [6.93224146e-01, 3.06775854e-01],
           [9.59335969e-01, 4.06640310e-02],
           [9.65380973e-01, 3.46190265e-02],
           [9.56271850e-01, 4.37281504e-02],
           [8.17245910e-01, 1.82754090e-01],
           [3.83233278e-01, 6.16766722e-01],
           [9.28107033e-04, 9.99071893e-01],
           [6.70466692e-02, 9.32953331e-01]])

**Calculate the score using cross validation**

``` python
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)
```

    array([0.784     , 0.728     , 0.744     , 0.75806452, 0.80645161])

Back to top
