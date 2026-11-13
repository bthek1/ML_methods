# Ensemble Tutorial

Ensemble Learning: Bagging Tutorial: Heart disease prediction

Author

Benedict Thekkel

dataset credits: https://www.kaggle.com/fedesoriano/heart-failure-prediction

### Data Loading

``` python
import pandas as pd

df = pd.read_csv("Data/heart.csv")
df.head()
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 40 | M | ATA | 140 | 289 | 0 | Normal | 172 | N | 0.0 | Up | 0 |
| 1 | 49 | F | NAP | 160 | 180 | 0 | Normal | 156 | N | 1.0 | Flat | 1 |
| 2 | 37 | M | ATA | 130 | 283 | 0 | ST | 98 | N | 0.0 | Up | 0 |
| 3 | 48 | F | ASY | 138 | 214 | 0 | Normal | 108 | Y | 1.5 | Flat | 1 |
| 4 | 54 | M | NAP | 150 | 195 | 0 | Normal | 122 | N | 0.0 | Up | 0 |

``` python
df.shape
```

    (918, 12)

``` python
df.describe()
```

|  | Age | RestingBP | Cholesterol | FastingBS | MaxHR | Oldpeak | HeartDisease |
|----|----|----|----|----|----|----|----|
| count | 918.000000 | 918.000000 | 918.000000 | 918.000000 | 918.000000 | 918.000000 | 918.000000 |
| mean | 53.510893 | 132.396514 | 198.799564 | 0.233115 | 136.809368 | 0.887364 | 0.553377 |
| std | 9.432617 | 18.514154 | 109.384145 | 0.423046 | 25.460334 | 1.066570 | 0.497414 |
| min | 28.000000 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | -2.600000 | 0.000000 |
| 25% | 47.000000 | 120.000000 | 173.250000 | 0.000000 | 120.000000 | 0.000000 | 0.000000 |
| 50% | 54.000000 | 130.000000 | 223.000000 | 0.000000 | 138.000000 | 0.600000 | 1.000000 |
| 75% | 60.000000 | 140.000000 | 267.000000 | 0.000000 | 156.000000 | 1.500000 | 1.000000 |
| max | 77.000000 | 200.000000 | 603.000000 | 1.000000 | 202.000000 | 6.200000 | 1.000000 |

### Treat Outliers

``` python
df[df.Cholesterol>(df.Cholesterol.mean()+3*df.Cholesterol.std())]
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 76 | 32 | M | ASY | 118 | 529 | 0 | Normal | 130 | N | 0.0 | Flat | 1 |
| 149 | 54 | M | ASY | 130 | 603 | 1 | Normal | 125 | Y | 1.0 | Flat | 1 |
| 616 | 67 | F | NAP | 115 | 564 | 0 | LVH | 160 | N | 1.6 | Flat | 0 |

``` python
df.shape
```

    (918, 12)

``` python
df1 = df[df.Cholesterol<=(df.Cholesterol.mean()+3*df.Cholesterol.std())]
df1.shape
```

    (915, 12)

``` python
df[df.MaxHR>(df.MaxHR.mean()+3*df.MaxHR.std())]
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|

``` python
df[df.FastingBS>(df.FastingBS.mean()+3*df.FastingBS.std())]
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|

``` python
df[df.Oldpeak>(df.Oldpeak.mean()+3*df.Oldpeak.std())]
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 166 | 50 | M | ASY | 140 | 231 | 0 | ST | 140 | Y | 5.0 | Flat | 1 |
| 702 | 59 | M | TA | 178 | 270 | 0 | LVH | 145 | N | 4.2 | Down | 0 |
| 771 | 55 | M | ASY | 140 | 217 | 0 | Normal | 111 | Y | 5.6 | Down | 1 |
| 791 | 51 | M | ASY | 140 | 298 | 0 | Normal | 122 | Y | 4.2 | Flat | 1 |
| 850 | 62 | F | ASY | 160 | 164 | 0 | LVH | 145 | N | 6.2 | Down | 1 |
| 900 | 58 | M | ASY | 114 | 318 | 0 | ST | 140 | N | 4.4 | Down | 1 |

``` python
df2 = df1[df1.Oldpeak<=(df1.Oldpeak.mean()+3*df1.Oldpeak.std())]
df2.shape
```

    (909, 12)

``` python
df[df.RestingBP>(df.RestingBP.mean()+3*df.RestingBP.std())]
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 109 | 39 | M | ATA | 190 | 241 | 0 | Normal | 106 | N | 0.0 | Up | 0 |
| 241 | 54 | M | ASY | 200 | 198 | 0 | Normal | 142 | Y | 2.0 | Flat | 1 |
| 365 | 64 | F | ASY | 200 | 0 | 0 | Normal | 140 | Y | 1.0 | Flat | 1 |
| 399 | 61 | M | NAP | 200 | 0 | 1 | ST | 70 | N | 0.0 | Flat | 1 |
| 592 | 61 | M | ASY | 190 | 287 | 1 | LVH | 150 | Y | 2.0 | Down | 1 |
| 732 | 56 | F | ASY | 200 | 288 | 1 | LVH | 133 | Y | 4.0 | Down | 1 |
| 759 | 54 | M | ATA | 192 | 283 | 0 | LVH | 195 | N | 0.0 | Up | 1 |

``` python
df3 = df2[df2.RestingBP<=(df2.RestingBP.mean()+3*df2.RestingBP.std())]
df3.shape
```

    (902, 12)

``` python
df.ChestPainType.unique()
```

    array(['ATA', 'NAP', 'ASY', 'TA'], dtype=object)

``` python
df.RestingECG.unique()
```

    array(['Normal', 'ST', 'LVH'], dtype=object)

``` python
df.ExerciseAngina.unique()
```

    array(['N', 'Y'], dtype=object)

``` python
df.ST_Slope.unique()
```

    array(['Up', 'Flat', 'Down'], dtype=object)

### Handle text columns using label encoding and one hot encoding

``` python
df4 = df3.copy()
df4.ExerciseAngina.replace(
    {
        'N': 0,
        'Y': 1
    },
    inplace=True)

df4.ST_Slope.replace(
    {
        'Down': 1,
        'Flat': 2,
        'Up': 3
    },
    inplace=True
)

df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    },
    inplace=True)

df4.head()
```

|  | Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 40 | M | ATA | 140 | 289 | 0 | 1 | 172 | 0 | 0.0 | 3 | 0 |
| 1 | 49 | F | NAP | 160 | 180 | 0 | 1 | 156 | 0 | 1.0 | 2 | 1 |
| 2 | 37 | M | ATA | 130 | 283 | 0 | 2 | 98 | 0 | 0.0 | 3 | 0 |
| 3 | 48 | F | ASY | 138 | 214 | 0 | 1 | 108 | 1 | 1.5 | 2 | 1 |
| 4 | 54 | M | NAP | 150 | 195 | 0 | 1 | 122 | 0 | 0.0 | 3 | 0 |

``` python
df5 = pd.get_dummies(df4, drop_first=True)
df5.head()
```

|  | Age | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease | Sex_M | ChestPainType_ATA | ChestPainType_NAP | ChestPainType_TA |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 40 | 140 | 289 | 0 | 1 | 172 | 0 | 0.0 | 3 | 0 | 1 | 1 | 0 | 0 |
| 1 | 49 | 160 | 180 | 0 | 1 | 156 | 0 | 1.0 | 2 | 1 | 0 | 0 | 1 | 0 |
| 2 | 37 | 130 | 283 | 0 | 2 | 98 | 0 | 0.0 | 3 | 0 | 1 | 1 | 0 | 0 |
| 3 | 48 | 138 | 214 | 0 | 1 | 108 | 1 | 1.5 | 2 | 1 | 0 | 0 | 0 | 0 |
| 4 | 54 | 150 | 195 | 0 | 1 | 122 | 0 | 0.0 | 3 | 0 | 1 | 0 | 1 | 0 |

``` python
X = df5.drop("HeartDisease",axis='columns')
y = df5.HeartDisease

X.head()
```

|  | Age | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | Sex_M | ChestPainType_ATA | ChestPainType_NAP | ChestPainType_TA |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 40 | 140 | 289 | 0 | 1 | 172 | 0 | 0.0 | 3 | 1 | 1 | 0 | 0 |
| 1 | 49 | 160 | 180 | 0 | 1 | 156 | 0 | 1.0 | 2 | 0 | 0 | 1 | 0 |
| 2 | 37 | 130 | 283 | 0 | 2 | 98 | 0 | 0.0 | 3 | 1 | 1 | 0 | 0 |
| 3 | 48 | 138 | 214 | 0 | 1 | 108 | 1 | 1.5 | 2 | 0 | 0 | 0 | 0 |
| 4 | 54 | 150 | 195 | 0 | 1 | 122 | 0 | 0.0 | 3 | 1 | 0 | 1 | 0 |

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
```

    array([[-1.42896269,  0.46089071,  0.85238015, ...,  2.06757196,
            -0.53547478, -0.22914788],
           [-0.47545956,  1.5925728 , -0.16132855, ..., -0.4836591 ,
             1.86750159, -0.22914788],
           [-1.74679706, -0.10495034,  0.79657967, ...,  2.06757196,
            -0.53547478, -0.22914788],
           ...,
           [ 0.37209878, -0.10495034, -0.61703246, ..., -0.4836591 ,
            -0.53547478, -0.22914788],
           [ 0.37209878, -0.10495034,  0.35947592, ...,  2.06757196,
            -0.53547478, -0.22914788],
           [-1.64085227,  0.3477225 , -0.20782894, ..., -0.4836591 ,
             1.86750159, -0.22914788]])

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)
```

``` python
X_train.shape
```

    (721, 13)

``` python
X_test.shape
```

    (181, 13)

### Train a model using standalone support vector machine and then using bagging

``` python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

scores = cross_val_score(SVC(), X, y, cv=5)
scores.mean()
```

    0.6906445672191528

**Use bagging now with svm**

``` python
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(estimator=SVC(), n_estimators=100, max_samples=0.8, random_state=0)
scores = cross_val_score(bag_model, X, y, cv=5)
scores.mean()
```

    0.6839656230816453

As you can see above, using bagging in case of SVM doesn’t make much difference in terms of model accuracy. Bagging is effective when we have high variance and instable model such as decision tree. Let’s explore how bagging changes the performance for a decision tree classifier.

### Train a model using decision tree and then using bagging

``` python
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5)
scores.mean()
```

    0.7193984039287907

**Use bagging now with decision tree**

``` python
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=0), 
    n_estimators=100, 
    max_samples=0.9, 
    oob_score=True,
    random_state=0
)

scores = cross_val_score(bag_model, X, y, cv=5)
scores.mean()
```

    0.8037016574585636

**You can see that with bagging the score improved from 71.93% to 80.37%**

### Train a model using Random Forest which itself uses bagging underneath

``` python
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
scores.mean()
```

    0.826998158379374

## Boosting

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

    0.8397790055248618

**Random forest gave even a better performance with 81.7% as score. Underneath it used bagging where it sampled not only data rows but also the columns (or features)**

Back to top
