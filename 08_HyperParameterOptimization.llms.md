# Hyper Parameter Optimization

Hyper Parameter Optimization

Author

Benedict Thekkel

**For iris flower dataset in sklearn library, we are going to find out best model and best hyper parameters using GridSearchCV**

![](iris_petal_sepal.png)

**Load iris flower dataset**

``` python
from sklearn import svm, datasets
iris = datasets.load_iris()
```

``` python
import pandas as pd
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
df[47:150]
```

|  | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | flower |
|----|----|----|----|----|----|
| 47 | 4.6 | 3.2 | 1.4 | 0.2 | setosa |
| 48 | 5.3 | 3.7 | 1.5 | 0.2 | setosa |
| 49 | 5.0 | 3.3 | 1.4 | 0.2 | setosa |
| 50 | 7.0 | 3.2 | 4.7 | 1.4 | versicolor |
| 51 | 6.4 | 3.2 | 4.5 | 1.5 | versicolor |
| ... | ... | ... | ... | ... | ... |
| 145 | 6.7 | 3.0 | 5.2 | 2.3 | virginica |
| 146 | 6.3 | 2.5 | 5.0 | 1.9 | virginica |
| 147 | 6.5 | 3.0 | 5.2 | 2.0 | virginica |
| 148 | 6.2 | 3.4 | 5.4 | 2.3 | virginica |
| 149 | 5.9 | 3.0 | 5.1 | 1.8 | virginica |

103 rows × 5 columns

## Approach 1: Use train_test_split and manually tune parameters by trial and error

``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
```

``` python
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
model.score(X_test, y_test)
```

    0.9555555555555556

## Approach 2: Use K Fold Cross validation

**Manually try suppling models with different parameters to cross_val_score function with 5 fold cross validation**

``` python
cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),iris.data, iris.target, cv=5)
```

    array([1.        , 1.        , 0.9       , 0.96666667, 1.        ])

``` python
cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),iris.data, iris.target, cv=5)
```

    array([0.96666667, 1.        , 0.96666667, 0.96666667, 1.        ])

``` python
cross_val_score(svm.SVC(kernel='rbf',C=20,gamma='auto'),iris.data, iris.target, cv=5)
```

    array([0.96666667, 1.        , 0.9       , 0.96666667, 1.        ])

**Above approach is tiresome and very manual. We can use for loop as an alternative**

``` python
kernels = ['rbf', 'linear']
C = [1,10,20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)

avg_scores
```

    {'rbf_1': 0.9800000000000001,
     'rbf_10': 0.9800000000000001,
     'rbf_20': 0.9666666666666668,
     'linear_1': 0.9800000000000001,
     'linear_10': 0.9733333333333334,
     'linear_20': 0.9666666666666666}

**From above results we can say that rbf with C=1 or 10 or linear with C=1 will give best performance**

## Approach 3: Use GridSearchCV

**GridSearchCV does exactly same thing as for loop above but in a single line of code**

``` python
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
clf.cv_results_
```

    {'mean_fit_time': array([0.00118256, 0.00104566, 0.0007266 , 0.00084271, 0.00088511,
            0.00065002]),
     'std_fit_time': array([6.73577797e-04, 4.38131552e-04, 1.68876617e-04, 3.96551972e-04,
            4.28567104e-04, 8.82425265e-05]),
     'mean_score_time': array([0.00081396, 0.00048532, 0.00048227, 0.00054522, 0.00063434,
            0.00044188]),
     'std_score_time': array([4.28254909e-04, 1.10279724e-04, 1.84006103e-04, 1.59358051e-04,
            3.89407295e-04, 6.20551122e-05]),
     'param_C': masked_array(data=[1, 1, 10, 10, 20, 20],
                  mask=[False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_kernel': masked_array(data=['rbf', 'linear', 'rbf', 'linear', 'rbf', 'linear'],
                  mask=[False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'C': 1, 'kernel': 'rbf'},
      {'C': 1, 'kernel': 'linear'},
      {'C': 10, 'kernel': 'rbf'},
      {'C': 10, 'kernel': 'linear'},
      {'C': 20, 'kernel': 'rbf'},
      {'C': 20, 'kernel': 'linear'}],
     'split0_test_score': array([0.96666667, 0.96666667, 0.96666667, 1.        , 0.96666667,
            1.        ]),
     'split1_test_score': array([1., 1., 1., 1., 1., 1.]),
     'split2_test_score': array([0.96666667, 0.96666667, 0.96666667, 0.9       , 0.9       ,
            0.9       ]),
     'split3_test_score': array([0.96666667, 0.96666667, 0.96666667, 0.96666667, 0.96666667,
            0.93333333]),
     'split4_test_score': array([1., 1., 1., 1., 1., 1.]),
     'mean_test_score': array([0.98      , 0.98      , 0.98      , 0.97333333, 0.96666667,
            0.96666667]),
     'std_test_score': array([0.01632993, 0.01632993, 0.01632993, 0.03887301, 0.03651484,
            0.0421637 ]),
     'rank_test_score': array([1, 1, 1, 4, 5, 6], dtype=int32)}

``` python
df = pd.DataFrame(clf.cv_results_)
df
```

|  | mean_fit_time | std_fit_time | mean_score_time | std_score_time | param_C | param_kernel | params | split0_test_score | split1_test_score | split2_test_score | split3_test_score | split4_test_score | mean_test_score | std_test_score | rank_test_score |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0.001183 | 0.000674 | 0.000814 | 0.000428 | 1 | rbf | {'C': 1, 'kernel': 'rbf'} | 0.966667 | 1.0 | 0.966667 | 0.966667 | 1.0 | 0.980000 | 0.016330 | 1 |
| 1 | 0.001046 | 0.000438 | 0.000485 | 0.000110 | 1 | linear | {'C': 1, 'kernel': 'linear'} | 0.966667 | 1.0 | 0.966667 | 0.966667 | 1.0 | 0.980000 | 0.016330 | 1 |
| 2 | 0.000727 | 0.000169 | 0.000482 | 0.000184 | 10 | rbf | {'C': 10, 'kernel': 'rbf'} | 0.966667 | 1.0 | 0.966667 | 0.966667 | 1.0 | 0.980000 | 0.016330 | 1 |
| 3 | 0.000843 | 0.000397 | 0.000545 | 0.000159 | 10 | linear | {'C': 10, 'kernel': 'linear'} | 1.000000 | 1.0 | 0.900000 | 0.966667 | 1.0 | 0.973333 | 0.038873 | 4 |
| 4 | 0.000885 | 0.000429 | 0.000634 | 0.000389 | 20 | rbf | {'C': 20, 'kernel': 'rbf'} | 0.966667 | 1.0 | 0.900000 | 0.966667 | 1.0 | 0.966667 | 0.036515 | 5 |
| 5 | 0.000650 | 0.000088 | 0.000442 | 0.000062 | 20 | linear | {'C': 20, 'kernel': 'linear'} | 1.000000 | 1.0 | 0.900000 | 0.933333 | 1.0 | 0.966667 | 0.042164 | 6 |

``` python
df[['param_C','param_kernel','mean_test_score']]
```

|     | param_C | param_kernel | mean_test_score |
|-----|---------|--------------|-----------------|
| 0   | 1       | rbf          | 0.980000        |
| 1   | 1       | linear       | 0.980000        |
| 2   | 10      | rbf          | 0.980000        |
| 3   | 10      | linear       | 0.973333        |
| 4   | 20      | rbf          | 0.966667        |
| 5   | 20      | linear       | 0.966667        |

``` python
clf.best_params_
```

    {'C': 1, 'kernel': 'rbf'}

``` python
clf.best_score_
```

    0.9800000000000001

``` python
dir(clf)
```

    ['__abstractmethods__',
     '__annotations__',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__sklearn_clone__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_abc_impl',
     '_build_request_for_signature',
     '_check_feature_names',
     '_check_n_features',
     '_check_refit_for_multimetric',
     '_estimator_type',
     '_format_results',
     '_get_default_requests',
     '_get_metadata_request',
     '_get_param_names',
     '_get_tags',
     '_more_tags',
     '_parameter_constraints',
     '_repr_html_',
     '_repr_html_inner',
     '_repr_mimebundle_',
     '_required_parameters',
     '_run_search',
     '_select_best_index',
     '_validate_data',
     '_validate_params',
     'best_estimator_',
     'best_index_',
     'best_params_',
     'best_score_',
     'classes_',
     'cv',
     'cv_results_',
     'decision_function',
     'error_score',
     'estimator',
     'fit',
     'get_metadata_routing',
     'get_params',
     'inverse_transform',
     'multimetric_',
     'n_features_in_',
     'n_jobs',
     'n_splits_',
     'param_grid',
     'pre_dispatch',
     'predict',
     'predict_log_proba',
     'predict_proba',
     'refit',
     'refit_time_',
     'return_train_score',
     'score',
     'score_samples',
     'scorer_',
     'scoring',
     'set_fit_request',
     'set_params',
     'transform',
     'verbose']

**Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters. This is useful when you have too many parameters to try and your training time is longer. It helps reduce the cost of computation**

``` python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2
)
rs.fit(iris.data, iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
```

|     | param_C | param_kernel | mean_test_score |
|-----|---------|--------------|-----------------|
| 0   | 20      | rbf          | 0.966667        |
| 1   | 1       | rbf          | 0.980000        |

**How about different models with different hyperparameters?**

``` python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
```

``` python
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': list(range(1, 21)),
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': list(range(1, 11))
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': list(range(1, 11))
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}
```

``` python
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df
```

|     | model                   | best_score | best_params               |
|-----|-------------------------|------------|---------------------------|
| 0   | svm                     | 0.986667   | {'C': 4, 'kernel': 'rbf'} |
| 1   | random_forest           | 0.966667   | {'n_estimators': 4}       |
| 2   | logistic_regression     | 0.966667   | {'C': 4}                  |
| 3   | naive_bayes_gaussian    | 0.953333   | {}                        |
| 4   | naive_bayes_multinomial | 0.953333   | {}                        |
| 5   | decision_tree           | 0.966667   | {'criterion': 'gini'}     |

**Based on above, I can conclude that SVM with C=1 and kernel=‘rbf’ is the best model for solving my problem of iris flower classification**

Back to top
