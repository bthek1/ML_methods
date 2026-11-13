# The Naive Bayes Tutorial

The Naive Bayes Tutorial: Email spam filter

Author

Benedict Thekkel

``` python
import pandas as pd
```

``` python
df = pd.read_csv("Data/spam.csv")
df.head()
```

|     | Category | Message                                           |
|-----|----------|---------------------------------------------------|
| 0   | ham      | Go until jurong point, crazy.. Available only ... |
| 1   | ham      | Ok lar... Joking wif u oni...                     |
| 2   | spam     | Free entry in 2 a wkly comp to win FA Cup fina... |
| 3   | ham      | U dun say so early hor... U c already then say... |
| 4   | ham      | Nah I don't think he goes to usf, he lives aro... |

``` python
df.groupby('Category').describe()
```

|  | Message |  |  |  |
|----|----|----|----|----|
|  | count | unique | top | freq |
| Category |  |  |  |  |
| ham | 4825 | 4516 | Sorry, I'll call later | 30 |
| spam | 747 | 641 | Please call our customer service representativ... | 4 |

``` python
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()
```

|     | Category | Message                                           | spam |
|-----|----------|---------------------------------------------------|------|
| 0   | ham      | Go until jurong point, crazy.. Available only ... | 0    |
| 1   | ham      | Ok lar... Joking wif u oni...                     | 0    |
| 2   | spam     | Free entry in 2 a wkly comp to win FA Cup fina... | 1    |
| 3   | ham      | U dun say so early hor... U c already then say... | 0    |
| 4   | ham      | Nah I don't think he goes to usf, he lives aro... | 0    |

``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam)
```

``` python
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]
```

    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])

``` python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)
```

    MultinomialNB()

**In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.**

MultinomialNB

    MultinomialNB()

``` python
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)
```

    array([0, 1])

``` python
X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)
```

    0.9863603732950467

**Sklearn Pipeline**

``` python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
```

``` python
clf.fit(X_train, y_train)
```

    Pipeline(steps=[('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])

**In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.**

Pipeline

    Pipeline(steps=[('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])

CountVectorizer

    CountVectorizer()

MultinomialNB

    MultinomialNB()

``` python
clf.score(X_test,y_test)
```

    0.9863603732950467

``` python
clf.predict(emails)
```

    array([0, 1])

Back to top
