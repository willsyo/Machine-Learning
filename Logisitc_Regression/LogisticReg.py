import numpy as np
import pandas as pd
import seaborn as sb
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare','Cabin', 'Embarked']

sb.countplot(x='Survived', data=titanic, palette='hls')
titanic.isnull().sum()

titanic_data = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)
print (titanic_data.head())

sb.boxplot(x='Pclass', y='Age', data=titanic_data, palette='hls')

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.isnull().sum()

titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()

gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_location = pd.get_dummies(titanic_data['Embarked'], drop_first=True)

titanic_data.drop(['Sex', 'Embarked'], 1, inplace=True)
titanic_dmy = pd.concat([titanic_data, gender, embark_location], axis=1)

titanic_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)
X = titanic_dmy.ix[:, (1, 2, 3, 4, 5, 6)].values
y = titanic_dmy.ix[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=25)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

print(classification_report(y_test, y_pred))