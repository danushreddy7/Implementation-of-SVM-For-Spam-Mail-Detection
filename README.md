# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: T DANUSH REDDY 
RegisterNumber: 212223040029

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy 
*/
```

## Output:
![SVM For Spam Mail Detection](sam.png)
# Result Output:
![282257583-78ccb346-ca7c-4a33-ad4c-e3355e1fddc6](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/d18d5a48-423a-41ea-843e-d8a2ded00393)
# data.head():
![282257589-139f19db-04ee-4e44-b04a-5f231988b90b](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/ebb0be01-a42b-4643-b4b8-3a94c65df979)
# data.info()
![282257595-646ac557-8f21-442c-8783-6a1085ec89fd](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/aaf40d49-0b01-4689-b897-82cfff9c48a0)
# data.isnull().sum()
![282257608-2eba109c-0bdd-468a-8bcf-d9258c23f8ef](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/5b3e889e-4282-418f-b58b-2d42c489edff)
![282257631-b0e5dbc6-7c5b-40fe-b610-86fc97828918](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/99fd5aac-057b-4e9f-a390-5824196d35f1)
# Y_prediction Value:
![282257837-6a911f5c-1e40-4047-9371-07a94f012cef](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/a03e8342-8ed3-49c5-b1ff-61d0f68d6db9)
# Accuracy Value:
![282257853-bad89364-aef2-4652-806d-09c5760c041e](https://github.com/danushreddy7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035740/acd03434-3d5e-4275-9b6c-aaaf0416a53f)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
