#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Read the csv file
data = pd.read_csv("adult 3.csv")
data.head()

data.shape

#Find the count of null value of each column
data.isna().sum()

print(data.occupation.value_counts())

print(data.gender.value_counts())

print(data.education.value_counts())

print(data['marital-status'].value_counts())

print(data['workclass'].value_counts())

data.occupation.replace({'?':'Others'},inplace=True)

print(data.occupation.value_counts())

data.workclass.replace({'?':'NotListed'},inplace=True)

print(data['workclass'].value_counts())

data = data[data['workclass']!= 'Without-pay']
data = data[data['workclass']!= 'Never-worked']

print(data['workclass'].value_counts())

data.shape

data = data[data['education']!= '5th-6th']
data = data[data['education']!= '1st-4th']
data = data[data['education']!= 'Preschool']

print(data.education.value_counts())

data.shape

#Redundancy
data.drop(columns='education', inplace=True)

data

plt.boxplot(data['age'])
plt.show()

data = data[(data['age']<=75) & (data['age']>=17)]

plt.boxplot(data['age'])
plt.show()

#initialize the lable encoder
encoder = LabelEncoder()
data['workclass'] = encoder.fit_transform(data['workclass'])
data['occupation'] = encoder.fit_transform(data['occupation'])
data['relationship'] = encoder.fit_transform(data['relationship'])
data['race'] = encoder.fit_transform(data['race'])
data['gender'] = encoder.fit_transform(data['gender'])
data['native-country'] = encoder.fit_transform(data['native-country'])
data['marital-status'] = encoder.fit_transform(data['marital-status'])
data

#Split the data into input and otput
x = data.drop(columns='income') #Input x
y = data['income'] #output y
x

scaler = StandardScaler()
x = scaler.fit_transform(x)
x

#Split the data into training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y) #stratify => it is try to divide the data equally or evenly.  y => target variable

xtrain

#machine learning algorithms
best_model = RandomForestClassifier()
best_model.fit(xtrain, ytrain) #input and output training data
predict = best_model.predict(xtest) #predict the data/value
predict

#check the accuracy and Create the classification report using precision, recall, f1-score, support
accuracy = accuracy_score(ytest, predict)
print(f'Accuracy is : {accuracy*100:.2f}')
print(classification_report(ytest, predict))

#save the trained model in .pkl format
import joblib
joblib.dump(best_model, 'model.pkl')
print("Model saved as model.pkl")