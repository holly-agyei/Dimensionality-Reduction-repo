import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("Diabetes_Dataset.csv")

#Size of dataset
print("Size of dataset: ", dataset.size)

#Drop all null values
df.dropna()

#Info about data
print(dataset.info())

#Statistical description of the the dataset
print("Statistical Description of the dataset \n", dataset.describe())

print(dataset.corr())

X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled, y_train)

from sklearn.metrics import f1_score

y_pred = svm_classifier.predict(X_test_scaled)
accuracy = f1_score(y_test, y_pred)
print("F1 score: ", accuracy)