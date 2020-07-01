import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

ds = pd.read_csv("train.csv")

print("\nThe Dataset looks like the following: ")
print(ds.head())
print("\nColumns of dataset are: ")
print(ds.columns)

print('\n')
print(ds[['Pclass', 'Survived']].groupby('Pclass', as_index = False).count())
print('\n')
print(ds[['Sex', 'Survived']].groupby('Sex', as_index = False).count())
print('\n')
print(ds[['Age', 'Survived']].groupby('Age', as_index = False).count())
print('\n')
print(ds[['SibSp', 'Survived']].groupby('SibSp', as_index = False).count())
print('\n')
print(ds[['Parch', 'Survived']].groupby('Parch', as_index = False).count())
print('\n')
print(ds[['Ticket', 'Survived']].groupby('Ticket', as_index = False).count())
print('\n')
print(ds[['Fare', 'Survived']].groupby('Fare', as_index = False).count())
print('\n')
print(ds[['Cabin', 'Survived']].groupby('Cabin', as_index = False).count())
print('\n')
print(ds[['Embarked', 'Survived']].groupby('Embarked', as_index = False).count())

print("null values in each column: ")
print(ds.isnull().sum())

print('\n')
print(ds.Age.agg(['mean', 'median']))
print(ds.Embarked.value_counts())

ds.fillna({'Age': 28, 'Embarked': 'S'},inplace = True)

y = ds.Survived
x = ds.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'],axis = 1)
print(x.head())

scaler = StandardScaler()
x.iloc[:,[2,3,4,5]] = scaler.fit_transform(x.iloc[:,[2,3,4,5]])
print(x.head())

x1 = pd.get_dummies(x[['Sex', 'Embarked']])
x = pd.concat((x.drop(['Sex', 'Embarked'], axis = 1), x1), axis = 1)
print(x.head())
print(x.columns)

BW = 0.25

dspclass = ds[['Pclass','Survived']]
ds_pclass = dspclass.groupby('Pclass').Survived.value_counts()

bar0 = [80, 97, 372]
bar1 = [136, 87, 119]

r0 = np.arange(len(bar0))
r1 = [x + BW for x in r0]

plt.bar(r0, bar0, color = 'red', width = BW, label = 'Not Survived')
plt.bar(r1, bar1, color = 'green', width = BW, label = 'Survived')
plt.xlabel('Passenger Class', fontweight = 'bold')
plt.ylabel('Number of Passengers', fontweight = 'bold')
plt.xticks([r + BW for r in range(len(bar0))], ['Class 1', 'Class 2', 'Class 3'])
plt.legend()
plt.show()

data=pd.concat((x.iloc[:,1:10],y),axis=1)
data = pd.melt(data,id_vars="Survived",var_name="features",value_name="values")
plt.figure(figsize=(16,8))
sns.violinplot(x="features",y="values",hue="Survived",data=data,split=True,inner="quart")
plt.xticks(rotation=45)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2)

c_lr = LogisticRegression(C = 0.01, class_weight='balanced')
c_lr.fit(x_train,y_train)
y_pred_lr = c_lr.predict(x_test)

c_rfc = RandomForestClassifier(n_estimators=500, criterion = 'entropy', bootstrap = True,
                               max_depth = 5, random_state = 0)
c_rfc.fit(x_train, y_train)
y_pred_rfc = c_rfc.predict(x_test)

c_svm = SVC(gamma = 'scale', random_state = 0)
c_svm.fit(x_train, y_train)
y_pred_svm = c_svm.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nAccuracy of LogisticRegressor: \n", acc_lr)
print('\nConfusion Matrix of LogisticRegressor: \n', cm_lr)

acc_rfc = accuracy_score(y_test, y_pred_rfc)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print("\nAccuracy of RandomForestClassifier: \n", acc_rfc)
print('\nConfusion Matrix of RandomForestClassifier: \n', cm_rfc)

acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\nAccuracy of Support Vector Machine: \n", acc_svm)
print('\nConfusion Matrix of Support Vector Machine: \n', cm_svm)

print("Would YOU be able to survive the sinking of the RMS Titanic? Enter details here and find out: ")

p_name = input("Enter name: ")
p_age = int(input("Enter Age: "))

details = [p_pclass, p_age, p_sib, p_par, p_fare, p_sex, p_emb]