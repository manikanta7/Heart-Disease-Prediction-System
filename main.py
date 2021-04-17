import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

column = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age','chd']
data.columns=column
data.head()
data.describe()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['famhist']=encoder.fit_transform(data['famhist'])
data['chd']=encoder.fit_transform(data['chd'])

data.head(5)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range =(0,100))

data['sbp'] = scale.fit_transform(data['sbp'].values.reshape(-1,1))
data.head()
data.describe()
data.head(50).plot(kind='area',figsize=(10,5))

from sklearn.model_selection import train_test_split
col = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age']
X_train, X_test, y_train, y_test = train_test_split(data[col], data['chd'], test_size=0.2, random_state=1234)
sns.set()
sns.heatmap(X_train.head(10),robust = True)

X_all = data[col]
y_all = data['chd']

from sklearn import svm
svm_clf = svm.SVC(kernel ='linear')
svm_clf.fit(X_train,y_train)
y_pred_svm =svm_clf.predict(X_test)
y_pred_svm
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm
from sklearn.metrics import accuracy_score
svm_result = accuracy_score(y_test,y_pred_svm)
print('SVM')
print("Accuracy :",svm_result*100)
recall_svm = cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])
precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])
recall_svm,precision_svm
##############################################
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors =5,n_jobs = -1,leaf_size = 60,algorithm='brute')
knn_clf.fit(X_train,y_train)

y_pred_knn = knn_clf.predict(X_test)
y_pred_knn
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_knn
knn_result = accuracy_score(y_test,y_pred_knn)
knn_result
###########################

#######################
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
ann_clf = MLPClassifier()

parameters = {'solver': ['lbfgs'],
             'alpha':[1e-4],
             'hidden_layer_sizes':(9,14,14,2),   # 9 input, 14-14 neuron in 2 layers,1 output layer
             'random_state': [1]}

 
acc_scorer = make_scorer(accuracy_score)


grid_obj = GridSearchCV(ann_clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)


ann_clf = grid_obj.best_estimator_


ann_clf.fit(X_train, y_train)

y_pred_ann = ann_clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
cm_ann

ann_result = accuracy_score(y_test,y_pred_ann)
ann_result

recall_ann = cm_ann[0][0]/(cm_ann[0][0] + cm_ann[0][1])
precision_ann = cm_ann[0][0]/(cm_ann[0][0]+cm_ann[1][1])
recall_ann,precision_ann

######################

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 9))


#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm_annk = confusion_matrix(y_test, y_pred)
results ={'Accuracy': [svm_result*100,ann_result*100],
          'Recall': [recall_svm*100,recall_ann*100],
          'Precision': [precision_svm*100,precision_ann*100]}
index = ['SVM','ANN']
results =pd.DataFrame(results,index=index)
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(9,9)).get_figure()
fig.savefig('Final Result.png')
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(6,6),ylim=[50,100]).get_figure()
fig.savefig('image.png')
results.plot(subplots=True,kind ='bar',figsize=(4,10))
#############################




age = int(input("What is your age? "))
total_cholstral = int(input("What is your cholastral? "))
heartrate = int(input("What is your heartrate? "))
restedbps = int(input("What is your restbps? "))
yersofsmoke = int(input("What is your smoke? "))
cigerateperday = int(input("What is your perday? "))
exercise_st = int(input("What is your excersice induced exang? "))
exbp = int(input("What is your exbp? "))
maxhrt = int(input("What is your maxhrt? "))
mets = int(input("What is your mets? "))
count=0;
if age>30:
    count=count+1;
if total_cholstral>250:
    count=count+1
if heartrate>85:
    count=count+1
if restedbps>120:
    count=count+1
if yersofsmoke>2:
    count=count+1
if cigerateperday>3:
    count=count+1
if exercise_st>3:
    count=count+1
if exbp>125:
    count=count+1
if maxhrt>90:
    count=count+1
if maxhrt<60:
    count=count+1
if count>=5:
    print ('Chance To Heart Attack')
if count<5:
    print ('No Chance To Heart Attack')
