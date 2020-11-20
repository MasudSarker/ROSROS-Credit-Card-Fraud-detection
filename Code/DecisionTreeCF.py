# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:51:10 2020

@author: masud
"""
#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support






# Import dataset
df_train=pd.read_csv('creditcard.csv')


### Credit Card Fraud and Non-Fraud ration with graph
target_count = df_train['Class'].value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#target_count.plot(kind='bar', title='Count (target)');

## End of Class ratio

# Here we use Robust Scaler technique for feature scalling
# Scale "Time" and "Amount"



df_train['scaled_amount'] = RobustScaler().fit_transform(df_train['Amount'].values.reshape(-1,1))
df_train['scaled_time'] = RobustScaler().fit_transform(df_train['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df_train.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()

# Define the prep_data function to extrac features 
def prep_data(df):
    X = df.drop(['Class'],axis=1, inplace=False)  
    X = np.array(X).astype(np.float)
    y = df[['Class']]  
    y = np.array(y).astype(np.float)
    return X,y

# Create X and y from the prep_data function 
X, y = prep_data(df_scaled)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(X_train.shape)

# ****** LogisticRegression Accuration test
##model = LogisticRegression()
##model.fit(X_train, y_train)
##y_pred = model.predict(X_test)

##accuracy = accuracy_score(y_test, y_pred)
##print("Accuracy: %.2f%%" % (accuracy * 100.0))
# End of accuracy test


# Random UnderSampling
undersam = RandomOverSampler()
# resample the training data
X_undersam, y_undersam = undersam.fit_sample(X_train,y_train)

#After resampling again accuracy count
model = DecisionTreeClassifier()
model.fit(X_undersam, y_undersam)
y_pred_under = model.predict(X_test)
print(X_test.shape)

print('Overall Oversampling:',classification_report(y_test, model.predict(X_test)))
accuracy = accuracy_score(y_test, y_pred_under)

#print("============ Decision Tree Classifier ================%")
#print("Accuracy After RandomUnderSampler: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test, y_pred_under)
print("Accuracy After ROC: %.2f%%" % (roc_auc * 100.0))
#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_under)
pre_scor= precision_score(y_test, y_pred_under)
re_scor = recall_score(y_test, y_pred_under)
f1_scor = f1_score(y_test, y_pred_under)
##print("\n ROC AUC Score:  %.2f%%" % (roc_auc * 100.0))
#print("Precision Score:  %.2f%%" % (pre_scor * 100.0))
#print("Recall Score:  %.2f%%" % (re_scor * 100.0))
#print('F1-Measure: %.2f%%' % (f1_scor * 100.0))

############################################
# Class count
# Define the prep_data function to extrac features 
def prep_data1(df):
    X1 = df[:, :-1]
    y1 = df[:, -1] 
    return X1,y1

# Create X and y from the prep_data function 
X, y = prep_data(df_scaled)
y= y.astype(np.int64)
#print(Counter(y))

df_ar_x = pd.DataFrame(X)
df_ar_y = pd.DataFrame(y)
df_xy=pd.concat([df_ar_x,df_ar_y],axis=1)
print('Before Split:',df_xy.shape)


#var i=0
i=0
two_split = np.array_split(df_xy, 2)

#****************** Slit-1 ****************#
data1 = np.array(two_split[0]).astype(np.float)
#data2 = np.array(two_split[1]).astype(np.float)
print('After Split:',data1.shape)
X1 , y1 = prep_data1(data1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=10)
print('X_Test Ratio:',X_test1.shape, 'Y-Ratio', y_test1.shape)
#----------
df1 = pd.DataFrame(y1, columns = ['Class'])
target_count1 = df1['Class'].value_counts()
print('Class-1 0:', target_count1[0])
print('Class-1 1:', target_count1[1])
print('Proportion-1:', round(target_count1[0] / target_count1[1], 0), ': 1')
#-------------
# Random UnderSampling
#RandomUnderSampler
#DecisionTreeClassifier
over1 = RandomOverSampler()
# resample the training data
X_over1, y_over1 = over1.fit_sample(X_train1,y_train1)
print(X_over1.shape)
print(X_over1.shape)

#----------
#df2 = pd.DataFrame(y_over1, columns = ['Class'])
#target_count2 = df2['Class'].value_counts()
#print('Class-2 0:', target_count2[0])
#print('Class-2 1:', target_count2[1])
#print('After OverSampling Proportion:', round(target_count2[0] / target_count2[1], 0), ': 1')

#-------------
#After resampling again accuracy count
model1 = DecisionTreeClassifier()
model1.fit(X_over1, y_over1)
y_pred_over1 = model1.predict(X_test1)
print('Random Oversampling:',classification_report(y_test1, model1.predict(X_test1)))

#****************** Slit-2 ****************#
#data1 = np.array(two_split[0]).astype(np.float)
data2 = np.array(two_split[1]).astype(np.float)
print('After Split -2:',data2.shape)
X2 , y2 = prep_data1(data2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=10)
print('X_Test Ratio:',X_test2.shape, 'Y-Ratio', y_test2.shape)
#----------
df2 = pd.DataFrame(y2, columns = ['Class'])
target_count2 = df2['Class'].value_counts()
print('Class-2 0:', target_count2[0])
print('Class-2 1:', target_count2[1])
print('Proportion-2:', round(target_count2[0] / target_count2[1], 0), ': 1')
#-------------
# Random UnderSampling
#RandomUnderSampler
#DecisionTreeClassifier
over2 = RandomOverSampler()
# resample the training data
X_over2, y_over2 = over2.fit_sample(X_train2,y_train2)
print(X_over2.shape)
print(X_over2.shape)

#----------
#df2 = pd.DataFrame(y_over1, columns = ['Class'])
#target_count2 = df2['Class'].value_counts()
#print('Class-2 0:', target_count2[0])
#print('Class-2 1:', target_count2[1])
#print('After OverSampling Proportion:', round(target_count2[0] / target_count2[1], 0), ': 1')

#-------------
#After resampling again accuracy count
model2 = DecisionTreeClassifier()
model2.fit(X_over2, y_over2)
y_pred_over2 = model2.predict(X_test2)
print('Random UnderSampling:',classification_report(y_test2, model2.predict(X_test2)))


def classifaction_report_csv(report):
    cd=0
    i=0
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-2]: 
        i=i+1
        row = {}
        if i==5:
            row_data = line.split('      ')
            print(row_data)
            row['class'] = row_data[0]
            print('Test',row_data[1])
            cd = row_data[1]
            row['precision'] = row_data[1]
            row['recall'] = row_data[2]
            row['f1_score'] = row_data[3]
           # row['support'] = row_data[4]
          # print(cd)
       # return cd
            
        
   # dataframe = pd.DataFrame.from_dict(report_data)
    #dataframe.to_csv('report.csv', index = False)
def classification_report_rst(report):
    i=0
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-2]:
        i=i+1
        row = {}
        i=i+1
        if i == 5:
            row_data = line.split('      ')
            print(row_data)
            precision = row_data[1]
            print('Precision:',precision)
            recall = row_data[2]
            print(recall)
            f1_score = row_data[3]   
            print(f1_score)
#    return precision,recall,f1_score

#abc = classification_report_rst(y_test1)
#call the classification_report first and then our new function

report = classification_report(y_test2, model2.predict(X_test2))
#p2,r2,f12 = classification_report_rst(report)
abcd = classifaction_report_csv(report)
print('Restul-2',abcd)
#p2 = classification_report_rst(report)
#print('Split-2 Precision:',p2,'Recall:',r2,'F1-Measure:',f12)
