# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:51:10 2020

@author: masud
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
model = RandomForestClassifier()
model.fit(X_undersam, y_undersam)
y_pred_under = model.predict(X_test)
print(X_test.shape)

print('Overall Oversampling:',classification_report(y_test, model.predict(X_test)))

#************ Average Result (Precision,Recall, F1-Measure) ***************#
def classifaction_report_rst(report):
    i=0
    lines = report.split('\n')
    for line in lines[2:-2]: 
        i=i+1
        if i==5:
            row_data = line.split('      ')
            return row_data
        
#************** End of Average Result ******************#

report0 = classification_report(y_test, model.predict(X_test))
avgrst0 =  classifaction_report_rst(report0)

accuracy = accuracy_score(y_test, y_pred_under)

#print("============ Decision Tree Classifier ================%")
#print("Accuracy After RandomUnderSampler: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test, y_pred_under)

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
model1 = RandomForestClassifier()
model1.fit(X_over1, y_over1)
y_pred_over1 = model1.predict(X_test1)


print('Random Oversampling:',classification_report(y_test1, model1.predict(X_test1)))

roc_auc1 = roc_auc_score(y_test1, model1.predict(X_test1))


report1 = classification_report(y_test1, model1.predict(X_test1))
avgrst1 =  classifaction_report_rst(report1)
print('Test Result:1',avgrst1)

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
model2 = RandomForestClassifier()
model2.fit(X_over2, y_over2)
y_pred_over2 = model2.predict(X_test2)
print('Random UnderSampling:',classification_report(y_test2, model2.predict(X_test2)))
roc_auc2 = roc_auc_score(y_test2, model2.predict(X_test2))



report = classification_report(y_test2, model2.predict(X_test2))
#p2,r2,f12 = classification_report_rst(report)
avgrst2 =  classifaction_report_rst(report)




print("Accuracy Split-1 ROC: %.2f%%" % (roc_auc1 * 100.0))
print("Accuracy Split-2 ROC: %.2f%%" % (roc_auc2 * 100.0))


f11measure = avgrst1[3][0:4]
f12measure = avgrst2[3][0:4]

print('Whole Precision:',avgrst0[1],'Recall:',avgrst0[2],'F1-Measure:',avgrst0[3])
print('Split-1 Precision:',avgrst1[1],'Recall:',avgrst1[2],'F1-Measure:',avgrst1[3])
print('Split-2 Precision:',avgrst2[1],'Recall:',avgrst2[2],'F1-Measure:',avgrst2[3])
p1 = round(float(avgrst1[1]),2)
p2 = round(float(avgrst2[1]),2)
print('Test-22',p1,'Test21',p2)
r1 = round(float(avgrst1[2]),2)
r2 = round(float(avgrst2[2]),2)
f1 = round(float(f11measure),2)
f2 = round(float(f12measure),2)

print('=============================================================\n\n')
print("== Random OverSampling (ROS) and Our model Result Values ==")

print("ROS ROC-AUC: %.2f%%" % (roc_auc * 100.0))
print("Accuracy Avg ROC: %.2f%%" % ((roc_auc1+roc_auc2)/2 * 100.0))

print('ROS Precision:',avgrst0[1],'ROS Recall:',avgrst0[2],'ROS F1-Measure:',avgrst0[3])
print('Avg Precision:',round(((p1+p2)/2),2) , 'Avg Recall:',round(((r1+r2)/2),2) ,'Avg F-Measure:', round(((f1+f2)/2),2) )


