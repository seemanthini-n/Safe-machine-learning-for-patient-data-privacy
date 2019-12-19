# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:08:38 2019

@author: Moorthy
consolidated codebase
"""
#######################exploratory data analysis###################################
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import neighbors,linear_model,svm #for knn, logistic regression, svc
from sklearn.neural_network import MLPClassifier # for lbfgs
from sklearn.naive_bayes import GaussianNB #for naive bayes
from sklearn.tree import DecisionTreeClassifier #for decision tree
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier #for bagged decision tree, random forest, extra trees,adaboost,gradient boosting, voting classifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split ,KFold, cross_val_score,validation_curve,cross_val_predict
import matplotlib.pyplot as plt
%matplotlib qt
#input data file
#bcdata= pd.read_csv("E:\\NCI\\Sem2\\ADM\\project\\data\\breast-cancer-wisconsin.data")
bcdata= pd.read_csv("E:\\NCI\\Sem2\\ADM\\project\\data\\encrypted_data_breast_cancer.csv")
#replace ? with NaN and fills with last valid value 
bcdata=bcdata.replace('?',np.NaN)
#remove for encrypted
#bcdata['Bare_Nuclei'] = bcdata['Bare_Nuclei'].fillna(method='ffill')
#drop id column remove for encrypted
#bcdata.drop(['Sample_code_number'], 1, inplace=True)
#Correlation Matrix with Heatmap 
sns.set(style='white', color_codes=True)
plt.figure(figsize=(18, 18))
sns.heatmap(bcdata.astype(float).corr(), linewidths=0.5, square=True, linecolor='white', annot=True)
plt.show()

#independent and dependent variable list
bcdata_features=bcdata.iloc[:,1:9]
bcdata_class=bcdata.iloc[:,-1]
# =============================================================================
# bcdata_features=np.array(bcdata.drop(['Class'],1))
# bcdata_class=np.array(bcdata['Class'])
# =============================================================================
#######################bcdata classification using knn###################################
knnclf=neighbors.KNeighborsClassifier()
knnkFold=KFold(n_splits=10)
knncrossvalresults = cross_val_score(knnclf, bcdata_features, bcdata_class, cv=knnkFold)
knnpredicted = cross_val_predict(knnclf, bcdata_features, bcdata_class, cv=knnkFold)
print ("KNN accuracy: ",knncrossvalresults.mean()*100, "%")
print(classification_report(bcdata_class,knnpredicted))
knnf1 = cross_val_score(knnclf, bcdata_features, bcdata_class, cv=knnkFold, scoring='f1_weighted')
print("f1:",knnf1.mean())
#######################bcdata classification using logistic regression###################################
#liblinear good choice for small datasets
bcdatalogregr=linear_model.LogisticRegression(solver='liblinear')
#split change from 5 to 10 increases accuracy by 2%
bckFold=KFold(n_splits=10)
logregrcrossvalresults = cross_val_score(bcdatalogregr, bcdata_features, bcdata_class, cv=bckFold)
#new
predicted = cross_val_predict(bcdatalogregr, bcdata_features, bcdata_class, cv=bckFold)
print ("logistic regression accuracy: ",logregrcrossvalresults.mean()*100, "%")
#new
print(classification_report(bcdata_class,predicted))
logf1=cross_val_score(bcdatalogregr, bcdata_features, bcdata_class, cv=bckFold)
print("f1:",logf1.mean())
#######################bcdata classification using SVC###################################
#gamma='auto' to remove warning message
bcSVC=svm.SVC(gamma='auto')
svckFold=KFold(n_splits=10)
svccrossvalresults = cross_val_score(bcSVC, bcdata_features, bcdata_class, cv=svckFold)
print ("SVC accuracy: ",svccrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(bcSVC, bcdata_features, bcdata_class, cv=svckFold)
#new
print(classification_report(bcdata_class,predicted))
svcf1=cross_val_score(bcSVC, bcdata_features, bcdata_class, cv=svckFold)
print("f1:",svcf1.mean())
#######################bcdata classification using lbfgs###################################
lbfgsclf = MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(11,))
bckfold = KFold(n_splits=10)
lbfgscrossvalresults = cross_val_score(lbfgsclf, bcdata_features, bcdata_class, cv=bckfold)
print ("lbfgs accuracy: ",lbfgscrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(lbfgsclf, bcdata_features, bcdata_class, cv=bckfold)
#new
print(classification_report(bcdata_class,predicted))
lbfgsf1=cross_val_score(lbfgsclf, bcdata_features, bcdata_class, cv=bckfold)
print("f1:",lbfgsf1.mean())
#######################bcdata classification using naive bayes###################################
nbclf = GaussianNB()
nbkfold = KFold(n_splits=10)
nbcrossvalresults = cross_val_score(nbclf, bcdata_features, bcdata_class, cv=nbkfold)
print ("naive bayes accuracy: ",nbcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(nbclf, bcdata_features, bcdata_class, cv=nbkfold)
#new
print(classification_report(bcdata_class,predicted))
nvf1=cross_val_score(nbclf, bcdata_features, bcdata_class, cv=nbkfold)
print("nvf1:",nvf1.mean())
#######################bcdata classification using decision tree###################################
treeclf = DecisionTreeClassifier()
treekfold = KFold(n_splits=10)
dtcrossvalresults = cross_val_score(treeclf, bcdata_features, bcdata_class, cv=treekfold)
print ("decision tree: ",dtcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(treeclf, bcdata_features, bcdata_class, cv=treekfold)
#new
print(classification_report(bcdata_class,predicted))
dtf1=cross_val_score(treeclf, bcdata_features, bcdata_class, cv=treekfold)
print("f1:",dtf1.mean())
#######################bcdata classification using Bagged Decision Tree###################################
dtkfold = KFold(n_splits=10, random_state=7)
dtclf = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=dtclf, n_estimators=100, random_state=7)
bdtcrossvalresults = cross_val_score(model, bcdata_features, bcdata_class, cv=dtkfold)
print ("Bagged Decision Tree accuracy: ",bdtcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(model, bcdata_features, bcdata_class, cv=dtkfold)
#new
print(classification_report(bcdata_class,predicted))
bdtf1=cross_val_score(model, bcdata_features, bcdata_class, cv=dtkfold)
print("f1:",bdtf1.mean())
#######################bcdata classification using Random forest###################################
rfcclf = RandomForestClassifier(n_estimators=50)
rfkfold = KFold(n_splits=10)
rfcrossvalresults = cross_val_score(rfcclf, bcdata_features, bcdata_class,cv=rfkfold)
print ("random forest accuracy: ",rfcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(rfcclf, bcdata_features, bcdata_class,cv=rfkfold)
#new
print(classification_report(bcdata_class,predicted))
rff1=cross_val_score(rfcclf, bcdata_features, bcdata_class,cv=rfkfold)
print("f1:",rff1.mean())
#######################bcdata classification using extra trees###################################
etkfold = KFold(n_splits=10, random_state=7)
etclf = ExtraTreesClassifier(n_estimators=100, max_features=7)
etcrossvalresults = cross_val_score(etclf, bcdata_features, bcdata_class,cv=etkfold)
print ("extra trees accuracy: ",etcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(etclf, bcdata_features, bcdata_class,cv=etkfold)
#new
print(classification_report(bcdata_class,predicted))
etf1=cross_val_score(etclf, bcdata_features, bcdata_class,cv=etkfold)
print("f1:",etf1.mean())
#######################bcdata classification using AdaBoost###################################
#decision tree default
adakfold = KFold(n_splits=10, random_state=7)
adaclf = AdaBoostClassifier(n_estimators=30, random_state=7)
adacrossvalresults = cross_val_score(adaclf, bcdata_features, bcdata_class,cv=adakfold)
print ("AdaBoost accuracy: ",adacrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(adaclf, bcdata_features, bcdata_class,cv=adakfold)
#new
print(classification_report(bcdata_class,predicted))
adaf1=cross_val_score(adaclf, bcdata_features, bcdata_class,cv=adakfold)
print("f1:",adaf1.mean())
#######################bcdata classification using Stochastic Gradient Boosting###################################
sgdkfold = KFold(n_splits=10, random_state=7)
sgdclf = GradientBoostingClassifier(n_estimators=100, random_state=7)
sgbcrossvalresults = cross_val_score(sgdclf, bcdata_features, bcdata_class,cv=sgdkfold)
print ("Stochastic Gradient Boosting accuracy: ",sgbcrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(sgdclf, bcdata_features, bcdata_class,cv=sgdkfold)
#new
print(classification_report(bcdata_class,predicted))
sgbf1=cross_val_score(sgdclf, bcdata_features, bcdata_class,cv=sgdkfold)
print("f1:",sgbf1.mean())
#######################bcdata classification using Stochastic Voting Ensemble###################################
vekfold = KFold(n_splits=10, random_state=7)
#sub models include logistic regression, decision tree, SVC
estimators = []
model1 = linear_model.LogisticRegression(solver='liblinear')
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = svm.SVC(gamma='auto')
estimators.append(('svm', model3))
# ensemble model
veensemble = VotingClassifier(estimators)
vecrossvalresults = cross_val_score(veensemble, bcdata_features, bcdata_class,cv=vekfold)
print ("Voting Ensemble accuracy: ",vecrossvalresults.mean()*100, "%")
#new
predicted = cross_val_predict(veensemble, bcdata_features, bcdata_class,cv=vekfold)
#new
print(classification_report(bcdata_class,predicted))
svef1=cross_val_score(veensemble, bcdata_features, bcdata_class,cv=vekfold)
print("f1:",svef1.mean())
