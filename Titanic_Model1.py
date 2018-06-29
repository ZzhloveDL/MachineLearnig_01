#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:36:12 2018

@author: zhangzihao
"""
# kaggle Titanic Survival Prediction Competition
import pandas as pd
import numpy as np
import datetime
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

#read the data and start
start = datetime.datetime.now()
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
train_dataset.info()
test_dataset.info()
def prepare_train(data):
    train_dataset = data
    df = pd.DataFrame(train_dataset)
    drop_c = ['PassengerId','Ticket','Embarked']
    df.drop(drop_c,axis=1,inplace=True)
    df['Age'].fillna(df['Age'].median(),inplace=True)
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    df['Family_size']=df['SibSp']+df['Parch']+1
    df['IsAlone']=1
    df['IsAlone'].loc[df['Family_size'] > 1] = 0
    df['Title']=df['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]
    df['FareRank']=pd.qcut(df['Fare'],5)
    df['AgeRank']=pd.cut(df['Age'],6)
    stat_min =10
    title_names = (df['Title'].value_counts()<stat_min)
    df['Title']=df['Title'].apply(lambda x:'Misc' if title_names.loc[x] ==True else x) 
    print("-"*30)
    
    #dummy feature
    dummy_Cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
    dummy_Sex = pd.get_dummies(df['Sex'],prefix='Sex')
    dummy_Pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')
    dummy_Title = pd.get_dummies(df['Title'],prefix='Title')
    df = pd.concat([df,dummy_Cabin,dummy_Sex,dummy_Pclass,dummy_Title],axis=1)
    #Encode the feature
    label = LabelEncoder()
    df['FareRank'] = label.fit_transform(df['FareRank'])
    df['AgeRank'] = label.fit_transform(df['AgeRank'])
    #drop useless colomn and split into x,y
    drop2 = ['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Cabin','Title']
    df.drop(drop2,axis=1,inplace=True)
    coef_column = list(df.columns)[1:]
    #show the prepared train data info
    print('Prepared TrainData')
    df.info()
    train = df.values
    train = train.astype(np.float32)
    trainx = train[:,1:]
    trainy = train[:,0]
    print('trainx_shape',trainx.shape)
    print('trainy_shape',trainy.shape)
    return trainx,trainy,coef_column

def prepare_test(data):
    df = pd.DataFrame(data)
    drop_c = ['PassengerId','Ticket','Embarked']
    df.drop(drop_c,axis=1,inplace=True)
    df['Age'].fillna(df['Age'].median(),inplace=True)
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    df['Family_size']=df['SibSp']+df['Parch']+1
    df['IsAlone']=1
    df['IsAlone'].loc[df['Family_size'] > 1] = 0
    df['Title']=df['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]
    df['FareRank']=pd.qcut(df['Fare'],5)
    df['AgeRank']=pd.cut(df['Age'],6)
    stat_min =10
    title_names = (df['Title'].value_counts()<stat_min)
    df['Title']=df['Title'].apply(lambda x:'Misc' if title_names.loc[x] ==True else x) 
    print("-"*30)
    
    #dummy feature
    dummy_Cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
    dummy_Sex = pd.get_dummies(df['Sex'],prefix='Sex')
    dummy_Pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')
    dummy_Title = pd.get_dummies(df['Title'],prefix='Title')
    df = pd.concat([df,dummy_Cabin,dummy_Sex,dummy_Pclass,dummy_Title],axis=1)
    #Encode the feature
    label = LabelEncoder()
    df['FareRank'] = label.fit_transform(df['FareRank'])
    df['AgeRank'] = label.fit_transform(df['AgeRank'])
    #drop useless colomn and split into x,y
    drop2 = ['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Cabin','Title']
    df.drop(drop2,axis=1,inplace=True)
    #show the info of prepared dataset
    print('Prepared TestData')
    df.info()
    test = df.values
    testx = test.astype(np.float32)
    print('testx_shape',testx.shape)
    return testx

X,Y,coef_column = prepare_train(train_dataset)
x = prepare_test(test_dataset)

#set the solver 1 Linear Logistic Regression
print('Result:')
print("-"*30)
clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#b_clf1 = ensemble.BaggingRegressor(clf1,n_estimators=20,max_samples=0.8,n_jobs=-1)
clf1.fit(X,Y)
pred1 =clf1.predict(x)
print('Logistic Regression CrossValidation',cross_validation.cross_val_score(clf1,X,Y,cv=5))
result1 = pd.DataFrame({"LR":pred1})
#result.to_csv('result_2LinearRegression.csv')
#check the score and coef_
score = clf1.score(X,Y)
coef = list(clf1.coef_.T)
print('bagging_score',score)
coef = pd.DataFrame({"columns":coef_column,"coef":coef})
print(coef)
print("-"*30)

#set solver 2 Support Vector Machine
clf2 = SVC()
clf2.fit(X,Y)
pred2 = clf2.predict(x)
result2 = pd.DataFrame({"SVR":pred2})
score2 = clf2.score(X,Y)
print('score2SVC',score2)
#print('SVC result',pred2[:8])
#result2.to_csv('SVC0.836.csv')
print("-"*30)

#set solver3 Random Forest
clf3 = ensemble.RandomForestClassifier(max_depth=20, random_state=0)
clf3.fit(X,Y)
pred3 = clf3.predict(x)
result3 = pd.DataFrame({"RF":pred3})
score3 = clf3.score(X,Y)
print('scoreRF',score3)
print('Random Forest CrossValidation',cross_validation.cross_val_score(clf3,X,Y,cv=5))
print("-"*30)

#set solver 4 Gradient Boost
clf4 = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=10, random_state=0)
clf4.fit(X,Y)
pred4 = clf4.predict(x)
result4 = pd.DataFrame({"GB":pred4})
#result4.to_csv('GB.csv')
score4 = clf4.score(X,Y)
print('scoreGB',score4)
print('Gradient Boosting CrossValidation',cross_validation.cross_val_score(clf4,X,Y,cv=5))
cal1 = clf4.feature_importances_.T
cal1 = list(cal1)
cal2 = pd.DataFrame({"columns":coef_column,"score":cal1})
print('Gradient Boosting Score:',score4)
print(cal2)
print("-"*30)

#set solver 5 XGBoost
clf5 = xgb.XGBClassifier(gamma = 10, max_depth = 200, n_estimaters = 1000)
clf5.fit(X,Y)
result5 = pd.DataFrame({"XGB":clf5.predict(x)})
score =clf5.score(X,Y)
print('scoreXGB',score)
#result5.to_csv('XGB0.836.csv')

#set solver 6 knn
clf6 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=18, p=2,
           weights='uniform')
clf6.fit(X,Y)
result6 = pd.DataFrame({"KNN":clf6.predict(x)})
score =clf6.score(X,Y)
print('scoreKNN',score)

#ensemble the result

ensemble = pd.concat([result1,result2,result3,result4,result5,result6],axis=1)
#ensemble.to_csv('ensemble_2.csv')
end = datetime.datetime.now()
print('running_time',end-start)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"Training samples")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"TrainSet")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"TestSet")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf5, u"LearningCurves", X, Y)










