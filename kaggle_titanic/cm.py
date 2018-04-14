#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:59:17 2018

@author: Chu Mai

"""

import pandas as pd
import matplotlib.pyplot as plt
rcParams = { 'figure.figsize': (4.0,3.0)}
plt.rcParams.update(rcParams)
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from gplearn.genetic import SymbolicRegressor

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#%%
train = pd.read_csv('train.csv')

train.info()

train.set_index('PassengerId', inplace = True)

train.head()

#%%
test = pd.read_csv('test.csv')
test.set_index('PassengerId', inplace = True)
test.head()

#%% join df and test, to have all values for training
df = pd.concat( [train, test], axis = 0, join = 'outer' )
df.tail()

#%% label encoding

df['Sex'].unique()

label_sex = preprocessing.LabelEncoder()
label_sex.fit( df['Sex'] )
#label_sex.inverse_transform( [0,1,1,0] )
df['Sex label'] = label_sex.transform( df['Sex'] )

#%%

df['Embarked'].unique()
df.loc[df['Embarked'].isnull(), 'Embarked'] = 'unknown'
label_embarked = preprocessing.LabelEncoder()
label_embarked.fit( df['Embarked'].unique() )

label_embarked.transform( df['Embarked'].unique() )
#label_embarked.inverse_transform( [0,1,2,3]  )

df['Embarked label'] = label_embarked.transform( df['Embarked'] )

#%%
df[ df['Age'].isnull()] # there are a lot of missing ages

df[ df['Age'].isnull()].describe()

df[ df['Age'].isnull()].median()

df.describe()

# conclusion: passengers with unknown ages travelling at median fare around half the overall median fare, less parent or children than overal mean, more siblings than overall mean, lower class than overall mean, less likely to survive than overal mean 

df[ df['Age'].isnull()]['Survived'].value_counts() # 125/177 did not survive

df[ df['Age'].isnull()]['Pclass'].value_counts() # majority in 3rd class (136/177), fewer in 1st class (30/177), 2nd class (11/177)
df[ (df['Age'].isnull() ) & (df['Pclass']==1)]

age_ranges = [[0.0, 10.0],[10.0, 15.0], [15.0, 20.0], [20.0, 30.0], [30.0, 60.], [60.0, 120.] ]

for [a,b] in age_ranges:
    df.loc[ (df['Age'] >= a) & (df['Age'] < b), 'Age range' ] = '%d : %d'%(a,b)

df.loc[ df['Age'].isnull(), 'Age range' ] = 'unknown'

label_age = preprocessing.LabelEncoder()
label_age.fit(df['Age range'])

df['Age label'] = label_age.transform( df['Age range'] )

#label_age.inverse_transform( df['Age label'].unique() )

#%%
# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(df["Age"][df["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = df["Age"].median()
    age_pred = df["Age"][((df['SibSp'] == df.loc[i,"SibSp"]) & (df['Parch'] == df.loc[i,"Parch"]) & (df['Pclass'] == df.loc[i,"Pclass"]))].median()
    if not np.isnan(age_pred) :
        df.loc[i, 'Age'] = age_pred
    else :
        df.loc[i, 'Age'] = age_med

#%%

#find title of a person (Mr, Mrs, Dr,...)
df['Title'] = df['Name'].str.replace(' ','').apply(lambda x: x[x.index(',')+1:x.index('.')])
df['Title'].unique().tolist()
df['Title'].value_counts()
df['Title'].isnull().any()
df['Title'] = df['Title'].str.replace( 'Mlle', 'Miss')
df['Title'] = df['Title'].str.replace( 'Ms', 'Miss')
df['Title'] = df['Title'].str.replace( 'Mme', 'Mrs')
df.loc[~df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'] ), 'Title'] = 'Rare'

label_title = preprocessing.LabelEncoder()
label_title.fit( df['Title'].unique().tolist() )
df['Title label'] = label_title.transform( df['Title'] )
#%% cabin where the passenger was located

df.loc[~df['Cabin'].isnull()]

df.loc[~df['Cabin'].isnull()]['Survived'].value_counts() # 136 survived, 68 not

df.loc[ (~df['Cabin'].isnull() ) & (df['Survived'] == 1)]

df['Cabin'].str[0].unique()

df['Cabin'].fillna('unknown',inplace = True)

label_cabin = preprocessing.LabelEncoder()
label_cabin.fit( df['Cabin'].str[0].unique().tolist() )

df['Cabin label'] = label_cabin.transform( df['Cabin'].str[0] )


#%% fill in missing Fare

df.loc[df['Fare'].isnull()]

df.loc[ (df['Pclass']==3) & (df['Embarked']== 'S') ]['Fare'].median()

df.loc[ df['Fare'].isnull() ] = 8.05

#%% family size 

# Create a family size descriptor from SibSp and Parch
df["Family size"] = df["SibSp"] + df["Parch"] + 1

# Create new feature of family size
df.loc[df["Family size"] == 1, 'FSize'] = 'Single'
df.loc[df["Family size"] == 2, 'FSize'] = 'SmallF'
df.loc[ (df["Family size"] >=3) & (df["Family size"]<=4), 'FSize'] = 'MediumF'
df.loc[df["Family size"] >= 5, 'FSize'] = 'LargeF'

label_fsize = preprocessing.LabelEncoder()
label_fsize.fit( df['FSize'].unique().tolist() )

df['FSize label'] = label_fsize.transform( df['FSize'] )

#%%
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

#Ticket = []
#for i in list(df["Ticket"]):
#    if not i.isdigit() :
#        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
#    else:
#        Ticket.append("X")
#
#df["Ticket label"] = Ticket
#df["Ticket label"].unique().tolist()
#df["Ticket"].head()

#%%
plt.figure()
sns.heatmap(df.corr(),xticklabels= df.corr().columns.values,yticklabels=df.corr().columns.values, cmap = cm.PiYG, center = 0., annot = True, fmt = '.2f' )
plt.savefig("figures/corr.png" , bbox_inches = 'tight', dpi = 500)

#%%
#features = ['Pclass', 'Fare', 'Sex label', 'Age label', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

features = ['Pclass', 'Fare', 'Sex label', 'Age', 'FSize label', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

df[ features ].isnull().any()

X_train = df.loc[ train.index, features ].values
Y_train = df.loc[ train.index, ['Survived'] ].values.ravel()

X_test = df.loc[ test.index, features ].values
Y_test = df.loc[ test.index, ['Survived']].values.ravel()

#%%
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=20)

#%%
# Modeling step Test differents algorithms 
random_state = 1
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

g.figure.savefig('figures/cv_scores_classifiers.png', bbox_inches='tight')


#%% TUNING PARAMETERS OF CLASSIFIERS 
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

#%%
#ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5],
              "min_samples_split": [2, 3],
              "min_samples_leaf": [1, 2, 3],
              "bootstrap": [False],
              "n_estimators" :[10, 50, 100],
              "criterion": ["gini", "entropy"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_

#%%
# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5],
              "min_samples_split": [2, 3, 5],
              "min_samples_leaf": [1, 2,  3, 5],
              "bootstrap": [False],
              "n_estimators" :[10, 100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_

#%%
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [10, 100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [2, 4, 8],
              'min_samples_leaf': [1, 5, 10, 100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_

#%%
### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_
#%%
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.2, 1.0, 6)):
    """Generate a simple plot of the test and training learning curve"""
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", markersize = 3.0,
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", markersize = 3.0,
             label="Cross-validation score")

    plt.legend(loc="lower right")
    plt.ylim([0.7, 1.0])
    return fig

#%%
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g.savefig('figures/convergence_RF.png', bbox_inches='tight')


g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g.savefig('figures/convergence_ExtraTrees.png', bbox_inches='tight')


g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g.savefig('figures/convergence_SVM.png', bbox_inches='tight')


g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g.savefig('figures/convergence_AdaBoost.png', bbox_inches='tight')


g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)
g.savefig('figures/convergence_GradientBoost.png', bbox_inches='tight')
#%%
test_Survived_RFC = pd.Series(RFC_best.predict(X_test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(X_test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(X_test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(X_test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(X_test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)
g.figure.savefig('figures/correlation_prediction_classifiers.png', bbox_inches='tight')


#%%
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

#votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

#votingC = VotingClassifier(estimators=[ ('extc', ExtC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

#votingC = VotingClassifier(estimators=[('svc', SVMC_best), ('adac',ada_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

#%%

df.loc[ test.index, 'Survived' ] = votingC.predict( df.loc[test.index, features] )

#%% 
ref_gp = pd.read_csv('submissions/gaussian_process.csv',index_col = 0)
ref_gp.columns = ['Genetic programming']

print('correlation of prediction with Genetic programming solution')
print(pd.concat( [df.loc[ test.index, 'Survived' ] , ref_gp], axis = 1 ).corr())

#%%

df.loc[test.index, 'Survived'].astype(int).reset_index().to_csv('submissions/voting_classifier_with_fsize.csv', sep =',', index =False)





