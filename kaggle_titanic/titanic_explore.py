#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:28:57 2018

@author: chumai
"""

import pandas as pd
import matplotlib.pyplot as plt
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

#%%
df = pd.read_csv('train.csv')

df.info()


df.set_index('PassengerId', inplace = True)

df.head(20)



#%%

df[['Pclass','Ticket', 'Fare','Cabin']]

df.loc[ ~df['Cabin'].isnull(),['Survived', 'Pclass','Ticket', 'Fare','Cabin'] ]

df.loc[ ~df['Cabin'].isnull() & df['Survived']==1,'Survived' ].value_counts()

df.loc[ ~df['Cabin'].isnull() & df['Survived']==1,['Survived', 'Pclass','Ticket', 'Fare','Cabin'] ]
#%%

test = pd.read_csv('test.csv')
test.set_index('PassengerId', inplace = True)
test.head()

#%% join df and test, to have all values for training

df_total = pd.concat( [df, test], axis = 0, join = 'outer' )

#%% label encoding

df['Sex'].unique()

label_sex = preprocessing.LabelEncoder()
label_sex.fit( df_total['Sex'] )
#label_sex.inverse_transform( [0,1,1,0] )
df['Sex label'] = label_sex.transform( df['Sex'] )

test['Sex label'] = label_sex.transform( test['Sex'] )

#%%

df_total['Embarked'].unique()
df_total.loc[df_total['Embarked'].isnull(), 'Embarked'] = 'unknown'
test.loc[test['Embarked'].isnull(), 'Embarked'] = 'unknown'
df.loc[df['Embarked'].isnull(), 'Embarked'] = 'unknown'

label_embarked = preprocessing.LabelEncoder()
label_embarked.fit( df_total['Embarked'].unique() )

label_embarked.transform( df['Embarked'].unique() )

#label_embarked.inverse_transform( [0,1,2,3]  )

df['Embarked label'] = label_embarked.transform( df['Embarked'] )

test['Embarked label'] = label_embarked.transform( test['Embarked'] )
#df.loc[df['Embarked'].isnull(),:]
#df.loc[df['Name'].str.contains('Icard'),:]
#df.loc[df['Name'].str.contains('Stone'),:]

#df['Embarked'] = df['Embarked'].replace('S', 1)
#df['Embarked'] = df['Embarked'].replace('C', 2)
#df['Embarked'] = df['Embarked'].replace('Q', 3)
#
#df.loc[ df['Embarked'].isnull(),'Embarked'] = 1.0 # the most probably

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
    df_total.loc[ (df_total['Age'] >= a) & (df_total['Age'] < b), 'Age range' ] = '%d : %d'%(a,b)

df.loc[ df['Age'].isnull(), 'Age range' ] = 'unknown'
df_total.loc[ df_total['Age'].isnull(), 'Age range' ] = 'unknown'

label_age = preprocessing.LabelEncoder()
label_age.fit(df_total['Age range'])

df['Age label'] = label_age.transform( df['Age range'] )

label_age.inverse_transform( df['Age label'].unique() )
## assume that missing ages are in range 20-30 (i+1 = 4)
#df.loc[ df['Age'].isnull(), 'Age'] = 4

#%%
for [a,b] in age_ranges:
    test.loc[ (test['Age'] >= a) & (test['Age'] < b), 'Age range' ] = '%d : %d'%(a,b)
test.loc[ test['Age'].isnull(), 'Age range' ] = 'unknown'
test['Age label'] = label_age.transform( test['Age range'] )

#%%

df['Age'].isnull().any()

df['Age'].fillna( df_total.loc[df_total['Pclass']==3,'Age' ].median() , inplace = True)
test['Age'].fillna( df_total.loc[df_total['Pclass']==3,'Age' ].median() , inplace = True)

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

for tit in ['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'theCountess', 'Jonkheer', 'Dona']:
    df['Title'] = df['Title'].str.replace( tit, 'Rare')

df_total['Title'] = df_total['Name'].str.replace(' ','').apply(lambda x: x[x.index(',')+1:x.index('.')])
df_total['Title'].value_counts()
df_total['Title'].isnull().any()
df_total['Title'] = df_total['Title'].str.replace( 'Mlle', 'Miss')
df_total['Title'] = df_total['Title'].str.replace( 'Ms', 'Miss')
df_total['Title'] = df_total['Title'].str.replace( 'Mme', 'Mrs')
df_total.loc[~df_total['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'] ), 'Title'] = 'Rare'


test['Title'] = test['Name'].str.replace(' ','').apply(lambda x: x[x.index(',')+1:x.index('.')])


df[ df['Title'].isin(['Capt', 'Col', 'Ms'])]
df[ df['Title'].isin(['theCountess', 'Jonkheer'])]
df[ df['Title'].isin(['Master','Capt', 'Col', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir'])]
   
label_title = preprocessing.LabelEncoder()
label_title.fit( df_total['Title'].unique().tolist() )
df['Title label'] = label_title.transform( df['Title'] )
#for i, j in zip( df['Title'].unique().tolist(), label_title.transform( df['Title'].unique().tolist() ).tolist() ):
#    print i, '-', j
test['Title label'] = label_title.transform( test['Title'] )

#%%
#test['Title'] = test['Name'].str.replace(' ','').apply(lambda x: x[x.index(',')+1:x.index('.')])
#test['Title'].unique().tolist()
#
#test['Title label'] = label_title.transform( test['Title'] )
#%% cabin where the passenger was located

df.loc[~df['Cabin'].isnull()]

df.loc[~df['Cabin'].isnull()]['Survived'].value_counts() # 136 survived, 68 not

df.loc[ (~df['Cabin'].isnull() ) & (df['Survived'] == 1)]

df['Cabin'].str[0].unique()

df['Cabin'].fillna('unknown',inplace = True)
df_total['Cabin'].fillna('unknown',inplace = True)
test['Cabin'].fillna('unknown',inplace = True)

label_cabin = preprocessing.LabelEncoder()
label_cabin.fit( df_total['Cabin'].str[0].unique().tolist() )

df['Cabin label'] = label_cabin.transform( df['Cabin'].str[0] )
test['Cabin label'] = label_cabin.transform( test['Cabin'].str[0] )

#%%
#match = re.match(r"([a-z]+)([0-9]+)", 'foofo21', re.I)
#
#match = re.match(r"([a-z]+)([0-9]+)", '', re.I)
#match = re.match(r"([a-z]+)([0-9]+)", np.nan, re.I)
#if match:
#    items = match.groups()

#%%
    
#df['Cabin'].apply(lambda x: )
#%%
df.head()

df.tail()

df.corr()

#%%
plt.figure()
sns.heatmap(df.corr(),xticklabels= df.corr().columns.values,yticklabels=df.corr().columns.values, cmap = cm.PiYG, center = 0., annot = True, fmt = '.2f' )
plt.savefig("figures/corr.png" , bbox_inches = 'tight', dpi = 500)

#%% random forest classifier

rf = RandomForestClassifier(n_estimators= 10, n_jobs = -1)

#divide into training (80%) and validation sets (20%)

df['Training'] = np.random.rand(len(df)) < 0.8

#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title label']

features = ['Pclass', 'Fare', 'Sex label', 'Age label', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

#features = ['Pclass', 'Fare', 'Sex label', 'Age', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

#features = ['Pclass', 'Fare', 'Sex label', 'Age label', 'Embarked label', 'SibSp', 'Parch']

#features = ['Pclass', 'Fare', 'Sex label', 'Age label', 'Embarked label']

df[features].isnull().any()

#for fea in features:
#    if df[fea].isnull().any():
#        print(fea, ' contains nan values')

#%%
x_train = df.loc[ df['Training'], features ].values
y_train = df.loc[ df['Training'], ['Survived'] ].values.ravel()

x_val = df.loc[ ~df['Training'], features ].values
y_val = df.loc[ ~df['Training'], ['Survived']].values.ravel()

rf.fit(x_train, y_train)

rf.feature_importances_

#%% plot importance of features
features_importances = pd.DataFrame( data = rf.feature_importances_.reshape([-1, len(features) ]) , columns = features)

features_importances.plot.bar()

features_importances.transpose().plot.pie(0)
#%%

#rf.fit(df.loc[ df['Training'], features ], df.loc[ df['Training'], ['Survived'] ])

#%%
# predict class
rf.predict(x_val)

# predict probability
rf.predict_proba(x_val)

plt.figure()
plt.hist(rf.predict_proba(x_val)[:,0])

sns.kdeplot(rf.predict_proba(x_val)[:,0] , bw = 0.05)

# predict log-probability
rf.predict_log_proba(x_val)

# predict accuracy
print('accuracy on training set\n')
rf.score(x_train, y_train)
print('accuracy on validation set\n')
rf.score(x_val, y_val)

#%%
growing_rf = RandomForestClassifier(n_estimators= 10, n_jobs = -1, warm_start = True)
scores = []

fig = plt.figure(figsize=(4,3))
for i in range(20):
    growing_rf.fit( x_train , y_train )
    growing_rf.n_estimators += 10

    scores.append( growing_rf.score( x_val, y_val) )
    
_ = plt.plot( scores, '-r')
plt.ylim([0.7,1.0])
plt.show()

#%% resampling the training sample
growing_rf = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', n_jobs = -1, warm_start = True)
scores = []
#%%
fig = plt.figure(figsize=(4,3))
for i in range(50):
    df['Training'] = np.random.rand(len(df)) < 0.8
    x_train = df.loc[ df['Training'], features ].values
    y_train = df.loc[ df['Training'], ['Survived'] ].values.ravel()
    
    x_val = df.loc[ ~df['Training'], features ].values
    y_val = df.loc[ ~df['Training'], ['Survived']].values.ravel()

    growing_rf.fit( x_train , y_train )
    growing_rf.n_estimators += 100

    scores.append( growing_rf.score( df[features].values, df['Survived'].values.ravel() ) )
#    scores.append( growing_rf.score( x_val, y_val) )
    
_ = plt.plot( scores, '-r')
plt.ylim([0.7,1.0])
plt.show()



#%%
dfval = df.loc[ ~df['Training']]

#dfval[['Pr1','Pr2']]= np.nan
#dfval[['Pr1','Pr2']]= rf.predict_proba(x_val)
#dfval.loc[:,'Pr1']= rf.predict_proba(x_val)[:,0]
#dfval.loc[:,'Pr']= rf.predict_proba(x_val)[:,0]

dfval = pd.concat([dfval, pd.DataFrame(index = dfval.index, columns = [['Pr1', 'Pr2']], data = rf.predict_proba(x_val)) ], axis = 1, join = 'inner' )

dfval['Survival prediction'] = rf.predict(x_val)
dfval.head()

dfval.loc[ dfval[['Pr1','Pr2']].max(axis = 1) < 0.7 ]

dfval.loc[ dfval[['Pr1','Pr2']].max(axis = 1) < 0.7 ][['Survived', 'Survival prediction']]

dfval.loc[ dfval['Survived'] != dfval['Survival prediction'] ][['Survived', 'Survival prediction']]

dfval.loc[ dfval['Survived'] != dfval['Survival prediction'] ]['Survived'].value_counts()


#%% fill in missing Fare
test[ features ].isnull().any()
test.loc[test['Fare'].isnull()]

test.loc[ (test['Pclass']==3) & (test['Embarked']== 'S') ]['Fare'].median()

test.loc[ test['Fare'].isnull() ] = 8.05

#%%
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=4)

# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": list(np.arange(2,7)),
              "min_samples_split": list(np.arange(2,3)),
              "min_samples_leaf": [1],
              "n_estimators" :[10, 50, 100, 700],
              "bootstrap":[True],
              "oob_score":[True],
              "criterion": ["gini","entropy"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(df[ features ].values , df['Survived'].values.ravel())

# best model among candidates
RFC_best = gsRFC.best_estimator_

RFC_best

# Best score
gsRFC.best_score_

RFC_best.fit(df[ features ].values , df['Survived'].values.ravel())

RFC_best.score(df[ features ].values , df['Survived'].values.ravel())

# save with pickle
#import pickle 
#pickle.dump(RFC_best,  open( 'RFC_best.p', "wb" ) )
#a = pickle.load( open( 'RFC_best.p', "rb" ))
#%%
test['Survived'] = RFC_best.predict( test[features].values )

test['Survived'].reset_index().to_csv('submissions/rf_best.csv', sep =',', index =False)

#%% fit again, 
growing_rf = RandomForestClassifier(n_estimators= 10, n_jobs = -1, warm_start = True, oob_score=True,  min_samples_split=2, min_samples_leaf=1, max_features=5, max_depth = None)

#%%
scores = []
fig = plt.figure(figsize=(4,3))
for i in range(10):
    growing_rf.fit( df[ features ].values , df['Survived'].values.ravel() )
    growing_rf.n_estimators += 50

    scores.append(growing_rf.score( df[ features ].values , df['Survived'].values.ravel() ))
    
_ = plt.plot(scores, '-r')
plt.ylim([0.7,1.0])
plt.show()
#%%
growing_rf.feature_importances_

pre = growing_rf.predict( test[ features ].values )

results = pd.DataFrame( index = test.index, data = pre, columns = ['Survived']).reset_index()

results.to_csv('submission_random_forest.csv', sep =',', index =False)


#%% classification after clustering

# clustering 
#join df and test, to have all values for training
df_total = pd.concat( [df, test], axis = 0, join = 'outer' )

#%% clustering without considering survival

#features = ['Fare', 'Sex label', 'Age label', 'Embarked label']
#features = ['Pclass', 'Fare', 'Age', 'Sex label', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

features = ['Pclass', 'Fare', 'Sex label', 'Age label', 'Embarked label', 'Title label', 'Cabin label', 'SibSp', 'Parch']

nclusters = 5

clusters = KMeans(n_clusters= nclusters).fit( df_total[features].values)

#clusters = AgglomerativeClustering(linkage='ward', n_clusters=nclusters)

df['Cluster'] = clusters.predict(df[features])

df['Cluster'].value_counts()

test['Cluster'] = clusters.predict(test[features])

test['Cluster'].value_counts()

test['Cluster'].unique()


#%%  clustering while considering survival
nclusters = 5
clusters = KMeans(n_clusters= nclusters).fit( df[features + ['Survived']].values)
df['Cluster'] = clusters.predict(df[features+ ['Survived']])
df['Cluster'].value_counts()

clf = RandomForestClassifier(criterion = 'entropy')
clf.fit( df[features].values, df['Cluster'].values.ravel() )

test['Cluster'] = clf.predict( test[features].values )

df['Cluster'].value_counts()
test['Cluster'].value_counts()

#%%

for i in test['Cluster'].unique():
#for i in [7]:
    print('cluster %d'%i)
    rf = RandomForestClassifier(n_estimators= 100, criterion ='entropy', n_jobs = -1, min_samples_split=2, min_samples_leaf=1, max_features=5)
    rf.fit( df.loc[ df['Cluster']==i, features ].values , df.loc[ df['Cluster']==i,'Survived'].values.ravel() )

    print(rf.score( df.loc[ df['Cluster']==i, features ].values , df.loc[ df['Cluster']==i,'Survived'].values.ravel() ))
    
#    rf.feature_importances_

    test.loc[ test['Cluster']==i, 'Survived' ] = rf.predict( test.loc[ test['Cluster']==i, features ].values )

#%%
test['Survived'] = test['Survived'].astype('int')
test['Survived'].reset_index().to_csv('submission_classification_clustering_8.csv', sep =',', index =False)

#%% neural network
from sklearn.neural_network import MLPClassifier

for i in test['Cluster'].unique():
    print('cluster %d'%i)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,30), random_state=1)

    clf.fit( df.loc[ df['Cluster']==i, features ].values , df.loc[ df['Cluster']==i,'Survived'].values.ravel()  )
    
    print(clf.score( df.loc[ df['Cluster']==i, features ].values , df.loc[ df['Cluster']==i,'Survived'].values.ravel() ) )
#%% K-neighbors classifier

nneighbors = 5
neigh = KNeighborsClassifier(n_neighbors= nneighbors)

neigh.fit( x_train, y_train ) 

pre = neigh.predict(x_val )

print('accuracy on training set\n')
neigh.score( x_train , y_train )

print('accuracy on validation set\n')
neigh.score( x_val, y_val )
#%%
scores = []

nneighbors_list = range(1,20)
plt.figure()

for i in nneighbors_list:

    neigh = KNeighborsClassifier(n_neighbors= i)
    
    neigh.fit( x_train, y_train ) 
    scores.append(neigh.score( x_val, y_val ))
    
_ = plt.plot(nneighbors_list, scores, '-r')
plt.ylim([0.5,1.0])
plt.show()
#%%
neigh = KNeighborsClassifier(n_neighbors=nneighbors)

neigh.fit( df[ features ].values,df['Survived'].values.ravel() ) 

pre = neigh.predict( test[ features ].values )

print('accuracy on training set\n')
neigh.score( df[ features ].values , df['Survived'].values.ravel() )




#%% gaussian process classifier

gp = GaussianProcessClassifier(n_jobs = -1)

gp.fit( x_train, y_train )

pre = gp.predict(x_val )

print('accuracy on training set\n')
gp.score( x_train , y_train )

print('accuracy on validation set\n')
gp.score( x_val, y_val )

#%% logistic regressor

logreg = linear_model.LogisticRegression(solver = 'lbfgs', penalty = 'l2', C=1.0e2, max_iter = 1000, warm_start = True)

#%%
logreg.fit(x_train, y_train )

pre = logreg.predict(x_val )

print('accuracy on training set\n')
logreg.score( x_train , y_train )

print('accuracy on validation set\n')
logreg.score( x_val, y_val )





#%% genetic programming 

genp = SymbolicRegressor(function_set = ('log', 'mul', 'div', 'add', 'sin', 'cos', 'max', 'min'),  population_size=500,generations=100, stopping_criteria=0.001,p_crossover=0.7, p_subtree_mutation=0.1,p_hoist_mutation=0.05, p_point_mutation=0.1,max_samples=0.9, verbose=1,parsimony_coefficient=0.01, random_state=0)

genp.fit(x_train, y_train)

print(genp._program)

pre = genp.predict(x_val )

print('accuracy on training set\n')
genp.score( x_train , y_train )

print('accuracy on validation set\n')
genp.score( x_val, y_val )
#%%
"""
#%
df2 = cp.deepcopy(df)
df2['Survived'] = df2['Survived'].astype('str').replace('0','-1').astype('int64')
df2.corr()

df.corr().loc['Survived'].plot.bar()

df.plot.scatter(x= 'Pclass', y = 'Survived')
df.plot.scatter(x= 'Fare', y = 'Survived')


df.corr()


df.describe()

df.groupby(by = 'Survived')['Sex'].value_counts()

df.groupby(by = 'Survived')['Pclass'].value_counts()

df.groupby(by = ['Survived','Pclass'])['Sex'].value_counts()

df.groupby(by = 'Pclass')['Survived'].value_counts()

df.groupby(by = 'Pclass')['Sex'].value_counts()

df.groupby(by = 'Pclass')['SibSp'].value_counts()

df.groupby(by = 'Pclass')['Parch'].value_counts()

df.groupby(by = ['Survived','Pclass'])['Age'].mean()

df.groupby(by = ['Survived','Pclass']).describe()


df.groupby(by = 'Pclass')['Sex'].value_counts().plot.pie()
df['Pclass'].value_counts().plot.pie()

df.groupby(by = 'Survived')['Pclass'].value_counts().plot.pie()
df['Pclass'].value_counts().plot.pie()

df['Survived'].value_counts().plot.pie()

#df.pivot()

df[['Survived', 'Ticket', 'Fare', 'Cabin','Embarked']]

df[['Survived', 'Pclass', 'Ticket', 'Fare', 'Cabin','Embarked']]

df.loc[df['Survived'] == 1, ['Pclass', 'Ticket', 'Fare', 'Cabin','Embarked']].sort_values(by= 'Pclass', ascending = True)

df['Ticket'].isnull().any()

df.sort_values(by = 'Fare', ascending= False)


df.sort_values(by = 'Fare', ascending= False)[['Survived', 'Name', 'Fare', 'Cabin','Embarked']]
"""
