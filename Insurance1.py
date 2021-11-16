# -*- coding: utf-8 -*-
"""
Created on Thu 1 Sept 15:51:39 2021

@author: aksha
Data : Insurance_marketing
Algorithm : Decision tree and Random forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)

path = "C:/Users/aksha/Desktop/Phython_spyder/ML/RandomForest/insurance/Dataset/Insurance_Marketing.csv"

df = pd.read_csv(path)
df_copy = df.copy()
df_copy.shape
df.columns
df.shape
df.describe()


fc = ['Marital','Job','Qualification','Default_Premium','Health_Insurance',
      'General_Insurance','Contact','Month',
      'Last_Contact_Day','Previous_Outcome','Client_Subscribed']
len(fc)

#check value_counts for fc variable
for f in fc:
    print('Feature name:{}\n{}'.format(f,df[f].value_counts()))


#convert values with less number to mode or others as per observation
#Marital 3: unknown : mode
df['Marital'] = df['Marital'].replace(3,df['Marital'].mode()[0])
df['Marital'].value_counts()

#Job: ??
#Qualification 5,2:mode
df['Qualification'] = df['Qualification'].replace([2,5],df['Qualification'].mode()[0])
df['Qualification'].value_counts()

#Default_Premium
df.drop(df[df['Default_Premium'] == 2].index, inplace = True,axis=0)
df['Default_Premium'].value_counts()
df.shape
#(41185, 21)

#Health_Insurance
df.drop(df[df['Health_Insurance'] == 1].index, inplace = True,axis=0)
df['Health_Insurance'].value_counts()

df['General_Insurance'].value_counts()

#Contact

#Month
df['Month'].value_counts()
#convert 8 9 5 2 into 'others' : keep it as it is for now 

#other variables seems balanced================================================


#check relation with Client_Subscribed that y variable------------------------------------------
#CHI SQUARE
#ensure that variable are label encoded : in our case variables are label encoded

import scipy.stats as stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency 
fc

#only married people have default premium

#creating a for loop for  chi- square code
for features in fc:
    ct = pd.crosstab(df[features],df['Client_Subscribed'])
    #observer value
    observed_values = ct.values
    
    #expected value
    val = chi2_contingency(ct)
    expected_val = val[3]
    
    #degree of freedom
    no_of_columns = len(df['Client_Subscribed'].unique())
    no_of_rows = len(df[features].unique())
    ddof = (no_of_rows-1)*(no_of_columns-1)
    #print('degree of freedom',ddof)
    
    #variacne ratio : learning rate
    alpha = 0.05
    
    #chi2 formula
    chi_square = sum([(o-e)**2/e for e,o in zip(observed_values,expected_val)])
    
    #chisquare statistics
    chi_sq_statistics = chi_square[0] +chi_square[1]
    #print(chi_sq_statistics)
    
    #critical value
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    critical_value
    
    #if condition stating the relation with target variable
    if chi_sq_statistics >= critical_value:
        print('reject H0,there is a relationship between {} and target variable'.format(features))
    else:
        print('retain H0,there is no relationship between {} and target variable'.format(features))

#DROP
#as per this General_Insurance has no relation with Client_Subscribed so we can drop it


df1 = df.copy()
    

#------------------------------------------------------------------------------
#categorical variable distribution ploting=========================================================
fc
cs = df.groupby('Previous_Outcome').Client_Subscribed.value_counts(normalize=False)
#cs
#cs.unstack()
cs.unstack().plot(kind='bar')

#Numeric variables==================================================
nc=['Age','Last_Contact_Duration','Contacts_During_Campaign','Previous_Contact_Days',
    'Contacts_Before_Campaign','Employement_Rates','Price_Variation','Consumer_Confidence_Index',
    'Insurance_Rate','No_Employees']
len(nc)
len(fc)
df.shape
#=========Work with numerical cols
nc
len(fc)

#Boxplot to check realtion of numnerical with target----------------------------------------------------------------
#y = nc
plt.figure(figsize=(8,5))
sns.boxplot(x='Client_Subscribed',y='Price_Variation',data=df, palette='rainbow')
plt.title("Price_Variation by Client_Subscribed, Insurance")
#-----------------------------------------------------------------------------
for f in nc:
    print('Feature name:{}\n{}'.format(f,df[f].value_counts()))


#Contacts_During_Campaign : more than 6 times can be one value and we can convert it to categorical
#Previous_Contact_Days: can be converted to categorical : called and not called
#Contacts_Before_Campaign:as per the observation  is categroical: more than 2 can be one value
#Employement_Rates
#as per the graph price variation have no impact on CLient Subscription
#as per the graph Consumer_Confidence_Index have no impact on CLient Subscription
#Insurance_Rate: less relation with target: will perfom anova
#No_Employees: less relation with target: will perfom anova

#-need to work on this#######################################################
#Contacts_During_Campaign : more than 6 times can be one value and we can convert it to categorical
#Contacts_Before_Campaign
#------------------------------------------------------------------------------
#Previous_Contact_Days: can be converted to categorical : called and not called
df['Previous_Contact_Days'] = np.where(df['Previous_Contact_Days']==999,0,1)
df.Previous_Contact_Days.value_counts()

len(fc)
cs = df.groupby('Previous_Contact_Days').Client_Subscribed.value_counts(normalize=False)
#cs
#cs.unstack()
cs.unstack().plot(kind='bar')
#chi2 test for Previous_Contact_Days done it is signigicant

#------------------------------------------------------------------------------
#Anova test to check relation of numerical variables with categorical target------
import statsmodels.api as sm
# ANOVA test
from statsmodels.formula.api import ols
nc
#Client_Subscribed
#Price_Variation

#function
def anovatest(x,y,data):
    model = ols('x~y',data=data).fit()
    anova = sm.stats.anova_lm(model,typ=2)
    pval = anova['PR(>F)'][0]
    
    # return if the feature is significant or not
    if pval < 0.05:
        msg = '{} is significant'.format(x.name)
    else:
        msg = '{} is not significant'.format(x.name)
    
    return(msg)
 
anovatest(df['Employement_Rates'],df[ 'Client_Subscribed'], data=df)

for features in nc:
    msg=anovatest(df[features], df[ 'Client_Subscribed'], data=df)
    print(msg)

'''
Age ,Last_Contact_Duration ,Contacts_During_Campaign is significant
Previous_Contact_Days ,Contacts_Before_Campaign,Employement_Rates is significant
Price_Variation is significant
#DROP
Consumer_Confidence_Index ,Insurance_Rate ,No_Employees is not significant
'''
########################################################################
#but as per boxplot df.Price_Variation.describe() should not be significant
#================================

#check for mulicolinearity for numerical variables
#create heatmap for nc variable
nc
cormat = df[nc].corr()
cormat = np.tril(cormat)
#as per the observation no multicolinearity

#--------Things we can do post 1st DT model building------------------------------
#1)Undersampling
#2)drop General_Insurance,Consumer_Confidence_Index ,Insurance_Rate ,No_Employees is not significant
#3)Feature selection
#4)Hyper parameter
#tree pruning
#-------Model 1 building=======================================================
#split data into train test.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

Xtrain,Xtest,ytrain,ytest = train_test_split(df.drop('Client_Subscribed',1),df.Client_Subscribed,test_size=0.3)

model = DecisionTreeClassifier(criterion='gini').fit(Xtrain,ytrain)
p = model.predict(Xtest)

from sklearn.metrics import classification_report

# confusion matrix and classification report
ms = pd.DataFrame({'actual':ytest,'predicted':p})
pd.crosstab(ms.actual,ms.predicted,margins=True)
print(classification_report(ms.actual,ms.predicted))



#--------------------------practice ------------------------------
fp = 0
fn = 0

tp = 0
tn = 0



for actual_value,predicted_value in zip(ms.actual,ms.predicted):
    # let's first see if it's a true (t) or false prediction (f)
    if predicted_value == actual_value: # t?
        if predicted_value == 1: # tp
            tp += 1
        else: # tn
            tn += 1
    else: # f?
        if predicted_value == 1: # fp
            fp += 1
        else: # fn
            fn += 1
            
confusion_matrix = [
    [tn, fp],
    [fn, tp]
]

confusion_matrix = np.array(our_confusion_matrix)
#-------------------------------------------------------------------
for actual_value,predicted_value in zip(ms.actual,ms.predicted):
    # let's first see if it's a true (t) or false prediction (f)
    if predicted_value == actual_value:# t?
        for 
    
        if predicted_value == 1: # tp
            tp += 1
        else: # tn
            tn += 1
    else: # f?
        if predicted_value == 1: # fp
            fp += 1
        else: # fn
            fn += 1
            
our_confusion_matrix = [
    [tn, fp],
    [fn, tp]
]

our_confusion_matrix = np.array(our_confusion_matrix)


#-----------------------------------------------------------------
#0 = 93%
#1 = 50%
#accuracy = 88%


#--------------Model Summary---------------------------------------------------
model_summary = pd.DataFrame([('gini_bs','model',93,50,88)],
                             columns=('Algorithm','ModelName','0%','1%','acc %'))
print(model_summary)


#==============================================================================
#model2 building 
#perform undersampling---------------------------------------------------------
df.Client_Subscribed.value_counts()
from imblearn.under_sampling import NearMiss

nm=NearMiss()
chx,chy = nm.fit_resample(df.drop('Client_Subscribed',1), df['Client_Subscribed'])


df2 = chx.join(chy)
df2.shape
df2.Client_Subscribed.value_counts()

#split data into train and test
Xtrain1,Xtest1,ytrain1,ytest1= train_test_split(df2.drop('Client_Subscribed',1),df2.Client_Subscribed,test_size=0.3)


print(Xtrain1.shape,Xtest1.shape)
model1 = DecisionTreeClassifier(criterion='gini').fit(Xtrain1,ytrain1)
p1 = model1.predict(Xtest1)

# confusion matrix and classification report
df1 = pd.DataFrame({'actual':ytest1,'predicted':p1})
pd.crosstab(df1.actual,df1.predicted,margins=True)
print(classification_report(df1.actual,df1.predicted))
#accuracy =  0.84

#---------------------------------------------------------------------
Model_dict = {'Algorithm':'gini_undrsmple','ModelName':'model1','0%':84,'1%':84,'acc %':84}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)

#first drop the unneccary cols and then perform undersampling***************
#===============================================================================
#2)drop General_Insurance,Consumer_Confidence_Index ,Insurance_Rate ,No_Employees is not significant
df3 = df2.copy()
df2.drop(columns=['General_Insurance','Consumer_Confidence_Index','Insurance_Rate',
                  'No_Employees'],axis=1,inplace=True)

Xtrain2,Xtest2,ytrain2,ytest2= train_test_split(df2.drop('Client_Subscribed',1),df2.Client_Subscribed,test_size=0.3)


print(Xtrain2.shape,Xtest2.shape)
model2 = DecisionTreeClassifier(criterion='gini').fit(Xtrain2,ytrain2)
p2 = model2.predict(Xtest2)

# confusion matrix and classification report
ms3 = pd.DataFrame({'actual':ytest2,'predicted':p2})
pd.crosstab(ms3.actual,ms3.predicted,margins=True)
print(classification_report(ms3.actual,ms3.predicted))
#accuracy =  0.84


#model summary
Model_dict = {'Algorithm':'gini_undrsmpl 1','ModelName':'model2','0%':84,'1%':82,'acc %':83}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)

#-------------------------------------------------------------------------------

#method 1: feature selection
from sklearn.feature_selection import RFE
rfe = RFE(model2,n_features_to_select=10).fit(Xtrain2,ytrain2)
supp = rfe.support_
rank = rfe.ranking_

impfeat1 = pd.DataFrame({'features':Xtrain2.columns,
                         'support':supp,
                         'ranking':rank})

impfeat1.sort_values('ranking')
#------------------------------------------------------------------------------

#grid : hyper parameter========================================================
from sklearn.model_selection import GridSearchCV

params = {'criterion':['gini','entropy'],
          'max_depth':np.arange(1,13),
          'min_samples_leaf':np.arange(3,10),
          'max_features':np.arange(2,14)}

#create a emty decision tree
dtclf = DecisionTreeClassifier()

grid = GridSearchCV(dtclf,param_grid=params,
                    scoring='accuracy',cv=10,n_jobs=-1).fit(Xtrain2,ytrain2)

grid.best_params_
grid.best_score_

'''
{'criterion': 'entropy',
 'max_depth': 8,
 'max_features': 12,
 'min_samples_leaf': 4}
'''


model3 = DecisionTreeClassifier(criterion='entropy',max_depth=8,max_features=12,
                               min_samples_leaf=4).fit(Xtrain2,ytrain2)

p3 = model3.predict(Xtest2)


# confusion matrix and classification report
ms4 = pd.DataFrame({'actual':ytest2,'predicted':p3})
pd.crosstab(ms4.actual,ms4.predicted,margins=True)
print(classification_report(ms4.actual,ms4.predicted))
#86% accuracy

Model_dict = {'Algorithm':'hyperPara entropy','ModelName':'model3','0%':80,'1%':93,'acc %':86}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)

###############################################################################

#alpha value #prune tree-------------------------------------------------------
path = model3.cost_complexity_pruning_path(Xtrain2, ytrain2)
alphas , impurity = path.ccp_alphas,path.impurities

clfs=[]

alphas
for a in alphas:
    model = DecisionTreeClassifier(ccp_alpha=a).fit(Xtrain2,ytrain2)
    clfs.append(model)

print(clfs)

trainscore = [clf.score(Xtrain2,ytrain2)for clf in clfs]
testscore = [clf.score(Xtest2,ytest2)for clf in clfs]  


fig,ax=plt.subplots()
ax.plot(alphas,trainscore,marker='x',label='train',drawstyle="steps-post")
ax.plot(alphas,testscore,marker='x',label='test',drawstyle="steps-post")
ax.set_xlabel("Alphas")
ax.set_ylabel("accuracy")
ax.legend()



params1 = ({'criterion':['gini','entropy'],'max_depth':np.arange(6,10),
            'min_samples_leaf':np.arange(3,6),'max_features':np.arange(10,14),
            'ccp_alpha':np.arange(0.0005,0.0016)})

dtclf1 = DecisionTreeClassifier()
grid1 = GridSearchCV(dtclf1,param_grid=params1,scoring='accuracy',
                     cv=10,n_jobs=-1).fit(Xtrain2,ytrain2)

grid1.best_params_
grid1.best_score_
#{'ccp_alpha': 0.0005, 'criterion': 'gini', 'max_depth': 7, 'max_features': 12, 'min_samples_leaf': 5}
model4 = DecisionTreeClassifier(criterion='gini',max_depth=8,max_features=11,
                                min_samples_leaf=3,ccp_alpha=0.0005).fit(Xtrain2,ytrain2)

p4=model4.predict(Xtest2)
ms5 = pd.DataFrame({'actual':ytest2,'predicted':p4})
pd.crosstab(ms5.actual,ms5.predicted,margins=True)
print(classification_report(ms5.actual,ms5.predicted))
#accuracy = 0.87

#model summary----------------

Model_dict = {'Algorithm':'gini_pruned','ModelName':'model4','0%':85,'1%':88,'acc %':87}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)


#random forest------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
model_rf1 = RandomForestClassifier().fit(Xtrain2,ytrain2)

p_rf= model_rf1.predict(Xtest2)

df_rf1 = pd.DataFrame({'actual':ytest2,'predicted':p_rf})
pd.crosstab(df_rf1.actual,df_rf1.predicted,margins=True)
print(classification_report(df_rf1.actual,df_rf1.predicted))


Model_dict = {'Algorithm':'RF-base','ModelName':'model_rf1','0%':84,'1%':92,'acc %':88}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)
#-----------------------------------------------------------------------------
#MODEL SUMMARY
#------***-------------------------***------------------------------------***---
'''
           Algorithm  ModelName  0%  1%  acc %
0            gini_bs      model  93  50     88
1     gini_undrsmple         m2  84  84     84
2    gini_undrsmpl 1     model2  84  82     83
3  hyperPara entropy     model3  80  93     86
4        gini_pruned     model4  85  88     87
5            RF-base  model_rf1  84  92     88
'''

#check the model###############################################################
'''
model_rf2 = RandomForestClassifier(ccp_alpha=0.0006,n_estimators=85,
                                   max_features=15,criterion='gini').fit(Xtrain2,ytrain2)
p_rf2= model_rf2.predict(Xtest2)
df_rf2 = pd.DataFrame({'actual':ytest2,'predicted':p_rf2})
pd.crosstab(df_rf2.actual,df_rf2.predicted,margins=True)
print(classification_report(df_rf2.actual,df_rf2.predicted))
#accuracy =0.85%
help(RandomForestClassifier)
'''
df_copy.shape
Xtrain2.shape
df_copy.columns
Xtrain2.columns
#'Consumer_Confidence_Index','Insurance_Rate', 'No_Employees',General_Insurance

df_copy.drop(columns=['Consumer_Confidence_Index','Insurance_Rate', 'No_Employees','General_Insurance'],axis=1,inplace=True)
df_copy.shape

Xtrain4,Xtest4,ytrain4,ytest4=train_test_split(df_copy.drop('Client_Subscribed',1),
                                               df_copy.Client_Subscribed,test_size=0.2)


Xtest4.shape
ytrain4.value_counts()

chk_pred = model_rf1.predict(Xtest4)
df_chk = pd.DataFrame({'actual':ytest4,'predicted':chk_pred})
pd.crosstab(df_chk.actual,df_chk.predicted,margins=True)
print(classification_report(df_chk.actual,df_chk.predicted))
#76 99 79 train4


ch2_p=model3.predict(Xtrain4)
df_chk = pd.DataFrame({'actual':ytrain4,'predicted':ch2_p})
pd.crosstab(df_chk.actual,df_chk.predicted,margins=True)
print(classification_report(df_chk.actual,df_chk.predicted))
#69 96 72


ch3_p = model2.predict(Xtrain4)
df_chk = pd.DataFrame({'actual':ytrain4,'predicted':ch3_p})
pd.crosstab(df_chk.actual,df_chk.predicted,margins=True)
print(classification_report(df_chk.actual,df_chk.predicted))
#71 91 81
###############################################################################
#Performing Hybrid sampling and then applying the algorithms ==================
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
df.shape
df.columns
df.drop(columns=['General_Insurance','Consumer_Confidence_Index','Insurance_Rate',
                  'No_Employees'],axis=1,inplace=True)
perc = 0.75
oversamp = SMOTE(sampling_strategy = perc)
undersamp = RandomUnderSampler(sampling_strategy = perc)

steps = [('o',oversamp),('u',undersamp)]

chx_h,chy_h = Pipeline(steps=steps).fit_resample(df.drop('Client_Subscribed',1),
                                             df.Client_Subscribed)

df_hybrid = chx_h.join(chy_h)

df_hybrid.shape
df_hybrid.Client_Subscribed.value_counts()

#-Split data into train and test-----------------------------------------------

Xtrain_h,Xtest_h,ytrain_h,ytest_h= train_test_split(df_hybrid.drop('Client_Subscribed',1),df_hybrid.Client_Subscribed,test_size=0.3)

Xtest_h.shape
#build Random forest model and DT on this data----------------------------------
from sklearn.model_selection import RandomizedSearchCV

params = {'criterion':['gini','entropy'],
          'max_depth':np.arange(10,16),
          'min_samples_leaf':np.arange(3,10),
          'max_features':np.arange(2,14)}

#create a emty decision tree
dtclf = DecisionTreeClassifier()

random_search = RandomizedSearchCV(dtclf,param_distributions=params,n_iter=5,
                    scoring='accuracy',cv=10,n_jobs=-1).fit(Xtrain_h,ytrain_h)

random_search.best_score_
0.91
bp1 = random_search.best_params_
#{'min_samples_leaf': 8, 'max_features': 13, 'max_depth': 15, 'criterion': 'entropy'}

m_h = DecisionTreeClassifier(criterion=bp1['criterion'],min_samples_leaf=bp1['min_samples_leaf'],
                             max_features=bp1['max_features'],
                             max_depth=bp1['max_depth']).fit(Xtrain_h,ytrain_h)


p_h = m_h.predict(Xtest_h)
df5 = pd.DataFrame({'actual':ytest_h,'predicted':p_h})
pd.crosstab(df5.actual,df5.predicted,margins=True)

print(classification_report(df5.actual,df5.predicted))

#$-model summary
Model_dict = {'Algorithm':'DT_hybrid','ModelName':'m_h','0%':92,'1%':90,'acc %':92}
model_summary= model_summary.append(Model_dict,ignore_index=True)
print(model_summary)
#---------------------------------------------------------
path = m_h.cost_complexity_pruning_path(Xtrain_h, ytrain_h)
alphas , impurity = path.ccp_alphas,path.impurities

clfs=[]

alphas
for a in alphas:
    model = DecisionTreeClassifier(ccp_alpha=a).fit(Xtrain_h,ytrain_h)
    clfs.append(model)

print(clfs)

trainscore = [clf.score(Xtrain_h,ytrain_h)for clf in clfs]
testscore = [clf.score(Xtest_h,ytest_h)for clf in clfs]  


fig,ax=plt.subplots()
ax.plot(alphas,trainscore,marker='x',label='train',drawstyle="steps-post")
ax.plot(alphas,testscore,marker='x',label='test',drawstyle="steps-post")
ax.set_xlabel("Alphas")
ax.set_ylabel("accuracy")
ax.legend()

m6 = DecisionTreeClassifier(criterion=bp1['criterion'],min_samples_leaf=bp1['min_samples_leaf'],
                             max_features=bp1['max_features'],
                             max_depth=bp1['max_depth'],
                             ccp_alpha=0.0002).fit(Xtrain_h,ytrain_h)


p6 = m6.predict(Xtest_h)
df6 = pd.DataFrame({'actual':ytest_h,'predicted':p6})
pd.crosstab(df6.actual,df6.predicted,margins=True)

print(classification_report(df6.actual,df6.predicted))

#--------------------