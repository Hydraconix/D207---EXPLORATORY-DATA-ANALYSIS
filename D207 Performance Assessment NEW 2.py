# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 03:21:19 2021

@author: Hydraconix
"""

# Standard Data Science Imports
import numpy as np
import pandas as pd
from pandas import DataFrame


#Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


#Statistics packages
import pylab
import statsmodels.api as sm
import statistics
from scipy import stats


# Import chi-square from scipy.stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import chi2


#Load data set into Pandas dataframe
df = pd.read_csv(r'C:\Users\Hydraconix\Desktop\DATA\churn_clean.csv')


#Checking data types
df.info()


#Checking for null values
df.isna().sum()


#Rename Last survey columns for better description of variables
df.rename(columns = {'Item1':'Timely Response',
                     'Item2':'Fixes',
                     'Item3':'Replacements',
                     'Item4':'Reliability',
                     'Item5':'Options',
                     'Item6':'Respectfulness',
                     'Item7':'Courteous',
                     'Item8':'Listening'},
                    inplace=True)


#Displaying the frequency distribution for churn
plt.figure(figsize=(5,5))
ax = sns.countplot(x=df['Churn'], palette="Blues", linewidth=1)
plt.show()


#Displaying the count of the Churn variable
print(df.Churn.value_counts())


#Displaying the percentage of the Churn variable
df.Churn.value_counts(normalize=True)*100


#Creating Contingency table to compare 2 variables (Churn & Timely Response)
contingency = pd.crosstab(df['Churn'],df['Timely Response'])
contingency


#Creating Heatmap (Churn & Timely Response)
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")


# Chi-Square Test: degrees of freedom (Churn & Timely Response)
stat, p, dof, expected = chi2_contingency(contingency)
print('dof=%d' % dof)


#Interpret test-statistic (Churn & Timely Response)
prob=0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
     
     
#Interpret p-value (Churn & Timely Response)
alpha = 1 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


#Creating Contingency table to compare 2 variables (Churn & Reliability)
contingency2 = pd.crosstab(df['Churn'],df['Reliability'])
contingency2


#Creating Heatmap (Churn & Reliability)
plt.figure(figsize=(12,8))
sns.heatmap(contingency2, annot=True, cmap="YlGnBu")


# Chi-Square Test: degrees of freedom (Churn & Reliability)
stat, p, dof, expected = chi2_contingency(contingency2)
print('dof=%d' % dof)


#Interpret test-statistic (Churn & Reliability)
prob=0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
     
     
#Interpret p-value (Churn & Reliability)
alpha = 1 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


#Creating Contingency table to compare 2 variables (Churn & Courteous)
contingency3 = pd.crosstab(df['Churn'],df['Courteous'])
contingency3


#Creating Heatmap (Churn & Courteous)
plt.figure(figsize=(12,8))
sns.heatmap(contingency3, annot=True, cmap="YlGnBu")


# Chi-Square Test: degrees of freedom (Churn & Courteous)
stat, p, dof, expected = chi2_contingency(contingency3)
print('dof=%d' % dof)


#Interpret test-statistic (Churn & Courteous)
prob=0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
     
     
#Interpret p-value (Churn & Courteous)
alpha = 1 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


#Create histograms of contiuous & categorical variables
df[['MonthlyCharge', 'Bandwidth_GB_Year', 'Timely Response', 'Courteous']].hist()
plt.savefig('churn_pyplot.jpg')
plt.tight_layout()


#Create Seaborn boxplots for continuous & categorical variables
sns.boxplot('MonthlyCharge', data = df)
plt.show()


sns.boxplot('Bandwidth_GB_Year', data = df)
plt.show()


sns.boxplot('Timely Response', data = df)
plt.show()


sns.boxplot('Courteous', data = df)
plt.show()


# Create dataframe for heatmap bivariate analysis of correlation
churn_bivariate = df[['MonthlyCharge', 'Bandwidth_GB_Year', 'Timely Response', 'Courteous']]


#Correlation Matrix
churn_bivariate.corr()


#Heatmaps for bivariate analysis of correlation
sns.heatmap(churn_bivariate.corr(), annot=True, cmap='coolwarm')
plt.show()


# Create a scatter plot of continuous variables MonthlyCharge & Bandwidth_GB_Year
churn_bivariate[churn_bivariate['MonthlyCharge'] < 300].sample(100).plot.scatter(x='MonthlyCharge', 
                                                                                 y='Bandwidth_GB_Year')

# Create a scatter plot of categorical variables TimelyResponse & Courteous
churn_bivariate[churn_bivariate['Timely Response'] < 7].sample(100).plot.scatter(x='Timely Response', 
                                                                                 y='Courteous')

# Create a hexplot of continuous variables MonthlyCharge & Bandwidth_GB_Year
churn_bivariate[churn_bivariate['MonthlyCharge'] < 300].plot.hexbin(x='MonthlyCharge', y='Bandwidth_GB_Year', gridsize=15)    



