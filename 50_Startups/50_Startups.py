import pandas as pd


#loading the dataset
startup = pd.read_csv("D:/BLR10AM/Assi/25.lasso ridge regression/Datasets_LassoRidge/50_Startups (1).csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Money spend on research and development",
                "Administration",
                "Money spend on Marketing",
                "Name of state",
                "Company profit"]

d_types =["Ratio","Ratio","Ratio","Nominal","Ratio"]

data_details =pd.DataFrame({"column name":startup.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": startup.dtypes})

            #3.	Data Pre-startupcessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of startup 
startup.info()
startup.describe()          


#rename the columns
startup.rename(columns = {'R&D Spend':'rd_spend', 'Marketing Spend' : 'm_spend'} , inplace = True)  

#data types        
startup.dtypes


#checking for na value
startup.isna().sum()
startup.isnull().sum()

#checking unique value for each columns
startup.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": startup.columns,
      "mean": startup.mean(),
      "median":startup.median(),
      "mode":startup.mode(),
      "standard deviation": startup.std(),
      "variance":startup.var(),
      "skewness":startup.skew(),
      "kurtosis":startup.kurt()}

EDA


# covariance for data set 
covariance = startup.cov()
covariance

# Correlation matrix 
co = startup.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphistartup repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(startup.iloc[:, :])


#boxplot for every columns
startup.columns
startup.nunique()

startup.boxplot(column=['rd_spend', 'Administration', 'm_spend', 'Profit'])   #no outlier

# here we can see lVO For profit
# Detection of outliers (find limits for RM based on IQR)
IQR = startup['Profit'].quantile(0.75) - startup['Profit'].quantile(0.25)
lower_limit = startup['Profit'].quantile(0.25) - (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#startup['Profit']= pd.DataFrame( np.where(startup['Profit'] < lower_limit, lower_limit, startup['Profit']))

import seaborn as sns 
sns.boxplot(startup.Profit);plt.title('Boxplot');plt.show()



# rd_spend
plt.bar(height = startup.rd_spend, x = np.arange(1, 51, 1))
plt.hist(startup.rd_spend) #histogram
plt.boxplot(startup.rd_spend) #boxplot


# Administration
plt.bar(height = startup.Administration, x = np.arange(1, 51, 1))
plt.hist(startup.Administration) #histogram
plt.boxplot(startup.Administration) #boxplot

# m_spend
plt.bar(height = startup.m_spend, x = np.arange(1, 51, 1))
plt.hist(startup.m_spend) #histogram
plt.boxplot(startup.m_spend) #boxplot




#profit
plt.bar(height = startup.Profit, x = np.arange(1, 51, 1))
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot


# Jointplot

sns.jointplot(x=startup['Profit'], y=startup['rd_spend'])



# Q-Q Plot
from scipy import stats
import pylab

stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show() 
# startupfit is normally distributed

stats.probplot(startup.Administration, dist = "norm", plot = pylab)
plt.show() 
# administration is normally distributed


stats.probplot(startup.rd_spend, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(startup.m_spend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup.iloc[:,[0,1,2]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=startup.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = pd.get_dummies(startup.iloc[:,[3]])
enc_df.columns
enc_df.rename(columns={"State_New York":'State_New_York'},inplace= True)

model_df = pd.concat([enc_df, df_norm, startup.iloc[:,4]], axis =1)

# Rearrange the order of the variables
model_df = model_df.iloc[:, [6, 0,1, 2, 3,4,5]]
"""
1.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Lasso and Ridge Regression Algorithm
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values, RMSE for different models in documentation and provide your explanation on it
Briefly explain the model output in the documentation. """


##################################
###LASSO MODEL###


from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.7, normalize = True)

#model building
lasso.fit(model_df.iloc[:, 1:], model_df.Profit)

# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(model_df.columns[1:]))
#state columns has lowest coefficent 

lasso.alpha

pred_lasso = lasso.predict(model_df.iloc[:, 1:])

# Adjusted r-square#
lasso.score(model_df.iloc[:, 1:], model_df.Profit)

#RMSE
np.sqrt(np.mean((pred_lasso - model_df.Profit)**2))

#####################
#lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha' : [1e-15,1e-10,0,40,80,160,320,1000,1900,1960,1970,2000,2001,4000]}

lasso_reg = GridSearchCV(lasso, parameters , scoring = 'r2' ,cv = 5)

lasso_reg.fit(model_df.iloc[:,1:],model_df.Profit)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(model_df.iloc[:,1:])

#Adjusted R- square
lasso_reg.score(model_df.iloc[:,1:],model_df.Profit)

#RMES

np.sqrt(np.mean((lasso_pred-model_df.Profit)**2))
 
### RIDGE REGRESSION ###

from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(model_df.iloc[:, 1:], model_df.Profit)

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(model_df.columns[1:]))

rm.alpha

pred_rm = rm.predict(model_df.iloc[:, 1:])

# adjusted r-square#
rm.score(model_df.iloc[:, 1:],model_df.Profit)

#RMSE
np.sqrt(np.mean((pred_rm - model_df.Profit)**2))

#####################
#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha' : [4000,8000,16000,32000,64000,150000,300000,600000,1200000]}

ridge_reg = GridSearchCV(ridge, parameters , scoring = 'r2' ,cv = 5)
ridge_reg.fit(model_df.iloc[:,1:], model_df.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(model_df.iloc[:,1:])

#Adjusted R- square
ridge_reg.score(model_df.iloc[:,1:], model_df.Profit)

#RMES

np.sqrt(np.mean((ridge_pred- model_df.Profit)**2))
