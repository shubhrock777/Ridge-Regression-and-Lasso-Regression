import pandas as pd


#loading the dataset
computer = pd.read_csv("D:/BLR10AM/Assi/25.lasso ridge regression/Datasets_LassoRidge/Computer_Data (1).csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Index row number (irrelevant ,does not provide useful Informatiom)",
                "Price of computer(relevant provide useful Informatiom)",
                "computer speed (relevant provide useful Informatiom)",
                "Hard Disk space of computer (relevant provide useful Informatiom)",
                "Random axis momery of computer (relevant provide useful Informatiom)",
                "Screen size of Computer (relevant provide useful Informatiom)",
                "Compact dist (relevant provide useful Informatiom)",
                "Multipurpose use or not (relevant provide useful Informatiom)",
                "Premium Class of computer (relevant provide useful Informatiom)",
                "advertisement expenses (relevant provide useful Informatiom)",
                "Trend position in market (relevant provide useful Informatiom)"]

d_types =["Count","Ratio","Ratio","Ratio","Ratio","Ratio","Binary","Binary","Binary","Ratio","Ratio"]

data_details =pd.DataFrame({"column name":computer.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": computer.dtypes})

            #3.	Data Pre-computercessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of computer 
computer.info()
computer.describe()          

#droping index colunms 
computer.drop(['Unnamed: 0'], axis = 1, inplace = True)


#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


computer['cd'] = LE.fit_transform(computer['cd'])
computer['multi'] = LE.fit_transform(computer['multi'])
computer['premium'] = LE.fit_transform(computer['premium'])

#data types        
computer.dtypes


#checking for na value
computer.isna().sum()
computer.isnull().sum()

#checking unique value for each columns
computer.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": computer.columns,
      "mean": computer.mean(),
      "median":computer.median(),
      "mode":computer.mode(),
      "standard deviation": computer.std(),
      "variance":computer.var(),
      "skewness":computer.skew(),
      "kurtosis":computer.kurt()}

EDA


# covariance for data set 
covariance = computer.cov()
covariance

# Correlation matrix 
co = computer.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphicomputer repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(computer.iloc[:, :])


#boxplot for every columns
computer.columns
computer.nunique()

computer.boxplot(column=['price','ads', 'trend'])   #no outlier

#for imputing HVO for Price column
"""
# here we can see lVO For Price
# Detection of outliers (find limits for RM based on IQR)
IQR = computer['Price'].quantile(0.75) - computer['Price'].quantile(0.25)
upper_limit = computer['Price'].quantile(0.75) + (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

computer['Price']= pd.DataFrame( np.where(computer['Price'] > upper_limit, upper_limit, computer['Price']))

import seaborn as sns 
sns.boxplot(computer.Price);plt.title('Boxplot');plt.show()"""



# Q-Q Plot
from scipy import stats
import pylab
import matplotlib.pyplot as plt

stats.probplot(computer.price, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.ads, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.trend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(computer.iloc[:,[1,2,3,4,8,9]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=computer.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = computer.iloc[:,[5,6,7]]

model_df = pd.concat([computer.iloc[:,[0]],enc_df, df_norm ], axis =1)



"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

##################################
###LASSO MODEL###

import numpy as np
from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.7, normalize = True)

#model building
lasso.fit(model_df.iloc[:, 1:], model_df.price)

# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(model_df.columns[1:]))
#state columns has lowest coefficent 

lasso.alpha

pred_lasso = lasso.predict(model_df.iloc[:, 1:])

# Adjusted r-square#
lasso.score(model_df.iloc[:, 1:], model_df.price)

#RMSE
np.sqrt(np.mean((pred_lasso -model_df.price)**2))

#####################
#lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha' : [1e-30,1e-38,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-10,0,40,80]}

lasso_reg = GridSearchCV(lasso, parameters , scoring = 'r2' ,cv = 5)

lasso_reg.fit(model_df.iloc[:,1:],model_df.price)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(model_df.iloc[:,1:])

#Adjusted R- square
lasso_reg.score(model_df.iloc[:,1:],model_df.price)

#RMES

np.sqrt(np.mean((lasso_pred-model_df.price)**2))
 
### RIDGE REGRESSION ###

from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(model_df.iloc[:, 1:], model_df.price)

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(model_df.columns[1:]))

rm.alpha

pred_rm = rm.predict(model_df.iloc[:, 1:])

# adjusted r-square#
rm.score(model_df.iloc[:, 1:],model_df.price)

#RMSE
np.sqrt(np.mean((pred_rm - model_df.price)**2))

#####################
#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha' : [1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,0,40,80]}

ridge_reg = GridSearchCV(ridge, parameters , scoring = 'r2' ,cv = 5)
ridge_reg.fit(model_df.iloc[:,1:], model_df.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(model_df.iloc[:,1:])

#Adjusted R- square
ridge_reg.score(model_df.iloc[:,1:], model_df.price)

#RMES

np.sqrt(np.mean((ridge_pred- model_df.price)**2))




