############################################################################################################
Problem Statements: -

1.	An analytics company has been tasked with the crucial job of finding out what factors affect a startup 
company and if it will be profitable or not. For this, they have collected some historical data and would 
like to apply multilinear regression to derive brief insights into their data. Predict profit, given 
different attributes for various startup companies.
###########################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sfa
from sklearn.metrics import r2_score

# Reading the data into Python
data = pd.read_csv('D:/Hands on/24_Multiple Linear Regression/Assignment/50_Startups.csv')

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# correlation coefficient
data.corr()

# Pairplot
sns.pairplot(data)

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isnull().sum()

# Spliting data into X and Y
X = data.iloc[:, 0:4]

Y = data.iloc[:, -1]

# Spliting into categorical and numerical features
num_features = X.select_dtypes(exclude = ['object'])

cat_features = X.select_dtypes(include = ['object'])

# Columns
cat_features.columns

# Creating dummy variables
cat_features1 = pd.get_dummies(cat_features, drop_first = True)

# for loop to see the outliers
for i in num_features.columns:
    plt.boxplot(num_features[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

# Scaling the data
minmax = StandardScaler()
num_features1 = pd.DataFrame(minmax.fit_transform(num_features), columns = num_features.columns)

# Concatinating X and Y
final_data = pd.concat([num_features1, cat_features1, Y], axis = 1)

# Columnds
final_data.columns 

# correlation coefficeint
data.corr()

# Renaming the columns
final_data.rename({'R&D Spend' : 'RD_Spend', 'State_New York' : 'State_NewYork',
                   'Marketing Spend' : 'Marketing_Spend'}, inplace = True, axis = 1)

# building the model
model1 = sfa.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State_Florida + State_NewYork', data = final_data).fit()

# Summary of the model
model1.summary()

# Checking for influence rows
sma.graphics.influence_plot(model1)

# droping the influencial obervarions
final_data.drop([45, 46, 48, 49], axis = 0, inplace = True)

# Building the model
model2 = sfa.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State_Florida + State_NewYork', data = final_data).fit()

# Summary of the model
model2.summary()

# Calculating VIF values
rd_r2 = sfa.ols('RD_Spend ~ Administration + Marketing_Spend + State_Florida + State_NewYork', data = final_data).fit().rsquared

rd_vif = (1/ (1-rd_r2))

adm_r2 = sfa.ols('Administration ~ RD_Spend + Marketing_Spend + State_Florida + State_NewYork', data = final_data).fit().rsquared

adm_vif = (1/ (1-adm_r2))

mar_r2 = sfa.ols('Marketing_Spend ~ Administration + RD_Spend  + State_Florida + State_NewYork', data = final_data).fit().rsquared

mar_vif = (1/ (1-mar_r2))

flo_r2 = sfa.ols('State_Florida  ~ Administration + RD_Spend  + Marketing_Spend + State_NewYork', data = final_data).fit().rsquared

flo_vif = (1/ (1-flo_r2))

Ny_r2 = sfa.ols('State_NewYork  ~ Administration + RD_Spend  + Marketing_Spend + State_Florida', data = final_data).fit().rsquared

Ny_vif = (1/ (1-Ny_r2))

# Creating a dataframe with VIF values
dict = pd.DataFrame({'VIF' : ['rd_vif', 'adm_vif', 'mar_r2', 'flo_r2', 'Ny_vif'],
        'Values' : [rd_vif, adm_vif, mar_r2, flo_r2, Ny_vif]})

# Spliting the data into x_input and y_output
x_input = final_data.iloc[:, 0:5]
y_output = final_data.iloc[:, -1]

# spliting the data into train and test
train, test = train_test_split(final_data, test_size = 0.2, random_state = 0) 

# Building the model
final_data = sfa.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State_Florida + State_NewYork', data = train).fit()

# Summary of the model
final_data.summary()

# Test prediction
test_pred = final_data.predict(test)

# Test error
test_err = test.Profit - test_pred

# RMSE
test_rmse = np.sqrt(np.mean(test_err * test_err))

# residual
resd = final_data.resid

# R2
test_r2score = r2_score(test.Profit, test_pred)

# Train predictions
train_pred = final_data.predict(train)

# Train error
train_err = train.Profit - train_pred

# R2
train_r2score = r2_score(train.Profit, train_pred)

# RMSE
train_rmse = np.sqrt(np.mean(train_err * train_err))

########################################################################################################
Problem Statements: -

2.	Perform multilinear regression with price as the output variable and document the different RMSE values.

#########################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sfa
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, r2_score


# Reading the data into Python
data = pd.read_csv('D:/Hands on/24_Multiple Linear Regression/Assignment/Computer_Data.csv')

data.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Checking for duplicates
data.duplicated().sum()

# Droping the duplicates
data.drop_duplicates(inplace = True)

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isnull().sum()

# Spliting the data into X and Y
Y = data.iloc[:, 0]

X = data.iloc[:, 1:]

# CReating Dummy variables
X = pd.get_dummies(X, drop_first = True)

# For loop for boxplot
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

# WInsorizer
winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = list(X.columns))

cleandata = pd.DataFrame(winsor.fit_transform(X), columns = X.columns)

# For loop for boxplot
for i in cleandata.columns:
    plt.boxplot(cleandata[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

# Sclaing
minmax = MinMaxScaler()
cleandata = pd.DataFrame(minmax.fit_transform(cleandata), columns = cleandata.columns)

# Concatinating two datasets
final_data = pd.concat([Y, cleandata], axis = 1)

# COlumns
final_data.columns

# COrrelation coefficient
data.corr()

# Linear regression object
lregree = LinearRegression()

# Spliting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(cleandata, Y, test_size = 0.2, random_state= 0)

# Builiding the model on training data
model1 = lregree.fit(x_train, y_train)

# Prediction on test data
test_pred = model1.predict(x_test)

# Residual
test_err =  y_test - test_pred

# R2
test_r2 = r2_score(y_test, test_pred)

# Prediction on train data
train_pred = model1.predict(x_train)

# Residuals
train_eer = y_train - train_pred

# R2
train_r2 = r2_score(y_train, train_pred)

#######################################################################################################
Problem Statements: -

3.	An online car sales platform would like to improve its customer base and their experience by 
providing them an easy way to buy and sell cars. For this, they would like an automated model which
 can predict the price of the car once the user inputs the required factors. 
 Help the business achieve their objective by applying multilinear regression on the given dataset.
 Please use the below columns for the analysis purpose: price, age_08_04, KM, HP, cc, Doors, Gears, 
 Quarterly_Tax, and Weight.

#######################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sfa
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, r2_score
from statsmodels.tools.tools import add_constant
import statsmodels.api as sma
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Reading the data into Python
data = pd.read_csv('D:/Hands on/24_Multiple Linear Regression/Assignment/Avacado_Price.csv')

data.columns

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isna().sum()

# Spliting the data into train and test
X = data.iloc[:, 1:]

Y = data.iloc[:, 0]

# for loop for boxplot
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for '+ str(i))
    plt.show()
    
# Winsorization
winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = list(X.columns))
X = winsor.fit_transform(X)

# for loop for boxplot
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for '+ str(i))
    plt.show()

# Correlation coefficient
X.corr()

# Adding constant
P = add_constant(X)

# Creating a base model
basemodel = sma.OLS(Y, X).fit()

# Model summary
basemodel.summary()

# VIF
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index = X.columns)
vif

# Deleting a feature which is having highest VIF
X.drop(['Gears'], axis = 1, inplace = True)

# Builind the model
basemodel2 = sma.OLS(Y, X).fit()

# SUmmary of the model
basemodel2.summary()

# Influencial graph
sma.graphics.influence_plot(basemodel2)

# SPliting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

# Building a model on train data
mlrmodel = sma.OLS(Y_train, X_train).fit()

# Prediction on test data
test_pred = mlrmodel.predict(X_test)

# R2
r2_score(Y_test, test_pred)

# Residuals
test_resid = Y_test - test_pred

# RMSE
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

# Prediction on train data
train_pred = mlrmodel.predict(X_train)

# R2
r2_score(Y_train, train_pred)

# Residuals
train_resid = Y_train - train_pred

# RMSE
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

#######################################################################################################

4.	With the growing consumption of avocados in the USA, a freelance company would like to do some 
analysis on the patterns of consumption in different cities and would like to come up with a 
prediction model for the price of avocados. For this to be implemented, build a prediction 
model using multilinear regression and provide your insights on it. Snapshot of the dataset is 
given below: -

#####################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Reading the data into Python
data = pd.read_csv('D:/Hands on/24_Multiple Linear Regression/Assignment/Avacado_Price.csv')

data.columns

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isna().sum()

# SPliting the data into train and test
X = data.iloc[:, 1:]

Y = data.iloc[:, 0]

# Columns
X.columns

# Correlation coefficient
X.corr()

# Pairplot
sns.pairplot(X)

# Unique values
X.region.nunique() 

# Lable encoder
lb = LabelEncoder()
X['region'] = lb.fit_transform(X['region'])
X['type'] = lb.fit_transform(X['type'])

# forloop for boxplot
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for ' + str())
    plt.show()
    
# Winsorization
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = list(X.columns))
X = winsor.fit_transform(X)

# forloop for boxplot
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for ' + str())
    plt.show()
    
# spliting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
lr = LinearRegression()

# Model builing on train data
model = lr.fit(x_train, y_train)

# Prediction on train data
train_pred = model.predict(x_train)

# RMSE
train_rmse = np.sqrt(np.mean(train_pred * train_pred))

# R2
r2_score(y_train, train_pred)

# residuals
train_resid = y_train - train_pred 

# Testing prediction
test_pred = model.predict(x_test)

# RMSE
train_rmse = np.sqrt(np.mean(test_pred * test_pred))

# R2
r2_score(y_test, test_pred)

# testing residual
test_resid = y_test - test_pred 
