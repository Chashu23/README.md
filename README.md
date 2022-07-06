# Chashu23
import pandas as pd
cost = pd.read_excel("C:\Data science\8888\Project Dataset\cost_and_price.xlsx")
cost.head()
cost.columns
cost.describe()
cost.dtypes

# EDA
# mean
cost.Enrolled_and_users.mean()
cost.Instructor.mean()
cost.Study_material_price.mean()
cost.Office_rent.mean()
cost.Infrastructure_cost.mean()
cost.Electricitybill.mean()
cost.Course_duration.mean()
cost.Price_of_course.mean()

# median
cost.Enrolled_and_users.median()
cost.Instructor.median()
cost.Study_material_price.median()
cost.Office_rent.median()
cost.Infrastructure_cost.median()
cost.Electricitybill.median()
cost.Course_duration.median()
cost.Price_of_course.median()

# mode
cost.Enrolled_and_users.mode()
cost.Instructor.mode()
cost.Study_material_price.mode()
cost.Office_rent.mode()
cost.Infrastructure_cost.mode()
cost.Electricitybill.mode()
cost.Course_duration.mode()
cost.Price_of_course.mode()

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# Variance/ Second moment business decision
cost.Enrolled_and_users.var()
cost.Instructor.var()
cost.Study_material_price.var()
cost.Office_rent.var()
cost.Infrastructure_cost.var()
cost.Electricitybill.var()
cost.Course_duration.var()
cost.Price_of_course.var()

# Standard deviation/ Second moment business desicion
cost.Enrolled_and_users.std()
cost.Instructor.std()
cost.Study_material_price.std()
cost.Office_rent.std()
cost.Infrastructure_cost.std()
cost.Electricitybill.std()
cost.Course_duration.std()
cost.Price_of_course.std()

range = max(cost.Enrolled_and_users) - min(cost.Enrolled_and_users)
range
range = max(cost.Instructor) - min(cost.Instructor)
range
range = max(cost.Study_material_price) - min(cost.Study_material_price)
range
range = max(cost.Office_rent) - min(cost.Office_rent)
range 
range = max(cost.Infrastructure_cost) - min(cost.Infrastructure_cost)
range
range = max(cost.Electricitybill) - min(cost.Electricitybill)
range
range = max(cost.Course_duration) - min(cost.Course_duration)
range
range = max(cost.Price_of_course) - min(cost.Price_of_course)
range


# Third moment business decision
cost.shape

plt.bar(height = cost.Enrolled_and_users, x = np.arange(1, 323, 1))
plt.bar(height = cost.Instructor, x = np.arange(1, 323, 1))
plt.bar(height = cost.Study_material_price, x = np.arange(1, 323, 1))
plt.bar(height = cost.Office_rent, x = np.arange(1, 323, 1))
plt.bar(height = cost.Infrastructure_cost, x = np.arange(1, 323, 1))
plt.bar(height = cost.Electricitybill, x = np.arange(1, 323, 1))
plt.bar(height = cost.Course_duration, x = np.arange(1, 323, 1))
plt.bar(height = cost.Price_of_course, x = np.arange(1, 323, 1))

plt.hist(cost.Enrolled_and_users)
plt.hist(cost.Instructor)
plt.hist(cost.Study_material_price)
plt.hist(cost.Office_rent)
plt.hist(cost.Infrastructure_cost)
plt.hist(cost.Electricitybill)
plt.hist(cost.Course_duration)
plt.hist(cost.Price_of_course)
help(plt.hist)

plt.boxplot(cost.Enrolled_and_users)

plt.boxplot(cost.Study_material_price)
plt.boxplot(cost.Office_rent)
plt.boxplot(cost.Infrastructure_cost)
plt.boxplot(cost.Electricitybill)
plt.boxplot(cost.Course_duration)
plt.boxplot(cost.Price_of_course)
help(plt.boxplot)

# Detection of outliers (find limits for salary based on IQR)
IQR = cost['Study_material_price'].quantile(0.75) - cost['Study_material_price'].quantile(0.25)
lower_limit = cost['Study_material_price'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Study_material_price'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Price_of_course'].quantile(0.75) - cost['Price_of_course'].quantile(0.25)
lower_limit = cost['Price_of_course'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Price_of_course'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Enrolled_and_users'].quantile(0.75) - cost['Enrolled_and_users'].quantile(0.25)
lower_limit = cost['Enrolled_and_users'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Enrolled_and_users'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Office_rent'].quantile(0.75) - cost['Office_rent'].quantile(0.25)
lower_limit = cost['Office_rent'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Office_rent'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Infrastructure_cost'].quantile(0.75) - cost['Infrastructure_cost'].quantile(0.25)
lower_limit = cost['Infrastructure_cost'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Infrastructure_cost'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Electricitybill'].quantile(0.75) - cost['Electricitybill'].quantile(0.25)
lower_limit = cost['Electricitybill'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Electricitybill'].quantile(0.75) + (IQR * 1.5)

IQR = cost['Course_duration'].quantile(0.75) - cost['Course_duration'].quantile(0.25)
lower_limit = cost['Course_duration'].quantile(0.25) - (IQR * 1.5)
upper_limit = cost['Course_duration'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_cost = np.where(cost['Study_material_price'] > upper_limit, True, np.where(cost['Study_material_price'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

outliers_cost = np.where(cost['Price_of_course'] > upper_limit, True, np.where(cost['Price_of_course'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

outliers_cost = np.where(cost['Enrolled_and_users'] > upper_limit, True, np.where(cost['Enrolled_and_users'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

outliers_cost = np.where(cost['Office_rent'] > upper_limit, True, np.where(cost['Office_rent'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

outliers_cost = np.where(cost['Electricitybill'] > upper_limit, True, np.where(cost['Electricitybill'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

outliers_cost = np.where(cost['Infrastructure_cost'] > upper_limit, True, np.where(cost['Infrastructure_cost'] < lower_limit, True, False))
cost_trimmed = cost.loc[~(outliers_cost), ]
cost.shape, cost_trimmed.shape

sns.boxplot(cost_trimmed.Study_material_price)
sns.boxplot(cost_trimmed.Price_of_course)
sns.boxplot(cost_trimmed.Enrolled_and_users)
sns.boxplot(cost_trimmed.Office_rent)
sns.boxplot(cost_trimmed.Electricitybill)
sns.boxplot(cost_trimmed.Course_duration)


# Scatterplot 
plt.scatter(x = cost.Price_of_course, y = cost.Study_material_price, color = 'green')

# Correlation
np.corrcoef(cost.Price_of_course, cost.Study_material_price)

# Covariance
cov_output = np.cov(cost.Price_of_course, cost.Study_material_price)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple linear regression
model = smf.ols('Study_material_price ~ Price_of_course', data = cost).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(cost['Price_of_course']))

# Regression Line
plt.scatter(cost.Study_material_price, cost.Price_of_course)
plt.plot(cost.Study_material_price, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = cost.Price_of_course - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(Price_of_course); y = Study_material_price

# Scatterplot
plt.scatter(x=np.log(cost['Price_of_course']), y=cost.Study_material_price, color='blue')

# Correlation
np.corrcoef(np.log(cost.Price_of_course), cost.Study_material_price)

model2 = smf.ols('cost.Price_of_course ~ np.log(cost.Study_material_price)', data = cost).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(cost['Price_of_course']))

# Regression Line
plt.scatter(np.log(cost.Study_material_price), cost.Price_of_course)
plt.plot(np.log(cost.Price_of_course), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = cost.Study_material_price - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = cost.Price_of_course, y = np.log(cost['Study_material_price']), color = 'orange')
np.corrcoef(cost.Price_of_course, np.log(cost.Study_material_price)) #correlation

model3 = smf.ols('np.log(Price_of_course) ~ Study_material_price', data = cost).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(cost['Study_material_price']))
pred3_Price_of_course = np.exp(pred3)
pred3_Price_of_course

# Regression Line
plt.scatter(cost.Price_of_course, np.log(cost.Study_material_price))
plt.plot(cost.Price_of_course, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = cost.Price_of_course - pred3_Price_of_course
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation
# x = Study_material_price; x^2 = Enrolled_and_users*Enrolled_and_users; y = log(Price_of_course)

model4 = smf.ols('np.log(Price_of_course) ~ Study_material_price + I(Study_material_price*Study_material_price)', data = cost).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(cost))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cost.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = cost.iloc[:, 1].values

plt.scatter(cost.Price_of_course, np.log(cost.Study_material_price))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = cost.Study_material_price - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(cost)
table_rmse

# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(cost, test_size = 0.2)

finalmodel = smf.ols('np.log(Price_of_course) ~ Study_material_price + I(Study_material_price*Study_material_price)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Price_of_course = np.exp(test_pred)
pred_test_Price_of_course

# Model Evaluation on Test data
test_res = test.Price_of_course - pred_test_Price_of_course
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Price_of_course = np.exp(train_pred)
pred_train_Price_of_course

# Model Evaluation on train data
train_res = train.Price_of_course - pred_train_Price_of_course
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
import pandas as pd
import numpy as np
cost = pd.read_excel("C:\Data science\8888\Project Dataset\cost_and_price.xlsx")
cost.head()
cost.columns
cost.shape

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Study_material_price
plt.bar(height = cost.Study_material_price, x = np.arange(1, 323, 1))
plt.hist(cost.Study_material_price) #histogram
plt.boxplot(cost.Study_material_price) #boxplot

# Price_of_course
plt.bar(height = cost.Price_of_course, x = np.arange(1, 323, 1))
plt.hist(cost.Price_of_course) #histogram
plt.boxplot(cost.Price_of_course) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cost['Study_material_price'], y=cost['Price_of_course'])

# Countplot
plt.figure(1, figsize=(322, 14))
sns.countplot(cost['Study_material_price'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cost.Price_of_course, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cost.iloc[:, :])
                             
# Correlation matrix 
cost.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price_of_course ~ Enrolled_and_users + Study_material_price + Office_rent + Infrastructure_cost + Electricitybill + Course_duration', data = cost).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cost_new = cost.drop(cost.index[[200]])

# Preparing model                  
ml_new = smf.ols('Price_of_course ~ Enrolled_and_users + Study_material_price + Office_rent + Infrastructure_cost + Course_duration + Electricitybill', data = cost_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Study_material_price = smf.ols('Study_material_price  ~ Enrolled_and_users + Office_rent + Infrastructure_cost + Course_duration + Price_of_course', data = cost).fit().rsquared  
vif_Study_material_price = 1/(1 - rsq_Study_material_price) 

rsq_Enrolled_and_users = smf.ols('Enrolled_and_users ~ Study_material_price + Office_rent + Infrastructure_cost + Course_duration + Price_of_course', data = cost).fit().rsquared  
vif_Enrolled_and_users = 1/(1 - rsq_Enrolled_and_users)

rsq_Infrastructure_cost = smf.ols('Infrastructure_cost ~ Study_material_price + Enrolled_and_users + Office_rent + Course_duration + Price_of_course', data = cost).fit().rsquared  
vif_Infrastructure_cost = 1/(1 - rsq_Infrastructure_cost) 

rsq_Office_rent = smf.ols('Office_rent ~ Study_material_price + Enrolled_and_users + Infrastructure_cost + Course_duration + Price_of_course', data = cost).fit().rsquared  
vif_Office_rent = 1/(1 - rsq_Office_rent) 

rsq_Price_of_course = smf.ols('Price_of_course ~ Study_material_price + Enrolled_and_users + Infrastructure_cost + Course_duration + Office_rent', data = cost).fit().rsquared  
vif_Price_of_course = 1/(1 - rsq_Price_of_course) 

# Storing vif values in a data frame
d1 = {'Variables':['Study_material_price', 'Enrolled_and_users', 'Infrastructure_cost', 'Office_rent', 'Price_of_course'], 'VIF':['vif_Study_material_price', 'vif_Enrolled_and_users', 'vif_Infrastructure_cost', 'vif_Office_rent', 'vif_Price_of_course']}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Study_material_price ~ Enrolled_and_users + Infrastructure_cost + Office_rent + Price_of_course', data = cost).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cost)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cost['Price_of_course'], lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cost_train, cost_test = train_test_split(cost, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Study_material_price ~ Enrolled_and_users + Infrastructure_cost + Office_rent + Price_of_course", data = cost_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cost_test)

# test residual values 
test_resid = test_pred - cost_test.Price_of_course
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cost_train)

# train residual values 
train_resid  = train_pred - cost_train.Price_of_course
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
