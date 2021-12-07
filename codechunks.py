#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew


# # read data

# In[ ]:


#read csv
df = pd.read_csv("filename.csv", encoding='utf-8')

#read excel
df = pd.read_csv("filename.csv", encoding='unicode_escape')

#read textfile to csv
df = pd.read_csv("filename.csv", encoding='unicode_escape')
df.to_csv(“file path”, index = None)


# # data checks

# In[ ]:


#initial data checks
print("# of features:", df.shape[1])
print("# of datapoints:", df.shape[0])
df.head()


# # data statistics

# In[ ]:


df.describe()


df.describe(include='object')



corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
df.info()


# In[ ]:


def performDowncast(df):

  """
  Function tpo apply downcasting 
  
  Input: A dataframe 

  Output: Dataframe with reduced memory usage
  
  
  """
  cols = df.dtypes.index.tolist()
  types = df.dtypes.values.tolist()

  for idx,dtype in enumerate(types):

      if 'int' in str(dtype):  # Downcasting for Int type variables
          if df[cols[idx]].min() > np.iinfo(np.int8).min and df[cols[idx]].max() < np.iinfo(np.int8).max:
              df[cols[idx]] = df[cols[idx]].astype(np.int8) # Downsize to int8
          elif df[cols[idx]].min() > np.iinfo(np.int16).min and df[cols[idx]].max() < np.iinfo(np.int16).max:
              df[cols[idx]] = df[cols[idx]].astype(np.int16) # Downsize to int16
          elif df[cols[idx]].min() > np.iinfo(np.int32).min and df[cols[idx]].max() < np.iinfo(np.int32).max:
              df[cols[idx]] = df[cols[idx]].astype(np.int32) # Downsize to int32
          else:
              df[cols[idx]] = df[cols[idx]].astype(np.int64) # Downsize as int64

      elif 'float' in str(dtype): # Downcasting for Float type variables
          if df[cols[idx]].min() > np.finfo(np.float16).min and df[cols[idx]].max() < np.finfo(np.float16).max:
              df[cols[idx]] = df[cols[idx]].astype(np.float16) # Downsize to float16
          elif df[cols[idx]].min() > np.finfo(np.float32).min and df[cols[idx]].max() < np.finfo(np.float32).max:
              df[cols[idx]] = df[cols[idx]].astype(np.float32) # Downsize to float32
          else:
              df[cols[idx]] = df[cols[idx]].astype(np.float64) # Downsize to float64

      elif dtype == np.object: # By default strings are treated as objects
          if cols[idx] == 'date':
              df[cols[idx]] = pd.to_datetime(df[cols[idx]], format='%Y-%m-%d')
          # else:
          #     df[cols[idx]] = df[cols[idx]].astype('category')

  return df


# # Get correlation values sorted

# In[ ]:


corr_mat = data[numerical_features].corr()
corr_mat = corr_mat.unstack()
corr_mat = corr_mat.sort_values(kind="quicksort").drop_duplicates()
corr_mat[corr_mat>0.5]


# # VIF for cheching multicollinearity

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculating VIF
vif = pd.DataFrame()
df = data.dropna()
vif["variables"] = [feature for feature in numerical_features if feature not in ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT', 'PURCHASES_TRX',
                                                                                'BALANCE', 'PURCHASES_FREQUENCY', 'PAYMENTS',
                                                                                'PURCHASES', 'CASH_ADVANCE_TRX', 'BALANCE_FREQUENCY',
                                                                                'INSTALLMENTS_PURCHASES']]
vif["VIF"] = [variance_inflation_factor(df[vif['variables']].values, i) for i in range(len(vif["variables"]))]
print(vif)


# In[ ]:


df.select_dtypes(include=['object'])

df.select_dtypes(include=['int', 'float'])

df.select_dtypes(exclude=['object'])


# # checking null values in percentage and filling null values

# In[ ]:


round(df.isnull().sum()/df.shape[0]*100,2)

df['Description'] = df['Description'].fillna("None")
df = df.dropna()


# # Checking the number of categories in a feature in dataframe

# In[ ]:


summarystats_categorical_feat = []
categorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']
summarystats_categorical_feat = [[feature, len(df[feature].unique())] for feature in categorical_feature]
data = pd.DataFrame(summarystats_categorical_feat, columns=["Features", "Unique Categories"])


# # Data profile

# In[ ]:


def data_profile(df):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=True)
    return stats_df


# # checking distribution of the complete dataset

# In[ ]:


df.hist(figsize=(14,14), xrot=45)
plt.show()


# # Merging daraset

# In[ ]:


# Joining datasets

df = data1.merge(data2, how= 'inner', on = "id")


# # datetime

# In[ ]:


#changing the date type into datetime

df['date'] =  pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


# # check skewness in a feature

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew
numeric_feats = data.dtypes[df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# # plot distribution

# In[ ]:


sns.distplot(df['target'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['target'] )
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')


#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df['target'], plot=plt)
plt.show()


# # Group by aggregation

# In[ ]:


df.groupby('A').agg(['min', 'max'])

temp1 = pd.DataFrame(df.groupby('column').agg(['sum'])['col'])
temp1.head()

data.groupby(['month']).groups.keys()
data.groupby('month').first()
data.groupby('month')['duration'].sum()
data.groupby('month')['date'].count()
data.groupby('month', as_index=False).agg({"duration": "sum"})

data.groupby(
   ['month', 'item']
).agg(
    {
         'duration':sum,    # Sum duration per group
         'network_type': "count",  # get the count of networks
         'date': 'first'  # get the first date per group
    }
)


# In[ ]:


grouped_multiple = df.groupby(['Team', 'Pos']).agg({'Age': ['mean', 'min', 'max']})
grouped_multiple.columns = ['age_mean', 'age_min', 'age_max']
grouped_multiple = grouped_multiple.reset_index()
print(grouped_multiple)


# # Exploratory Data Analysis
# 
# # Bar plot with value counts

# In[ ]:


df_1 = pd.DataFrame(df['day_of_month'].value_counts())
fig, ax = plt.subplots(figsize=(12,12))
graph = sns.barplot(x=df_1.index, y=df_1.day_of_month, palette="Set2")
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center", rotation = 90)
ax.set(xlabel='xlabel', ylabel='count', title='')


# # Pie chart

# In[ ]:


labels = 'unsuccessful', 'successful'
sizes = [df.Y[data['Y']==0].count(),          df.Y[data['Y']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['lightgrey', 'lightblue'])
ax1.axis('equal')
plt.title("", size = 20)
plt.show()


# # Boxplot

# In[ ]:


plt.figure(figsize=(16,8))  
ax = sns.boxplot(x="discrete", y="continous", hue=" ", data=df, palette="Set3")


# In[ ]:



#countplot

plt.figure(figsize=(14, 6))

sns.countplot(x='X7', hue = 'Y',data = df,               palette=['limegreen', 'darkblue'])
plt.xlabel("")
plt.legend(['call unsuccessful', 'successful'], loc='upper right')
plt.ylabel("")
plt.title("")


# # barplot 

# In[ ]:



data1= data.reindex(index=order_by_index(df.index, index_natsorted(df['column'])))
fig, ax = plt.subplots(figsize=(14,8))
graph = sns.countplot(ax= ax, x='X8', hue = 'Y',data = data1,               palette=['lightgrey', 'blue'])
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha="center")

bars = graph.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    ax.text(left.get_x() + left.get_width()/2., height_l - 550, '{0:.0%}'.format(height_l/total), ha="center", color = "blue")
    ax.text(right.get_x() + right.get_width()/2., height_r - 550, '{0:.0%}'.format(height_r/total), ha="center",  color = "white") 

plt.ylabel("Count")
plt.title("")


# # Pair plot

# In[ ]:


sns.set(style="ticks")
sns.pairplot(df[["Sell","Taxes","Acres","Beds"]], hue="Beds")


# # scatter plot

# In[ ]:



sns.scatterplot(continous, continous, hue=discrete, style=discrete)
plt.show()


# # Outlier Detection:

# In[ ]:


attributes = []
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = df[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')

# Removing outlier using IQR

# Removing (statistical) outliers for Amount
Q1 = df.Amount.quantile(0.05)
Q3 = df.Amount.quantile(0.95)
IQR = Q3 - Q1
Df = df[(df.Amount >= Q1 - 1.5*IQR) & (df.Amount <= Q3 + 1.5*IQR)]


# In[ ]:


#Capping the outlier rows with Percentiles
upper_lim = df['column'].quantile(.95)
lower_lim = df['column'].quantile(.05)
df.loc[(df[column] > upper_lim),column] = upper_lim
df.loc[(df[column] < lower_lim),column] = lower_lim


# # Binning 

# In[ ]:


#Numerical Binning Example
data['bin'] = pd.cut(df['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])


# In[ ]:


conditions = [
    df['Country'].str.contains('Spain'),
    df['Country'].str.contains('Italy'),
    df['Country'].str.contains('Chile'),
    df['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')


# # transforms
# 
# ## log

# In[ ]:


df['log+1'] = (df['value']+1).transform(np.log)
#Negative Values Handling
#Note that the values are different
df['log'] = (df['value']-df['value'].min()+1) .transform(np.log)


# 
# Once, we know the skewness level, we should know whether it is positively skewed or negatively skewed.
# 
# **Positively skewed data:
# If tail is on the right, it is right skewed data. It is also called positive skewed data.**
# 
# Common transformations of this data include **square root, cube root, and log**
# 
# **Cube root transformation:**
# The cube root transformation involves converting x to x^(1/3). This is a fairly strong transformation with a substantial effect on distribution shape: but is weaker than the logarithm. It can be applied to negative and zero values too. Negatively skewed data.
# 
# **Square root transformation:**
# Applied to positive values only. Hence, observe the values of column before applying.
# 
# **Logarithm transformation:**
# The logarithm, x to log base 10 of x, or x to log base e of x (ln x), or x to log base 2 of x, is a strong transformation and can be used to reduce right skewness.
# 
# **Negatively skewed data:
# If the tail is to the left of data, then it is called left skewed data. It is also called negatively skewed data.
# Common transformations include square , cube root and logarithmic.**
# 
# 
# **Square transformation:**
# The square, x to x², has a moderate effect on distribution shape and it could be used to reduce left skewness.
# Another method of handling skewness is finding outliers and possibly removing them.

# # root - left skewed tail is on left
# 

# In[ ]:


for col in ["X1","X2", "X3", "X4","X6", "X7"]:
    df[col] = np.cbrt(df[col])

df["X10"] = np.sqrt(df["X10"])


# # Box cox transformation not applicable for negative data
# 
# ![](Boxcox.PNG)
# 
# ![](Boxcox1.PNG)

# In[ ]:


from scipy.stats import boxcox

# Box-Cox Transformation in Python
df.insert(len(df.columns), 'A_Boxcox', 
              boxcox(df.iloc[:, 0])[0])


# # Grouping

# In[ ]:


df.groupby('id').agg(lambda x: x.value_counts().index[0])
#Pivot table Pandas Example
df.pivot_table(index='column_to_group', columns='column_to_encode', values='aggregation_column', aggfunc=np.sum, fill_value = 0)


# In[ ]:


#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = data.groupby('column_to_group')

sums = grouped[sum_cols].sum().add_suffix('_sum')
avgs = grouped[mean_cols].mean().add_suffix('_avg')

new_df = pd.concat([sums, avgs], axis=1)


# # Feature split 

# In[ ]:


data.name.str.split(" ").map(lambda x: x[0])


# # Scaling – standard scaling (mean = 0 , sd = 1)

# In[ ]:


# Instantiate
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit_transform
df_scaled = scaler.fit_transform(df)
df_scaled.shape


# # Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 

# In[ ]:


data['normalized'] = (data['value'] - data['value'].min()) / (data['value'].max() - data['value'].min())


# # model imports

# In[3]:


#Regression

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# # Regression

# In[ ]:


from sklearn.model_selection import train_test_split

y = data.loc[:,'charges']
df1 = data.drop(['charges','region'], axis = 1)
X = df1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape,y_train.shape)
print(X_val.shape, y_val.shape)
X_train = X_train.reset_index(drop =True)
y_train = y_train.reset_index(drop = True)
X_val = X_val.reset_index(drop = True)
y_val = y_val.reset_index(drop = True)


# # Define a cross validation strategy
# 
# We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation

# In[ ]:


import sklearn.metrics as metrics
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    
    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    
    results = pd.DataFrame({'EV': [round(explained_variance,4)], 'r2':[round(r2,4)], 'MAE': [round(mean_absolute_error,4)],'mse': [round(mse,4)], 'rmse': [round(np.sqrt(mse),4)]})
    return results


# # Base models
# 

# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train,y_train )
pred = reg.predict(X_val)
lr = regression_results(y_val,pred )


# # OLS 

# In[ ]:


from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()


# # Decision Tree Regression

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model_name = "Decision Tree Regression"
dt = DecisionTreeRegressor(random_state = 3, max_features = 5)
dt.fit(X_train, y_train)
dt_y_pred =dt.predict(X_val)
dtr= regression_results(y_val,dt_y_pred )


# # Random Forest Regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators = 10, max_depth = 5, random_state = 3)
RF.fit(X_train, y_train)
RF_y_pred =RF.predict(X_val)
rf = regression_results(y_val,RF_y_pred )
plt.barh(X_train.columns, RF.feature_importances_)


# #  LASSO Regression :
# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline

# In[ ]:


from sklearn.pipeline import make_pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# # Polynomial Regression

# In[ ]:


poly = PolynomialFeatures(degree = 3)
plreg = LinearRegression()

plr_model = Pipeline(steps = [('plotfeat',poly), ('regressor', plreg)])
plr_model.fit(X_train, y_train)
y_pred =plr_model.predict(X_val)
plr = regression_results(y_val,y_pred )


# # Elastic Net Regression :
# again made robust to outliers

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# # Kernel Ridge Regression :

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# # Gradient Boosting Regression :
# With huber loss that makes it robust to outliers

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# # XGBoost :

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# # LightGBM :

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# # Score table

# In[ ]:


### data = pd.concat([lr, plr, dtr, rf]).reset_index()
data.index = ['LR', 'PLR', 'DTR', 'RF']
data.drop(['index'], axis =1 )


graph = sns.barplot(x=data.index, y=data.EV, palette="Set2")
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center", rotation = 90)
ax.set(xlabel='xlabel', ylabel='count', title='')


# In[ ]:





# In[ ]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20 , 30],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [50, 100]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[ ]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_val, y_val)


# # Classification

# In[ ]:


#classification
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.metrics import average_precision_score
from sklearn.ensemble import AdaBoostClassifier as ab
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier as sg
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import numpy as np


# ## One encoding
# 
# One hot encoding for categorical features:¶
# Our models cannot understand categorical featues so we need to convert categorical features into continous features. I used One-hot encoding to convert the categorical features in the form of '0's and '1's respectively.

# In[ ]:


#one hot encoding
df = pd.get_dummies(df, columns=['columns'])
df.head()


# # Label encoding

# In[ ]:


#label encoding

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
data['col'] = labelencoder.fit_transform(data['col'])


# 
# ## train test split

# In[ ]:


y = df.loc[:,'label']
df1 = df.drop(['label'], axis = 1)
X = df1
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify= y, test_size=0.20, random_state=42)
print(X_train.shape,y_train.shape)
print(X_val.shape, y_val.shape)
X_train = X_train.reset_index(drop =True)
y_train = y_train.reset_index(drop = True)
X_val = X_val.reset_index(drop = True)
y_val = y_val.reset_index(drop = True)


# # Cross validation score

# In[ ]:


model_name=['naive_bayes','LogisticRegression','RandomForestClassifier','KNeighborsClassifier','GradientBoostingClassifier','AdaBoostClassifier','DecisionTreeClassifier']
models_list= [NB(),LR(max_iter=600),RF(),KNN(),GB(),ab(),dt()]
for i, j in zip(model_name, models_list):
    scores = cross_validate(j, X_train, y_train, cv=5)
    print(i+"--"+ "Accuracy: %0.2f (+/- %0.2f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))


# In[ ]:


def plot_curve(train_scores,test_scores,train_sizes,fig_name,title,xlabel,ylabel):
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)


  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)


  plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
  plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")


  plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
  plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

  plt.title(title)
  plt.xlabel(xlabel), 
  plt.ylabel(ylabel), 
  plt.legend(loc="best")
  #plt.tight_layout()
  #plt.savefig(fig_name, bbox_inches='tight')


# # Logistic Regression

# In[ ]:


# fit a logistic regression model to the data
def logisticRegression(train_x, train_y,test_x, test_y):
  model = LogisticRegression(max_iter = 1200, random_state= 42, n_jobs = -1)
  model.fit(train_x, train_y)
  print(model)
  # make predictions
  expected = test_y
  predicted = model.predict(test_x)
  # summarize the fit of the model
  print(classification_report(expected, predicted))
  print(confusion_matrix(expected, predicted))
  disp = plot_confusion_matrix(model, test_x, test_y,
                                 cmap=plt.cm.Blues)
  disp.ax_.set_title("Confusion Matrix")
  plt.figure(figsize=(10,6))
  sns.set(font_scale=1.4)
  print("Confusion Matrix")
  plt.show()


# In[ ]:


print(logisticRegression(X_train, y_train, X_val, y_val))


# # Random forest

# In[ ]:


def randomForest(train_x, train_y,test_x, test_y):
  clf = RandomForestClassifier(n_estimators = 10, max_depth =35, random_state=0)
  clf.fit(train_x,train_y)
  print(classification_report(test_y, clf.predict(test_x)))
  print(confusion_matrix(test_y, clf.predict(test_x)))
  disp = plot_confusion_matrix(clf, test_x, test_y,
                                 cmap=plt.cm.Blues)
  disp.ax_.set_title("Confusion Matrix")
  plt.figure(figsize=(10,6))
  sns.set(font_scale=1.4)
  print("Confusion Matrix")
  plt.show()


# In[ ]:


print(randomForest(X_train, y_train, X_val, y_val))


# # Decision Tree classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
def randomForest(train_x, train_y,test_x, test_y):
  clf = DecisionTreeClassifier(max_depth =3, random_state = 42)
  clf.fit(X_train, y_train)
  print(classification_report(test_y, clf.predict(test_x)))
  print(confusion_matrix(test_y, clf.predict(test_x)))
  disp = plot_confusion_matrix(clf, test_x, test_y,
                                 cmap=plt.cm.Blues)
  disp.ax_.set_title("Confusion Matrix")
  plt.figure(figsize=(10,6))
  sns.set(font_scale=1.4)
  print("Confusion Matrix")
  plt.show()


# # Adaboost Classifier

# In[ ]:


def AdaBoost(X_train,y_train,X_val,y_val):
  classifier = AdaBoostClassifier(
      DecisionTreeClassifier(max_depth=1),
      n_estimators=200
  )
  classifier.fit(X_train, y_train)

  predictions = classifier.predict(X_val)
  print(classification_report(y_val, predictions))
  print(confusion_matrix(y_val, predictions))
  disp = plot_confusion_matrix(classifier, X_val, y_val,
                                  cmap=plt.cm.Blues, normalize = 'true')
  disp.ax_.set_title("Confusion Matrix")
  print("Confusion Matrix")
  plt.show()


# # SMOTE Oversampling

# In[ ]:


X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
counter = Counter(y_resampled)
print(counter)


# # Hyper parameter tuning using grid search Decision Tree classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'max_depth': [1,2,3,4,5], 
                     'min_samples_split': [2,4,6,8,10]}]
scores = ['recall']
for score in scores:
    
    print(f"Tuning hyperparameters for {score}")
    
    clf = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters,
        scoring = f'{score}_macro'
    )
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds,
                                 clf.cv_results_['params']):
        print(f"{mean:0.3f} (+/-{std*2:0.03f}) for {params}")


# # Hyper parameter tuning using grid search RF classifier

# In[ ]:


n_estimators = [50, 100]
max_depth = [5, 8, 15]
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

forest = RandomForestClassifier(random_state = 1)
gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)


# In[ ]:


forestOpt = RandomForestClassifier(random_state = 1, max_depth = 15,     n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1)
                                   
modelOpt = forestOpt.fit(x_train, y_train)
y_pred = modelOpt.predict(x_test)

