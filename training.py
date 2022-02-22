  

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()



df.info()

"""### The target the we will use to guide the exploration is **Churn**"""

#Checking for the null values

df.isnull().sum()

#We don't have any null values in our dataset

"""## Data Manipulation"""

#Lets drop the customer ID column
df = df.drop(["customerID"], axis = "columns")

df.columns

# Total charges column is in object type.. lets convert it into int type

df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors='coerce')

df.isnull().sum()

df[np.isnan(df["TotalCharges"])]

"""* It can also be noted that the Tenure column is 0 for these entries even though the MonthlyCharges column is not empty.

Let's see if there are any other 0 values in the tenure column.
"""

df[df["tenure"]==0].index

"""* There are no additional missing values in the Tenure column. 

Let's delete the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.
"""

df.drop(df[df["tenure"]==0].index, axis ="rows", inplace = True )

df[df["tenure"]==0].index

#Rows have been deleted

df["SeniorCitizen"] = df["SeniorCitizen"].map({0:"No", 1:"Yes"})

df["SeniorCitizen"].head()

df["InternetService"].describe()



df.columns

"""# <span style="font-family:serif; font-size:28px;"> Checking outliers </span>"""

mean = np.mean(df.MonthlyCharges)
std = np.std(df.MonthlyCharges)
print('mean of the MonthlyCharges is', mean)
print('std. deviation is', std)

threshold = 3
outlier = []
for i in df.MonthlyCharges:
    z = (i-mean)/std
    if z > threshold:
        outlier.append(i)
print('outlier in dataset is', outlier)

mean = np.mean(df.TotalCharges)
std = np.std(df.TotalCharges)
print('mean of the TotalCharges is', mean)
print('std. deviation is', std)

threshold = 3
outlier = []
for i in df.TotalCharges:
    z = (i-mean)/std
    if z > threshold:
        outlier.append(i)
print('outlier in dataset is', outlier)

mean = np.mean(df.tenure)
std = np.std(df.tenure)
print('mean of the tenure is', mean)
print('std. deviation is', std)

threshold = 3
outlier = []
for i in df.tenure:
    z = (i-mean)/std
    if z > threshold:
        outlier.append(i)
print('outlier in dataset is', outlier)

# IQR method

x = ['tenure','MonthlyCharges']
def count_outliers(df,col):
        q1 = df[col].quantile(0.25,interpolation='nearest')
        q2 = df[col].quantile(0.5,interpolation='nearest')
        q3 = df[col].quantile(0.75,interpolation='nearest')
        q4 = df[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if df[col].min() > LLP and df[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = df[df[col]<LLP][col].size
            y = df[df[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x:
    count_outliers(df,i)


df.head()

df['churn_rate'] = df['Churn'].replace("No", 0).replace("Yes", 1)

df.drop(["Churn"], axis = "columns", inplace = True)

df1 = pd.get_dummies(data=df, columns = ["SeniorCitizen",'gender', 'Partner', 'Dependents', 
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'],drop_first=True)

df1.head()

df1 = df1[['SeniorCitizen_Yes', 'tenure', 'MonthlyCharges', 'TotalCharges','gender_Male', 'Partner_Yes', 'Dependents_Yes',\
       'PhoneService_Yes', 'MultipleLines_No phone service',\
       'MultipleLines_Yes', 'InternetService_Fiber optic',\
       'InternetService_No', 'OnlineSecurity_No internet service',\
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',\
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',\
       'DeviceProtection_Yes', 'TechSupport_No internet service',\
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',\
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',\
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',\
       'PaymentMethod_Credit card (automatic)',\
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','churn_rate']]

df1 = df1.astype(int)  
for i in df1.columns:
    df1[i]= df1[i].astype(str).astype(int)
    print(df1.dtypes)



x = df1.drop(['churn_rate'], axis = 1)
y = df1['churn_rate']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()  
df2=sc.fit_transform(x)
df_scaled = pd.DataFrame(df2 ,columns = df1.columns[:-1])

df_scaled.head()


from sklearn.ensemble import RandomForestClassifier
cl_rf = RandomForestClassifier(bootstrap= True, criterion= 'entropy', max_depth= 9, max_features= 'sqrt', n_estimators= 200)
cl_rf.fit(x_train,y_train)
y_pred2 = cl_rf.predict(x_test)

print(classification_report(y_test,y_pred2))
print("Training score: ", cl_rf.score(x_train, y_train))
print("Testing score: ", cl_rf.score(x_test, y_test))




#saving the model 
import joblib
joblib.dump(cl_rf,'churn_80.pkl')






