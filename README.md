# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

```
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train
```
##PROGRAM:
/Write your code here/
```
import pandas as pd
import numpy as np
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI411 - Neural Networks/Churn_Modelling.csv")
df
df.isnull().sum()
#Check for Duplicate Values
df.duplicated()
df.describe()
#Detect the Outliers
# Outliers are any abnormal values going beyond
df['Exited'].describe()
""" Normalize the data - There are range of values in different columns of x are different. 
To get a correct ne plot the data of x between 0 and 1 
LabelEncoder can be used to normalize labels.
It can also be used to transform non-numerical labels to numerical labels.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
'''
MinMaxScaler - Transform features by scaling each feature to a given range. 
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
# Since values like Row Number, Customer Id and surname  doesn't affect the output y(Exited).
#So those are not considered in the x values
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape
```
##OUTPUT:
/ Show the result/
## Dataset

![image](https://user-images.githubusercontent.com/114219474/191921858-f1a04568-f231-42a8-a36c-787860b78164.png)

## Checking For Null Values

![image](https://user-images.githubusercontent.com/114219474/191923433-8ce78620-453d-4567-9b82-bc2f1dd781f9.png)

## Checking For Duplicate Values

![image](https://user-images.githubusercontent.com/114219474/191923727-e2e367dc-8947-4542-b87c-d1989daf5b4e.png)

## Describing Data

![image](https://user-images.githubusercontent.com/114219474/191923987-a6d6d3f6-bd19-4298-81dd-fb89090aacb0.png)

## Checking For Outliers In Exited Column

![image](https://user-images.githubusercontent.com/114219474/191924239-74c5a86f-acbe-4fe4-a8d4-cc4502ecbc6d.png)

## Normalized Dataset

![image](https://user-images.githubusercontent.com/114219474/191924512-ab1bbbb6-326d-4ef9-bf70-069920d69c14.png)

## Describing Normalized Dataset

![image](https://user-images.githubusercontent.com/114219474/191924819-8b45657d-be37-4125-8c80-cbf7cb1107de.png)

## X-Values

![image](https://user-images.githubusercontent.com/114219474/191925082-e0cf2768-301e-47ac-b280-91bc53504063.png)

## Y-Value

![image](https://user-images.githubusercontent.com/114219474/191925637-78493b0d-4096-4708-b975-546e039b081a.png)

## X_Train Values

![image](https://user-images.githubusercontent.com/114219474/191925962-130b5154-0217-4e56-b0fd-a58eedf773cf.png)

## X_Train Size

![image](https://user-images.githubusercontent.com/114219474/191926201-d5195055-4e0a-4ec8-b281-94543e7d873f.png)

## X_Test Values

![image](https://user-images.githubusercontent.com/114219474/191926671-6e434835-b3cc-4b3d-af69-550f0588615e.png)

## X_Test Size

![image](https://user-images.githubusercontent.com/114219474/191926913-9de9caf0-5541-44f8-a0ec-56a632229a78.png)

## X_Train Shape

![image](https://user-images.githubusercontent.com/114219474/191927135-e519934d-e79f-4314-93a0-19d0294615ad.png)
















##RESULT
/Type your result here/

Data preprocessing is performed in a data set downloaded from Kaggle
0 comments on commit ef9167e
