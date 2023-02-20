import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df=pd.read_csv('loan-train.csv')

df.drop(['Loan_ID'],axis='columns',inplace=True)

df.Gender.fillna('Female',inplace=True)

df.Married.fillna('Yes',inplace=True)

df.Dependents.replace('3+',4,inplace=True)

df.Dependents.dropna(axis='rows',inplace=True)

a=df[df.Dependents.isnull()].index.tolist()

df.drop(a,inplace=True,axis=0)

df.Self_Employed.fillna('No',inplace=True)

df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)

df.Loan_Amount_Term.fillna('360.0',inplace=True)

df.dropna(inplace=True,axis='rows')

LE=LabelEncoder()

df['Gender']=LE.fit_transform(df['Gender'])
df['Married']=LE.fit_transform(df['Married'])
df['Education']=LE.fit_transform(df['Education'])
df['Self_Employed']=LE.fit_transform(df['Self_Employed'])
df['Property_Area']=LE.fit_transform(df['Property_Area'])
df['Loan_Status']=LE.fit_transform(df['Loan_Status'])

x=df.drop(['Loan_Status'],axis='columns').values
y=df.Loan_Status.values

RF=RandomForestClassifier()

RF.fit(x,y)

pickle.dump(RF,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

