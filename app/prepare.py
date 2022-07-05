import pandas as pd 
import os

drop_list = ['Loan_ID']

dataset = pd.read_csv('data/data_train.csv', sep=',')
for item in drop_list:
    dataset.drop(item,axis=1,inplace=True)
os.remove('data/data_train.csv')
dataset.to_csv('data/data_train.csv')

dataset2 = pd.read_csv('data/additional.csv', sep=',')
for item in drop_list:
    dataset2.drop(item,axis=1,inplace=True)
os.remove('data/additional.csv')
dataset2.to_csv('data/additional.csv')

predict = pd.read_csv('data/predict.csv', sep=',')
for item in drop_list:
    predict.drop(item, axis=1, inplace=True)
predict = predict.iloc[: , :-1]
os.remove('data/predict.csv')
predict.to_csv('data/predict.csv')