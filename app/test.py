import pandas as pd
import os


df = pd.read_csv('data/predict2.csv', sep=',')
print(df.shape)
print(df.columns)
#print(df.info())


#[17.06.2022 22:21] Adrian Kordas
#python main.py --goal predict --data 'data/data_test.csv'

#[17.06.2022 22:22] Adrian Kordas
#python main.py --goal train --drift n --dataset 'data/data_train.csv'

#[17.06.2022 22:22] Adrian Kordas
#python main.py --goal post-training --dataset 'data/additional.csv'

