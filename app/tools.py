import os
import shutil
from numpy import dtype
import pandas as pd
from plotly import data

def get_current_model_info():
    for model_name in os.listdir('models'):
        if '.pkl' in model_name:
            model_attributes = model_name.split('_')
            number = model_attributes[1]
            accuracy = model_attributes[2]
            return {'number': number, 'accuracy': accuracy, 'name': model_name}
    return None

def detect_drift(new_accuracy, old_model_info):
    if int(new_accuracy) >= int(old_model_info.get('accuracy')):
        return False
    else:
        return True

def write_training_log_info(info):
    log_file = open('training_logs.log', 'a')
    log_file.write(info)
    log_file.write('\n')
    log_file.close()

def move_old_model(modelname):
    shutil.move(f'models/{modelname}', f'models/old/{modelname}')
    
def add_some_trash_data(dataset):
    values = [[1,1,1,0,0,4583,1508,128,360,1,0,1], [0,0,0,0,1,4583,0,133,360,0,1,0]]
    pandas_frame = pd.DataFrame(values, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status'])
    new_dataset = dataset.append(pandas_frame, ignore_index=True)
    return new_dataset

def first_dataset():
    if len(os.listdir('dataset')) == 0:
        return True
    else:
        return False

def remove_old_dataset():
    os.remove('dataset/data.csv')

def get_best_model_path_name():
    items = os.listdir('models')
    if len(items) <= 1:
        return None
    else:
        for item in items:
            if '.pkl' in item:
                return item

def change_column_types(df, text_columns, categories):
    dataframe = df
    #drop pustych text_row !!!!!!!!!!!! columns na text_columns
    for column in categories:
        dataframe.dropna(subset=[column], inplace=True)
    #zmiana text_column na unique id
    for text_column in text_columns:
        dataframe[text_column] = df.groupby(text_column).ngroup()
    #dataframe.mean(skipna=True)
    dataframe.dropna(inplace=True)
    dataframe.isnull().sum()
    dataframe.corr()
    return dataframe

def drop_unnecesary_columns(df, columns):
    dataframe = df
    for item in columns:
        dataframe.drop(item,axis=1,inplace=True)
    return dataframe