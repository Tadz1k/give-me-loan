from numpy import True_
import pandas as pd
import logging
import pycaret
import pickle
from pycaret.classification import *
from datetime import date
import tools
import sys

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def train(dataset_path, drift_trigger, post_training = False):
    drop_list = ['Loan_ID']
    text_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    categories = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
    #Add new data to existing dataset
    if post_training:
        if tools.first_dataset(): return False
        additional_dataset = tools.change_column_types(pd.read_csv(dataset_path, sep=',', index_col=False), text_columns, categories)
        additional_dataset = additional_dataset.iloc[1: , :]
        dataset = pd.read_csv('dataset/data.csv', sep=',', index_col=False)
        frames = [dataset, additional_dataset]
        dataset = pd.concat(frames)
    #New training
    else:
        dataset = tools.change_column_types(pd.read_csv(dataset_path, sep=',', index_col=False), text_columns, categories)


    dataset = tools.drop_unnecesary_columns(dataset, drop_list)
    # #If dataset directory is empty - save new dataset to csv file
    if tools.first_dataset():
        dataset.to_csv('dataset/data.csv', index=False)
    if drift_trigger:
        dataset = tools.add_some_trash_data(dataset)
        exp1 = setup(data=dataset, target='Loan_Status', session_id=121, experiment_name='Test', normalize=True, transformation=True, log_experiment = True, log_plots = True, silent=True)

    if not drift_trigger:
        exp1 = setup(data=dataset, target='Loan_Status', session_id=121, experiment_name='Test', silent=True, normalize=True, transformation=True, log_experiment = True, log_plots = True, n_jobs=1)


    best = compare_models()
    new_model = create_model(best)
    tuned_model = tune_model(new_model)
    final = finalize_model(tuned_model)
    result = predict_model(final)
    result_frame = pull()
    print(result_frame)
    accuracy = result_frame.at[0, 'Accuracy']
    accuracy = int(accuracy*1000)

    current_model_info = tools.get_current_model_info()

    today = date.today()
    datetime_string = today.strftime("%Y-%m-%d %H:%M:%S")
    print(dataset.shape)
    #If no models in directory
    if current_model_info is None:
        pickle.dump(final, open(f'models/model_1_{accuracy}_acc.pkl', 'wb'))
        tools.write_training_log_info(f'[{datetime_string}] FIRST MODEL ARRIVED! ACCURACY: {accuracy}')
    #If there is some model in models directory - lets check which is better
    else:
        drift = tools.detect_drift(accuracy, current_model_info)
        if drift:
            tools.write_training_log_info(f'[{datetime_string}] DRIFT DETECTED. OLD ACCURACY : {current_model_info.get("accuracy")} NEW ACCURACY : {accuracy}')
        if not drift:
            new_model_number = int(current_model_info.get('number')) + 1
            pickle.dump(final, open(f'models/model_{new_model_number}_{accuracy}_acc.pkl', 'wb'))
            tools.move_old_model(current_model_info.get('name'))
            tools.write_training_log_info(f'[{datetime_string}] NEW MODEL IN THE FAMILY! ACCURACY : {accuracy}. OLD MODEL "{current_model_info.get("name")}" IS IN "MODELS/OLD"')
            if not tools.first_dataset():
                tools.remove_old_dataset()
                dataset.to_csv('dataset/data.csv', index=False)
