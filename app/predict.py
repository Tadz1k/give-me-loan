from imp import load_module
import pickle
import pandas as pd
import tools
from pycaret.classification import load_model 
from pycaret.classification import * 

def predict(data_path):
    drop_list = ['Loan_ID']
    text_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    categories = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
    
    model_path_name = tools.get_best_model_path_name()
    
    if model_path_name is None : return 'Brak modelu'
    
    df = tools.change_column_types(pd.read_csv(data_path, sep=',', index_col=False), text_columns, categories)
    df = tools.drop_unnecesary_columns(df, drop_list)
    #df = df.fillna(value=df.mean())

    #pickled_model = pickle.load(open(f'models/{model_path_name}', 'rb'))
    model_path_name = model_path_name.replace('.pkl', '')
    saved_model = load_model(f'models/{model_path_name}')
    exp_clf101 = setup(data = df, target = 'Loan_Status', use_gpu=False, silent=True)
    print(saved_model)
    result = predict_model(saved_model, data=df)
    #result = pickled_model.predict(df)
    print(result)
