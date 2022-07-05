![logo](https://i.imgur.com/VYodzpx.png)



# Table of contents

1. [About give me loan](#About-give-me-loan)
2. [About the dataset](#About-the-dataset)
3. [Main components](#Main-components)
4. [Architecture diagram](#Architecture-diagram)
5. [Getting started](#Getting-started)<br>
    5.1. [Python using command line](#Python-using-command-line)<br>
    5.2. [Docker image](#Docker-image)<br>
    5.3. [Arguments](#Arguments)<br>
    5.4. [Arguments overview](#Arguments-overview)<br>
6. [Example usage](#Example-usage)<br>
7. [How it works](#How-it-works)<br>
    7.1. [Data preparation](#Data-preparation)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;7.1.1. [Removal of unnecessary columns](#Removal-of-unnecessary-columns)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;7.1.2. [Data grouping](#Data-grouping)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;7.1.3. [Delete records that contain blank cells](#Delete-records-that-contain-blank-cells)<br>
    7.2. [Training](#Training)<br>
    7.3. [Creating new model](#Creating-new-model)<br>
    7.4. [Post training](#Post-training)<br>
    7.5. [Prediction](#Prediction)<br>
8. [Other systems](#Other-systems)<br>
    8.1. [Drift detection](#Drift-detection)<br>
    8.2. [Log system](#Log-system)<br>
    8.3. [MLFlow Tracking](#MLFlow-tracking)<br>
    8.4. [Docker](#Docker)<br>
9. [PL Documentation](#PL-documentation)
10. [Known bugs](#Known-bugs)
11. [Authors](#Authors)

 
# About give me loan

The main goal of the project is to support bank employees in making decisions about granting a loan. The use of machine learning methods can significantly shorten the process of issuing a credit decision. The project includes the entire pipeline including training, retraining, creating a new model, and performing predictions. The application allows you to train the best machine learning algorithm available in the pycaret library, and to supervise the system. Supervision is carried out using the mlflow component and the built-in log system that presents information about drift detection, detection of a new model, etc.

### About the dataset

The dataset was obtained from the website kaggle.com. It contains important customer data in the context of granting a loan. The data stored in the collection is:

![image](https://user-images.githubusercontent.com/50612974/176903477-41f68c37-65b3-488a-ba58-f2a7f6a543ee.png)

Source of dataset : [CLICK HERE (kaggle.com)]( https://www.kaggle.com/datasets/shaijudatascience/loan-prediction-practice-av-competition?select=train_csv.csv)


### Main components:
* Training model
* Post-training of the model
* Drift detection
* Prediction
* Log system

### Architecture diagram

![image](https://i.imgur.com/gAuNh5F.png)


# Getting started

The application can be launched with a regular command on the command line, and it can be run in a Docker image. Starting the application requires the provision of appropriate parameters on the basis of which the system will launch the appropriate component. Necessary libraries must be installed before using the application. These libraries are used during system operation. The most important of them are:

* pycaret - allows you to quickly train the model with a small amount of code
* mlflow - allows you to track and supervise the training of the model in the system
* pandas - enables the operation of modifying csv files on entry and exit from the application.

### Python using command line

You can use virtual enviroment, or work in your global enviroment. To create and activate venv:

`python -m venv venv`

`venv/scripts/Activate` - for Windows

First, you need to install all the necessary libraries. This can be done manually by installing the libraries mentioned above, or you can download this repository and install all listed dependencies. This can be done with the command:

`pip install -r requirements.txt`

### Docker image

The project contains a Dockerfile that includes the most important elements of the environment. These are the required libraries, an indication of the startup file, and the specification that the startup file should have appropriate parameters.
An image with a docker container can be created with the command:

`docker build -t python-image .`

## Arguments

The operation of the application always starts with the main.py file, but requires additional input arguments depending on the goal you want to achieve.

### Arguments overview
```
goal (train / predict / post-training)
dataset (path to training / post-training csv file)
data (path to the csv file with the data that should be classified)
drift (drift trigger - add some trashy data to decrease accuracy or not y/n)
```

#### Example usage

##### Predict data:

```python main.py --goal predict --data 'data/data_test.csv' ```  
or:  
```docker run python-image --goal predict --data 'data/data_test.csv' ```  
or:  
```docker run - publish 8000:8000 python-image --goal predict --data 'data/data_test.csv' ```  


Command telling the application that the user wants to perform a prediction of the data placed in the `data/data_test.csv` directory.

##### train model:

```python main.py --goal train --drift n --dataset 'data/data_train.csv' ```   
or:  
```docker run python-image --goal train --drift n --dataset 'data/data_train.csv'```  
or:  
```docker run - publish 8000:8000 python-image --goal train --drift n --dataset 'data/data_train.csv'```  

The drift trigger is marked with 'n' that means no artificial data will be added to reduce the quality of the model.

##### post-train model

```python main.py --goal post-training –dataset 'data/additional.csv' ```  
or:  
```docker run python-image --goal post-training –dataset 'data/additional.csv' ```  
or:  
```docker run - publish 8000:8000 python-image --goal post-training –dataset 'data/additional.csv' ```  


the command informs that the user wants to train an existing model. Then, an additional piece of information from the `data/additional.csv` file is added to the current data set.


# How it works

The project structure includes additional folders `models` and `dataset`, in which the models and the dataset file are stored, respectively. In the `dataset` directory there is an additional `old` subdirectory where old models are stored. In the name of the model file, there is an order number and the accuracy factor multiplied by 1000.

![model](https://i.imgur.com/bVS7ney.png)


If there is no data in the models directory and the dataset directory, then we are creating a new model, saving the data and exiting.
However, if there are data in these catalogs, it should be compared whether the new model is better (in this we are helped by the drift detection) and whether it is worth saving this model. If so - the new model is better, then the old model is moved to the 'old' subdirectory, and the new one with the sequence number is saved in the main 'models' directory. In the directory 'dataset' the dataset is replaced.

### Data preparation

The data has been wiped and tidied up. For this purpose, the functions offered by the Pandas package were used. Actions that have been applied to the dataset:

#### Removal of unnecessary columns

At the beginning of data preparation, columns are thrown that do not add any value in the context of training the model. Therefore, the Application ID column `(Loan_ID)` is rejected using the drop function (column, axis = 1). The axis = 1 parameter means that the function is to delete the entire column of data.

#### Data grouping
Many columns contain data that grouped applicants. For example, `gender` (text - Male or Female), `education level` (Graduate or Not Graduate). To improve the model training process, grouping functions were used for specific columns. Corresponding groups are marked with numbers. Therefore, gender is treated as `0 or 1` in the input data set for the training function. The same applies to the other classifying columns. The function `df.groupby(column).ngroup()` was used for this task.

#### Delete records that contain blank cells
None of the information in the application can be averaged (if missing). Data such as income, loan amount, etc. should not be averaged. Therefore, at the design stage, it was decided to discard all records in the dataset where the cells contain blank values. For this task the dropna function was used (inplace = True).


The data preparation functions are the same for each step of the system operation.


Function dropping unnecesary columns:
```
def drop_unnecesary_columns(df, columns):
    dataframe = df
    for item in columns:
        dataframe.drop(item,axis=1,inplace=True)
    return dataframe
```

Function changing data types:
```
def change_column_types(df, text_columns, categories):
    dataframe = df
    for column in categories:
        dataframe.dropna(subset=[column], inplace=True)

    for text_column in text_columns:
        dataframe[text_column] = df.groupby(text_column).ngroup()

    dataframe.dropna(inplace=True)
    dataframe.corr()
    
    return dataframe
```

## Training

Training is a key element of the entire project infrastructure. All actions take place in the `train.py` file. First, the data provided by the user is processed (described in the section Data set). Three directories have been created in the project file structure. The first is a directory called "dataset", the second is "models" and the third is "old". The last one is in the 'models' parent directory.
This chapter will be divided into three points: creating a completely new model, creating a model and comparing it with an existing one, and post-training.

### Creating new model

When creating a new model - the dataset and models directory is empty. After such verification by the application - the data set is saved to the dataset directory. If an argument causing data drift was given at the input - then an additional portion of data is added to the file in the application's cache in order to reduce the final precision of the model.
After the data is prepared, the project "pipeline" is prepared. This is done using the `setup` function provided by the pycaret library. Parameters such as the data set (pandas library dataframe type), the purpose of the classification (which column will be classified), the name of the experiment, and additional supplementary parameters, such as `silent = True`, indicating that the application is to operate without the administrator's intervention, are passed to the function.
Then the` compare_models function` is run, which returns to the variable best the algorithm that has the best predispositions to train the indicated set. After that, functions are performed to optimize the model and its quality (accuracy coefficient).
The application saves the information that a new model has appeared in the system and saves it in the .pkl format in the `models` directory. The file contains the accuracy coefficient and the ordinal number in the name.

### Training the model and comparing it with the existing one

When a model already exists in the Models directory, the application does not replace the data set or the model immediately. The training process is the same as in the point above, but additionally asks the following question:

_Is the current model better than the previous one?_

If so - the old model should be moved to the models / old directory, and the new one should be saved in the models directory with the next ordinal number and the new accuracy value. The old dataset is deleted and replaced with the new one. The log system records relevant information.
If not, it means that a drift has been detected and changes are discarded. The log system records information about the detected drift.

### Post training

The training data file has the same shape as the file saved in the dataset directory. A post-training package is added to the existing data set using the concat function included in the pandas library. The training method is the same as in the previous sections. Finally, the quality of the model is compared as if you were training with an existing model. If it turns out that the quality of the model has increased, then the new data set (with additional records from the training file) and the model are saved in target directories.

### Prediction

The system offers a data prediction function. The input includes the path to the data set to be classified. The prediction is made in the predict.py file, it prepares the data using the same functions as during training, it is connected to the "pipeline" using the setup function. The model in the .pkl format is read from the models directory, and then the prediction is performed using the predict_model function of the pycaret library. The result is displayed on the console.

![image](https://user-images.githubusercontent.com/50612974/176908297-d2d6269f-4419-4352-88ee-88cd19173421.png)


## Other systems

### Drift detection


Drift detection works by finding the difference in acuracy between the new and old models. If the coefficient deteriorates, then the drift detector does not allow to record a new model. The application allows for the artificial addition of erroneous data to reduce the quality of the model - for presentation purposes.

Forcing the drift detection is as follows:

```
def add_some_trash_data(dataset):
    values = [[1,1,1,0,0,4583,1508,128,360,1,0,1], [0,0,0,0,1,4583,0,133,360,0,1,0]]
    pandas_frame = pd.DataFrame(values, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status'])
    print(pandas_frame)
    new_dataset = dataset.append(pandas_frame, ignore_index=True)
    return new_dataset
```


### Log system

The application has the ability to save logs after training, retraining, or during drift detection. This allows you to easily manage the application and check its current status and history.

![image](https://user-images.githubusercontent.com/50612974/176908366-8b64b84d-0924-490e-a5ea-d38a4e7bd551.png)


System saves logs when a new model appears (previously there was none), when the new model is better than the previous one, and when data drift has been detected.

### MLFlow tracking

The system uses the mlflow library to track the progress and results of training. The view can be started with the `mlflow ui` command. Training data are saved to the mlruns directory and can be read at any time using the mlflow HTTP server, which starts at localhost: 5000 (by default).

![image](https://user-images.githubusercontent.com/50612974/176909860-52d2c3a1-1ea3-4921-9f2a-dd0374c50743.png)

### Docker

The application can be launched in two ways. Using the classic command line or using the docker image. Creating a docker image (an application for containerization of projects) required creating an additional file - `dockerfile`. This file contains information about the project, how it is prepared to run, and how the application should be treated in the image.

Docker image has been prepared using the command:

`docker build -t python-image .`

The container can be started in two ways:

```docker run –publish 8000: 8000 python-image```

```docker run python-image```

Additionally, for the correct operation of the application, the parameters sent to the main.py file should be added to the command. The parameters will be presented in the next part of the documentation.



## PL DOCUMENTATION

Full documentation in Polish is in the repository in the `documentation_PL` directory.


## Known bugs

The project has minor problems with the pandas library - sometimes the prediction is not working properly and mismatch n_features errors are displayed. Then you have to run the prediction again.

## Authors

Adrian Kordas

Piotr Skibiński

Wojciech Szczepański

_As part of a project in college
Polish-Japanese Academy of Information Technology_


