# cdbprom

This repository contains a series of scripts used to make predictions of bacterial promoters associated with the Comprehensive Directory of Bacterial Promoters.

A training dataset encompassing experimentally validated promoter sequences from 11 bacteria is firstly used to train the model.
The dataset **training_data.csv** contains the sequences of nucleotides. It is also included the negative sequences.

All the sequences in **training_data.csv** (except for the label) need to be converted to DNA Duplex Stability (DDS) with the **stability_converter.py** script.
Once the sequences are converted, they can be applied in the **predictor.py** script.

The classifier employed in CDBProm is composed of two models:
 Step 1: the training with the whole **training_data.csv**
 Step 2: the training with the misclassification from **step 1**

The pre-trained models are also available in the files **xgboost_1.model** and **xgboost_2.model**, respectively.

If you have your own sequences and you want to submit them to CDBProm's predictor, you need to:
 1. Download the models (**xgboost_1.model** and **xgboost_2.model**)
 2. Load the models in a Python environment.
 3. The library xgboost needs to be loaded
    
    
    3.1 - You need the package xgboost (e.g. import xgboost as xgb)
    
    3.2 - You need to load the model 1; e.g. xgb1 = xgb.Booster("path/to/the/downloaded/model/xgboost_1.model")
    
    3.3 - You need to load the model 2; e.g. xgb2 = xgb.Booster("path/to/the/downloaded/model/xgboost_2.model")

    
 5. You need to convert your sequences to DDS (please use the stability_converter.py script).
 **Please note that the script stability_converter.py might need to be adapted to the format of your input sequences**


 7. Your converted sequences can be predicted with xgb1.predict(your_data) and xgb2.predict(your data).
 8. A sequence is predicted as a promoter if the prediction of xgb1 is equal to 1 **OR** if the prediction of xgb2 is equal to 1.


**If you have any questions, don't hesitate in reaching out at gustavo.sganzerla@dal.ca**
