from sklearn.metrics import roc_auc_score
import re
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd


def get_variable_types(df): 
    types = {
        "binary": [],
        "cat_nom": [],
        "cat_ord": [],
        "continuous": [],
        "all": [],
        "target": []
    }
    
    for var_name in df.columns:
        if re.search('^i',var_name):
            types["all"].append(var_name)
        if re.search('^ib_',var_name):
            types["binary"].append(var_name)
        elif re.search('^icn_',var_name):
            types["cat_nom"].append(var_name)
        elif re.search('^ico_',var_name):
            types["cat_ord"].append(var_name)
        elif re.search('^if_',var_name):
            types["continuous"].append(var_name)
        elif re.search('^ob_',var_name):
            types["target"] = var_name
        else:
            print ("WARNING: unable to identify the type of:", var_name)

    return types
 
def impute_nas(df):
    df = df.copy()
    na_cols = df.columns[df.isna().any()].tolist()
    
    col_types = get_variable_types(df)
    
    for col in na_cols:
        if col in col_types["continuous"] or col in col_types["cat_ord"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df
    
    
def gini_score(obs, pred):
    return 2*roc_auc_score(obs, pred) - 1


def send_submission(filename, preds):
    url = 'http://mfalonso.pythonanywhere.com/api/v1.0/uploadpredictions'
    filename = "submissions/{}".format(filename)
    
    to_submit = pd.DataFrame(dict(id=pd.read_csv("input/oot0.csv")["id"],
                                  pred=preds))
    
    to_submit.to_csv(filename, sep=',')
    
    f = open(filename, 'rb')
    
    files = {'file': (filename, f)}
    
    with open("input/python_anywhere_pass.secret", 'r') as f:
        pw = f.read().strip()
    
    print("Sending submission to server...")
    rsub = requests.post(url,
                         files=files,
                         auth=HTTPBasicAuth('drussotto', pw))
    
    resp_str = str(rsub.text)
    
    print ("RESULT SUBMISSION: ", resp_str)
    
    f.close()

    



