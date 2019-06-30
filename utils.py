from sklearn.metrics import roc_auc_score
import re
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from sklearn.feature_selection import RFE
from math import floor
from time import sleep

TARGET = 'ob_target'

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


def KS(b,a):  
    """Function that received two parameters; first: a binary variable representing 0=good and 1=bad, 
    and then a second variable with the prediction of the first variable, the second variable can be continuous, 
    integer or binary - continuous is better. Finally, the function returns the KS Statistics of the two lists."""
    try:
        tot_bads=1.0*sum(b)
        tot_goods=1.0*(len(b)-tot_bads)
        elements = zip(*[a,b])
        elements = sorted(elements,key= lambda x: x[0])
        elements_df = pd.DataFrame({'probability': b,'gbi': a})
        pivot_elements_df = pd.pivot_table(elements_df, values='probability', index=['gbi'], aggfunc=[sum,len]).fillna(0)
        max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
        for i in range(len(pivot_elements_df)):
            perc_goods =  (pivot_elements_df.iloc[i]['len'] - pivot_elements_df.iloc[i]['sum']) / tot_goods
            perc_bads = pivot_elements_df.iloc[i]['sum']/ tot_bads
            cum_perc_goods += perc_goods
            cum_perc_bads += perc_bads
            A = cum_perc_bads-cum_perc_goods
            if abs(A['probability']) > max_ks:
                max_ks = abs(A['probability'])
    except:
        max_ks = 0
    return max_ks


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
    
    return resp_str



def retrieve_scores(result):
    return tuple([x.split("; ")[0] for x in result.split("= ")[1:]])
    
def build_results_csv(filename, features, result, create_file=False):
    ks2, gini, grade = retrieve_scores(result)
    feats = ";".join(features)
    
    
      
    new_row = pd.DataFrame(dict(n=[len(features)], ks2=[ks2], gini=[gini],
                             grade=[grade], feats=[feats]))
    
    if create_file:
        new_row.to_csv(filename, index=False)
        return new_row
    else:
        acc = pd.read_csv(filename)
        new_results = pd.concat([acc, new_row],ignore_index=True)
        new_results.to_csv(filename, index=False)
    
        return new_results

def perform_rfe(model, train, test, filename, to_remove=None):
    
    if to_remove is None:
        to_remove = floor(0.3*len(train.columns))
    
    
    X = train.drop(TARGET, axis=1)
    y = train[TARGET]
    
    model.fit(X, y)
    preds = model.predict_proba(test)[:,1]
    build_results_csv(filename,
                      X.columns,
                      send_submission("doesnt_matter.csv", preds),
                      create_file=True)
    sleep(3)



    for i in range(to_remove):
        rfe = RFE(model, n_features_to_select=len(X.columns)-1).fit(X, y)
        
        preds = rfe.predict_proba(test)[:,1]
        
        X = X.iloc[:,rfe.get_support()]
        test = test.iloc[:, rfe.get_support()]
        
        results = build_results_csv(filename,
                          X.columns,
                          send_submission("doesnt_matter.csv", preds))
        sleep(3)
    
    return results
    
def save_hyperparameters(params, results, filename, create_file):
    ks2, gini, grade = retrieve_scores(results)
    
    new_row = pd.DataFrame(dict(ks2=[ks2], gini=[gini],
                             grade=[grade], params=[params]))
    
    if create_file:
        new_row.to_csv(filename, index=False)
        return new_row
    else:
        acc = pd.read_csv(filename)
        new_results = pd.concat([acc, new_row],ignore_index=True)
        new_results.to_csv(filename, index=False)
        return new_results
