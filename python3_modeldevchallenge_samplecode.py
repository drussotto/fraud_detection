print ("IMPORTING LIBRARIES...")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import requests
from requests.auth import HTTPBasicAuth


#DOWLOADING FILE FROM DROPBOX FIRST TIME
import urllib.request as urllib2
import os.path
import time
import random
while not os.path.exists('dev.csv') or not os.path.exists('oot0.csv'):
    time.sleep (3*random.random()); #Sleeping less than 3 seconds before going to Dropbox - avoid too many students at once.
    if not os.path.exists('dev.csv'):
        print ("DOWLOADING FILE dev.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/yn6hvc0x9sjxbsa/dev.csv?dl=1")
        output = open('dev.csv','wb')
        output.write(csvfile.read())
        output.close()
    if not os.path.exists('oot0.csv'):
        print ("DOWLOADING FILE oot0.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/i2l3iexmun0bkp2/oot0.csv?dl=1")
        output = open('oot0.csv','wb')
        output.write(csvfile.read())
        output.close()  
#DOWLOADING FILE FROM DROPBOX FIRST TIME

    
print ("LOADING DATASETS...")
df = pd.read_csv("dev.csv") #DEV-SAMPLE
dfo = pd.read_csv("oot0.csv")#OUT-OF-TIME SAMPLE

print ("IDENTIFYING TYPES...")
in_model = []
list_ib = set()  #input binary
list_icn = set() #input categorical nominal
list_ico = set() #input categorical ordinal
list_if = set()  #input numerical continuos (input float)
list_inputs = set()
output_var = 'ob_target'

for var_name in df.columns:
    if re.search('^i',var_name):
        list_inputs.add(var_name)
        print (var_name,"is input")
    if re.search('^ib_',var_name):
        list_ib.add(var_name)
        print (var_name,"is input binary")
    elif re.search('^icn_',var_name):
        list_icn.add(var_name)
        print (var_name,"is input categorical nominal")
    elif re.search('^ico_',var_name):
        list_ico.add(var_name)
        print (var_name,"is input categorical ordinal")
    elif re.search('^if_',var_name):
        list_if.add(var_name)
        print (var_name,"is input numerical continuos (input float)")
    elif re.search('^ob_',var_name):
        output_var = var_name
    else:
        print ("ERROR: unable to identify the type of:", var_name)


print ("STEP 1: DOING MY TRANSFORMATIONS...")

print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")
#in_model = list_inputs #['ib_var_1','icn_var_22','ico_var_25','if_var_65']
in_model = ['ib_var_1','icn_var_22','ico_var_25','if_var_65']

print ("STEP 3: DEVELOPING THE MODEL...")
X = df[in_model]
y = df[output_var]

Xo = dfo[in_model]

#model = sm.Logit(y,X)
try:
    model = sm.GLM(y,X, family = sm.families.Binomial())
    result = model.fit()
    print (result.summary())
    y_pred = result.predict(X)
    yo_pred = result.predict(Xo)
except np.linalg.linalg.LinAlgError as err:

   if 'Singular matrix' in err.message:
        print ("MODEL-INVALID (Singular Matrix)")
   else:
        raise

print ("STEP 4: ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(df[output_var], y_pred)-1
print ("GINI DEVELOPMENT=", gini_score)

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


KS_score = KS(df[output_var],y_pred)
print ("KS DEVELOPMENT=", KS_score) 

print ("STEP 5: SUBMITTING THE RESULTS...")
dfo['pred'] = yo_pred
dfo_tosend = dfo[list(['id','pred'])]

i=1
filename = "group_Z_sub"+str(i)+".csv"
dfo_tosend.to_csv(filename, sep=',')

url = 'http://mfalonso.pythonanywhere.com/api/v1.0/uploadpredictions'

files = {'file': (filename, open(filename, 'rb'))}
rsub = requests.post(url, files=files, auth=HTTPBasicAuth('<my_user_name_goes_here>', '<my_password_goes_here>'))
resp_str = str(rsub.text)
print ("RESULT SUBMISSION: ", resp_str)



