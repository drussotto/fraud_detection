import pandas as pd
from utils import perform_rfe, impute_nas, TARGET, send_submission
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("input/dev.csv").drop("id", axis=1) #DEV-SAMPLE
test = impute_nas(pd.read_csv("input/oot0.csv").drop("id", axis=1))#OUT-OF-TIME SAMPLE

model = RandomForestClassifier(random_state=20190628, n_estimators=1000)

results = perform_rfe(model,
                      train,
                      test,
                      "random_forest_rfe.csv",
                      to_remove=80)



#results = pd.read_csv("random_forest_rfe.csv")
num_removed = 82 - results["n"]

fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=num_removed[:70], y=results["grade"][:70], ax=ax)

best = results[results["grade"] == max(results["grade"])]


model = RandomForestClassifier(random_state=20190628, n_estimators=1000)
model.fit(train[best["feats"].item().split(";")], train[TARGET])


preds = model.predict_proba(test[best["feats"].item().split(";")])[:,1]


result = send_submission("rf_after_rfe.csv", preds)