from datetime import date
import pandas as pd
from sklearn.utils import resample

da = pd.read_csv("disagreed_proc_og.csv", sep=",")
ur = pd.read_csv("unrelated_proc_og.csv", sep=",")
ag = pd.read_csv("agreed_proc_og.csv", sep=",")

# 20% of 175,592*3 is 105,355


print("Disagree")
print(da.shape)
print(da.isnull().sum())
# 2 % of 105,355  == 2,107
da = da.sample(frac=1)
da_upsampled = resample(da,
                        replace=True,     # sample with replacement
                        n_samples=175592,    # to match majority class
                        random_state=123)
da_val = da_upsampled[:2107]
da = da_upsampled[2107:]
print(da.head)
print(da_val.head)


print("Agree")
print(ag.shape)
print(ag.isnull().sum())
# 29 % of 105,355  == 30,552
ag = ag.sample(frac=1)
ag_upsampled = resample(ag,
                        replace=True,     # sample with replacement
                        n_samples=175592,    # to match majority class
                        random_state=123)
ag_val = ag_upsampled[:30552]
ag = ag_upsampled[30552:]
print(ag.head)
print(ag_val.head)


print("Unrelated")
print(ur.shape)
print(ur.isnull().sum())
ur = ur.sample(frac=1)
# 68 % of 105,355  == 71,641 No upsampling needed
ur = ur.sample(frac=1)
ur_val = ur[:71641]
ur = ur[71641:]
print(ur.head)
print(ur_val.head)

pdList = [da, ag, ur]
new_df = pd.concat(pdList)
new_df = new_df.sample(frac=1)
print("Training Data")
print(new_df.shape)
new_df.to_csv("Merged_Train.csv", index=False)

pdList = [da_val, ag_val, ur_val]
new_df = pd.concat(pdList)
new_df = new_df.sample(frac=1)
print("Validation Data")
print(new_df.shape)
new_df.to_csv("Merged_Val.csv", index=False)


# val = pd.read_csv("Merged_Val.csv", sep=",")
# train = pd.read_csv("Merged_Train.csv", sep=",")
# print(val.shape)
# print(val.isnull().sum())
# print(train.shape)
# print(train.isnull().sum())
