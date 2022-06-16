import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import re

guanyuan_col = pd.read_csv('all_after_rf.csv', encoding='utf-8', index_col=0)
all_data = guanyuan_col[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
       'job_home', 'edu_first', 'college_major', 'edu_onjob',
       'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
       'exp_uform', 'party_age', 'ruralexp', 'overseas']]
all_target = guanyuan_col[['tenure']]
# 获取需要进入测试的list名单
test_name_table = pd.read_csv("D://data//codes//test_full.csv", encoding="gbk")
list_name = test_name_table.书记姓名.values.tolist()
new_list_name = []
for name in list_name:
    new_name = re.sub('\\(.*?\\)', '', name)
    new_name = re.sub('\\（.*?\\）', '', new_name)
    new_list_name.append(new_name)
list_name = new_list_name

train_table = guanyuan_col[~guanyuan_col.name.isin(list_name)]
train_table.index = range(len(train_table))

test_table = guanyuan_col[guanyuan_col.name.isin(list_name)]
test_table.index = range(len(test_table))

# train_index = train_table[['name', 'rID_v1115', 'officerID', 'job_province', 'job_city']]
train_data = train_table[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
       'job_home', 'edu_first', 'college_major', 'edu_onjob',
       'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
       'exp_uform', 'party_age', 'ruralexp', 'overseas']]
train_target = train_table[['tenure']]

test_index = test_table[['name', 'rID_v1115', 'officerID', 'job_province', 'job_city']]
test_data = test_table[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
       'job_home', 'edu_first', 'college_major', 'edu_onjob',
       'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
       'exp_uform', 'party_age', 'ruralexp', 'overseas']]
test_target = test_table[['tenure']]

test_predict = test_data.copy()

# estimator = RandomForestClassifier(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_data, train_target)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_class']=result
#
# estimator = RandomForestRegressor(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_data, train_target)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_reg']=result


x_index = []
y_index = []
min = -500
min_index = 10
for i in range(10, 100):
    regressor = XGBR(n_estimators=i*10)
    print(i)
    result = cross_val_score(regressor, all_data, all_target, cv=10, scoring='neg_mean_squared_error')
    x_index.append(i)
    y_index.append(np.mean(result))
    if np.mean(result) > min:
        min = np.mean(result)
        min_index = i
print("min score is {}".format(min))
print("index is {}".format(min_index))
plt.figure(figsize=(50, 30), dpi=100)
plt.plot(x_index, y_index)
plt.show()

# estimator = XGBR(n_estimators=500).fit(train_data, train_target)
# result = estimator.predict(test_data)
# print(estimator.score(test_data, test_target))

# test_predict = test_predict[['pred_class', 'pred_reg', 'pred_xgbr']]
# final = pd.concat([test_index, test_predict, test_target], axis=1)
# final.to_csv('test_0212_with_xgbr.csv', encoding='utf-8')
# print(final)
