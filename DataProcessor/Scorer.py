import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import re
import datetime
import time

test_name_table = pd.read_csv("D://data//codes//ids_test.csv", encoding="utf-8")
test_name_table["rID_v1115"] = test_name_table["rID_v1115"].astype('float64')
list_name = test_name_table.rID_v1115.values.tolist()
new_list_name = []
for name in list_name:
    new_list_name.append(name)
list_name = new_list_name
set1 = set(list_name)

# 读取全部样本的数据并且合并
# full_table_1 = pd.read_csv("D://data//codes//data_1.csv", encoding="gbk")
# buchun = pd.read_csv('D://data//codes//补充样本.csv', encoding='utf-8')
# full_table = pd.concat([full_table_1, buchun])
full_table = pd.read_csv('D://data//codes//0218完整样本.csv', encoding='utf-8')
full_table['job_home'] = np.where(full_table['job_province']==full_table['home_prov'], 1, 0)
full_table['job_home_city'] = np.where(full_table['job_city']==full_table['home_city'], 1, 0)
full_table["rID_v1115"] = full_table["rID_v1115"].astype('float32')
tenure_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(full_table[full_table.tenure.isin(tenure_list)])
# print(full_table[full_table.tenure.isin(tenure_list)]['tenure'].unique())


# print(full_table[full_table.rID_v1115.isin(list_name)])
# test_table = full_table[full_table.name.isin(list_name)]
# test_in_full_table = set(list(test_table["name"].value_counts().index))


full_table.to_csv("full_data_with_name_0218.csv")
# test_table.to_csv("test_data_with_name.csv")
#
# full_table_without_name = full_table[['tenure',   'jobrank', "gender_fem",  'age','job_home_city',
#                                              'ethnicity', 'job_home', 'edu_first', 'college_major',
#                                              'edu_onjob', 'edu_onjob_major', 'firstjob_type', 'work_age',
#                                              'exp_firm', 'exp_mform', 'exp_uform', 'party_age', 'ruralexp',
#                                              'overseas']]
# test_table_without_name = test_table[['tenure',   'jobrank', "gender_fem",  'age','job_home_city',
#                                              'ethnicity', 'job_home', 'edu_first', 'college_major',
#                                              'edu_onjob', 'edu_onjob_major', 'firstjob_type', 'work_age',
#                                              'exp_firm', 'exp_mform', 'exp_uform', 'party_age', 'ruralexp',
#                                              'overseas']]
# full_table_without_name.to_csv("full_data_without_name.csv")
# test_table_without_name.to_csv("test_data_without_name.csv")
# print(test_in_full_table)


