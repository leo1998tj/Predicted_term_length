import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBRegressor as XGBR
import re

# 设置变量参数
ethnicity = {"Han":1, "Uyghur":2, "Hui":3, "Mongol":4, "Zang":5, "Zhuang":6, "Manchu":7, "other":8}
gender = {"male":0, "famale":1}
administrative_rank = {"prefecture":1, "sub-provincial":2}
whether_sheorhe_serves_in_her_his_province_of_origin = {"Yes":1, "No":0}
highest_full_time_degree = {"high school or below":1, "junior college":2, "college":3, "master":4, "doctor":5,}
highest_part_time_degree = {"none":0, "college":1, "graduate study":2, "MBA":3,"doctor":4}
major_in_the_undergraduate_study = {"engineering":1, "natural science":2, "arts, history and philosophy":3, "political science and law":4, "economics, management, and business":5, "other social science":6}
major_in_the_highest_part_time_degree = {"engineering":1, "natural science":2, "arts, history and philosophy":3, "political science and law":4, "economics, management, and business":5, "other social science":6}
whether_she_he_has_studied_overseas = {"Yes":1, "No":0}
# “the rustication movement of China’s educated youth”
whether_she_he_has_experienc = {"Yes":1, "No":0}
# whether_she_he_has_worked in factories, companies, and business sector before being promoted to a public office with rank at the county level
wshhwi = {"Yes":1, "No":0}
# whether_she_he_has_worked in local and subnational governments (M-form organizations) before being promoted to a public office with rank at the county level
wshhwilas= {"Yes":1, "No":0}
# whether_she_he_has_worked in the central government (U-form organizations) before being promoted to a public office with rank at the county level
wshhwitcg = {"Yes":1, "No":0}


# 读取全部数据进入程序
filename = 'full_data_with_name_04162'
# train_switch = 'all'
train_switch = 'part'

guanyuan = pd.read_csv("D://data//codes//DataProcessor//{}.csv".format(filename), encoding="utf-8", index_col=0)
guanyuan["rID_v1115"] = guanyuan["rID_v1115"].astype('float32')
guanyuan.index = range(len(guanyuan))
# guanyuan = pd.read_csv('DataProcessor/guanyuan.csv', encoding='utf-8', index_col=0)
# guanyuan = guanyuan.reset_index(drop=True, inplace=True)
# print(guanyuan)
# print(guanyuan.isnull().sum())
#
# 不同列使用不同策列填充
columns_zero = ["exp_firm", "exp_mform", "exp_uform", "ruralexp", "overseas"]
columns_class = ["ethnicity", "edu_first", "college_major", "edu_onjob", "edu_onjob_major", "firstjob_type"]
columns_reg = ["age", "work_age", "party_age"]
columns_type = ["ethnicity", "edu_first", "college_major", "edu_onjob", "edu_onjob_major", "firstjob_type",
                "exp_firm", "exp_mform", "exp_uform", "ruralexp", "overseas",
                "gender_fem", "jobrank", "job_home_city", "job_home"]
columns_full = ["tenure", "gender_fem", "jobrank" , "job_home_city", "job_home", "exp_firm", "exp_mform", "exp_uform", "ruralexp", "overseas"]

# 填充为0的数值
for column in columns_zero:
    column_new = guanyuan.loc[:, column].values.reshape(-1, 1)
    imp_0 = SimpleImputer(strategy="constant", fill_value=0)
    guanyuan[column] = imp_0.fit_transform(column_new)
    # guanyuan[column] = guanyuan[column].astype('float64')
    # print(guanyuan[column].dtypes)

# 处理分类型的变量
oe = OrdinalEncoder()
label = LabelEncoder()
for col in columns_type:
    try:
        guanyuan.loc[guanyuan[col].notna(), [col]] = oe.fit_transform(guanyuan[col].dropna().values.reshape(-1, 1))
    except:
        print(col)
    # guanyuan[col] = label.fit_transform(guanyuan[col])


# 剔除任期列，任期为最后求得的target
guanyuan_index = guanyuan[['name', 'rID_v1115', 'officerID', 'job_province', 'job_city']]
guanyuan_data = guanyuan[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
       'job_home', 'edu_first', 'college_major', 'edu_onjob',
       'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
       'exp_uform', 'party_age', 'ruralexp', 'overseas']]
guanyuan_target = guanyuan[['tenure']]

# 使用随机森林填补缺失值
guanyuan_data_reg = guanyuan_data.copy()
guanyuan_data_reg=pd.DataFrame(guanyuan_data_reg,dtype=np.float)

sort_index = np.argsort(guanyuan_data_reg.isnull().sum(axis=0)).values


for i in sort_index:
    # print(guanyuan_data_reg.columns[i])
    if guanyuan_data_reg.columns[i] not in columns_full:
        # 构建新的特征矩阵及标签
        df = guanyuan_data_reg
        fillc = df.iloc[:,i]
        df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(guanyuan_target)], axis=1)
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        # 分出训练集和测试集
        Y_train = fillc[fillc.notnull()]
        Y_test = fillc[fillc.isnull()]
        X_train = df_0[Y_train.index, :]
        X_test = df_0[Y_test.index, :]

        # 用随机森林回归来预测缺失值
        if guanyuan_data_reg.columns[i] in columns_reg:
            rfc = RandomForestRegressor(n_estimators = 100)
            rfc = rfc.fit(X_train, Y_train)
            Y_predict = rfc.predict(X_test)
        else:
            rfc = RandomForestClassifier(n_estimators = 100)
            rfc = rfc.fit(X_train, Y_train)
            Y_predict = rfc.predict(X_test)

        # 填补回来
        guanyuan_data_reg.loc[guanyuan_data_reg.iloc[:, i].isnull(), guanyuan_data_reg.columns[i]] = Y_predict

guanyuan_col = pd.concat([guanyuan_index, guanyuan_data_reg, guanyuan_target], axis=1)
guanyuan_col.index = range(len(guanyuan_col))
# guanyuan_col.to_csv('all_after_rf.csv', encoding='utf-8')
#
# guanyuan_col = pd.read_csv('all_after_rf.csv', encoding='utf-8', index_col=0)

# 获取需要进入测试的list名单
test_name_table = pd.read_csv("D://data//codes//补充样本0218.csv", encoding="utf-8")
test_name_table["rID_v1115"] = test_name_table["rID_v1115"].astype('float64')
list_name = test_name_table.rID_v1115.values.tolist()
new_list_name = []
for name in list_name:
    new_list_name.append(name)
list_name = new_list_name
# 旧的提取逻辑
# test_name_table = pd.read_csv("D://data//codes//test_full.csv", encoding="gbk")
# list_name = test_name_table.书记姓名.values.tolist()
# new_list_name = []
# for name in list_name:
#     new_name = re.sub('\\(.*?\\)', '', name)
#     new_name = re.sub('\\（.*?\\）', '', new_name)
#     new_list_name.append(new_name)
# list_name = new_list_name
# guanyuan_col = guanyuan_col.copy()

train_table = guanyuan_col[~guanyuan_col.rID_v1115.isin(list_name)]
train_table.to_csv('train_table_0428_full.csv', encoding='utf-8')

train_table.drop(train_table[(train_table.tenure < 3)].index, inplace=True)
train_table.drop(train_table[(train_table.tenure > 5 )].index, inplace=True)

train_table.index = range(len(train_table))

test_table = guanyuan_col[guanyuan_col.rID_v1115.isin(list_name)]
test_table.index = range(len(test_table))
guanyuan_col.to_csv('full_table_0428.csv', encoding='utf-8')
train_table.to_csv('train_table_0428_part.csv', encoding='utf-8')
test_table.to_csv('test_table_0428.csv', encoding='utf-8')
#
# guanyuan_data = guanyuan_col[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
#        'job_home', 'edu_first', 'college_major', 'edu_onjob',
#        'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
#        'exp_uform', 'party_age', 'ruralexp', 'overseas']]
# guanyuan_target = guanyuan_col[['tenure']]
#
# # train_index = train_table[['name', 'rID_v1115', 'officerID', 'job_province', 'job_city']]
# train_data = train_table[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
#        'job_home', 'edu_first', 'college_major', 'edu_onjob',
#        'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
#        'exp_uform', 'party_age', 'ruralexp', 'overseas']]
# train_target = train_table[['tenure']]
#
# test_index = test_table[['name', 'rID_v1115', 'officerID', 'job_province', 'job_city']]
# test_data = test_table[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
#        'job_home', 'edu_first', 'college_major', 'edu_onjob',
#        'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
#        'exp_uform', 'party_age', 'ruralexp', 'overseas']]
# test_target = test_table[['tenure']]
#
# # guanyuan_col.to_csv('all_after_rf_0414.csv', encoding='utf-8')
# # train_table.to_csv('train_after_rf_0414.csv', encoding='utf-8')
# # test_table.to_csv('test_after_rf_0414.csv', encoding='utf-8')
# test_predict = test_data.copy()
# #
# # print(test_data.isnull().sum())
#
# if train_switch == 'all':
#     train_x = guanyuan_data
#     train_y = guanyuan_target
# else:
#     train_x = train_data
#     train_y = train_target
#
# print(train_data.isnull().sum())
# print(train_target.isnull().sum())
#
# estimator = RandomForestClassifier(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_x, train_y)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_class_rf']=result
#
# estimator = RandomForestRegressor(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_x, train_y)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_reg_rf']=result
#
# estimator = GradientBoostingClassifier(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_x, train_y)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_class_gbdt']=result
#
# estimator = GradientBoostingRegressor(random_state=0, n_estimators=500)
# model_result = estimator.fit(train_x, train_y)
# result = model_result.predict(test_data)
# result = pd.DataFrame(result)
# test_predict['pred_reg_gbdt']=result
# # for i in range(5, 100):
# #     estimator = XGBR(n_estimators=(i * 100)).fit(train_data, train_target)
# #     result = estimator.predict(test_data)
# #     print("{}:{}".format(i*100, estimator.score(test_data, test_target)))
#
# test_predict = test_predict[['pred_class_rf', 'pred_reg_rf', 'pred_class_gbdt', 'pred_reg_gbdt']]
# final = pd.concat([test_index, test_predict, test_target], axis=1)
# path = '{}_{}_result.csv'.format(filename, train_switch)
# final.to_csv('{}_{}_result.csv'.format(filename, train_switch), encoding='utf-8')
# print(final)
#
