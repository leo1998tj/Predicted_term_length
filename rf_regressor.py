import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import re

#
guanyuan = pd.read_csv('DataProcessor/guanyuan.csv', encoding='utf-8', index_col=0)
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

# 处理分类型的变量
oe = OrdinalEncoder()
label = LabelEncoder()
for col in columns_type:
    guanyuan.loc[guanyuan[col].notna(), [col]] = oe.fit_transform(guanyuan[col].dropna().values.reshape(-1, 1))
    # guanyuan[col] = label.fit_transform(guanyuan[col])


# 剔除任期列，任期为最后求得的target
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


# X = [guanyuan_data_reg]
# mse = []
# for x in X:
#     estimator = RandomForestRegressor(random_state=0, n_estimators=500)
#     scores = cross_val_score(estimator, x, guanyuan_target, scoring='neg_mean_squared_error', cv=10).mean()
#     print(scores)
#     mse.append(scores * -1)
# print(mse)


test_table = pd.read_csv("D://data//codes//DataProcessor//test_data_with_name.csv", encoding="utf-8")

result_final = test_table.copy()
test_table = test_table.drop(test_table[(test_table.tenure < 3)].index, inplace=True)
test_data = test_table[['jobrank', 'age', 'job_home_city', 'ethnicity', "gender_fem",
       'job_home', 'edu_first', 'college_major', 'edu_onjob',
       'edu_onjob_major', 'firstjob_type', 'work_age', 'exp_firm', 'exp_mform',
       'exp_uform', 'party_age', 'ruralexp', 'overseas']]
test_target = test_table[['tenure']]


estimator = RandomForestClassifier(random_state=0, n_estimators=500)
model_result = estimator.fit(guanyuan_data_reg, guanyuan_target)
result = model_result.predict(test_data)
result = pd.DataFrame(result)
result_final['pred_class']=result

estimator = RandomForestRegressor(random_state=0, n_estimators=500)
model_result = estimator.fit(guanyuan_data_reg, guanyuan_target)
result = model_result.predict(test_data)
result = pd.DataFrame(result)
result_final['pred_reg']=result

result_final['tenure'] = test_target

final = result_final[['tenure', 'pred_class', 'pred_reg']]
final.to_csv('test2.csv', encoding='utf-8')
print(final)
