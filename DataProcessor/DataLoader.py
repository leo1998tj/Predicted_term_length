import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

guanyuan = pd.read_csv('D://data//codes//0210官员数据.csv', encoding='utf-8')
guanyuan_without_useless_cloumns = guanyuan[['tenure', 'job_province', 'job_city', 'jobrank', "gender_fem",  'age',
                                             'ethnicity', 'job_home', 'edu_first', 'college_major',
                                             'edu_onjob', 'edu_onjob_major', 'firstjob_type', 'work_age',
                                             'exp_firm', 'exp_mform', 'exp_uform', 'party_age', 'ruralexp',
                                             'overseas', 'home_city', 'home_prov', 'officerID']]
buchun = pd.read_csv('D://data//codes//补充样本.csv', encoding='utf-8')
buchun_new = buchun[['tenure', 'job_province', 'job_city', 'jobrank', "gender_fem",  'age',
                                             'ethnicity', 'job_home', 'edu_first', 'college_major',
                                             'edu_onjob', 'edu_onjob_major', 'firstjob_type', 'work_age',
                                             'exp_firm', 'exp_mform', 'exp_uform', 'party_age', 'ruralexp',
                                             'overseas', 'home_city', 'home_prov', 'officerID']]
guanyuan_without_beizhu = pd.concat([guanyuan_without_useless_cloumns, buchun])
# guanyuan_without_beizhu = guanyuan_without_useless_cloumns
guanyuan_without_beizhu['job_home'] = np.where(guanyuan_without_beizhu['job_province']==guanyuan_without_beizhu['home_prov'], 1, 0)
guanyuan_without_beizhu['job_home_city'] = np.where(guanyuan_without_beizhu['job_city']==guanyuan_without_beizhu['home_city'], 1, 0)
final_guanyuan = guanyuan_without_beizhu[['tenure',   'jobrank', "gender_fem",  'age','job_home_city',
                                             'ethnicity', 'job_home', 'edu_first', 'college_major',
                                             'edu_onjob', 'edu_onjob_major', 'firstjob_type', 'work_age',
                                             'exp_firm', 'exp_mform', 'exp_uform', 'party_age', 'ruralexp',
                                             'overseas']]
print(final_guanyuan)


final_guanyuan.to_csv("guanyuan.csv")
# print(guanyuan_without_beizhu.isnull().sum())




