import numpy as np
import pandas as pd
import lightgbm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
from xgboost.sklearn import  XGBClassifier
from lightgbm.sklearn import  LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense,Masking
from keras.layers import LSTM
from keras.models import load_model
import h5py
from sklearn.preprocessing import StandardScaler

#构建网络层

#model =Sequential()
#model.add(Masking(mask_value=0,input_shape=(6,15)))
#model.add(LSTM(128, input_shape=(6, 15)))
#model.add(Dense(1,activation='sigmoid'))
AB_features_mean_dict={'AST': 265.7649374462886,
 'Age': 61.99081631939693,
 'Alkalinephos': 103.30392278793136,
 'BUN': 23.800783629152104,
 'BaseExcess': -1.6583773702608087,
 'Bilirubin_direct': 2.0574301609316694,
 'Bilirubin_total': 2.195082314533883,
 'Calcium': 7.650211220950771,
 'Chloride': 106.23729760262994,
 'Creatinine': 1.5237691478081694,
 'DBP': 63.109905477596314,
 'FiO2': 0.6337038756156832,
 'Fibrinogen': 286.3355787302802,
 'Gender': 0.5589273592596904,
 'Glucose': 136.22181776902823,
 'HCO3': 23.6567700132683,
 'HR': 84.56358484348917,
 'Hct': 30.859478373879803,
 'Hgb': 10.388532225723932,
 'ICULOS': 26.991223576609915,
 'Lactate': 2.7263216389942446,
 'MAP': 82.5672411714298,
 'Magnesium': 2.0550322983387685,
 'O2Sat': 97.1917182498469,
 'PTT': 42.17045222694993,
 'PaCO2': 40.800210443404445,
 'Phosphate': 3.5269320094441445,
 'Platelets': 195.5363523964943,
 'Potassium': 4.129519540971935,
 'Resp': 18.72211994003603,
 'SBP': 123.78039605671232,
 'SaO2': 93.89096171434177,
 'Temp': 36.97640643288965,
 'TroponinI': 8.754286518446602,
 'WBC': 11.328748275605989,
 'pH': 7.376091731188639}
all_columns=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 
             'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 
             'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
             'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
             'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 
             'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
standard_lizer=joblib.load("./A_std_norm.m")
#min_max_lizer=joblib.load("./A_min_max_norm.m")
def fill_null_with_mean(data):
    df=pd.DataFrame(data,columns= all_columns)
    feature_name=[i for i in all_columns if i in ['HR', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
        'pH', 'SaO2', 'Creatinine', 'Bilirubin_direct', 'Lactate',  
       'Bilirubin_total', 'WBC' ,'Platelets', 'Age']]
    for col in feature_name:
        if col=="SepsisLabel":
            continue
        else:
            if df[col].isnull().all():#if all null fill mean 
                df[col]=df[col].fillna(AB_features_mean_dict[col])
            else: 
                df[col].fillna(method="pad",inplace=True)#padding with before data
            #print(df.isnull())
                df[col].fillna(AB_features_mean_dict[col],inplace=True)
    return standard_lizer.transform(np.array(df[feature_name][0:])) #返回的是一个数组

def get_sepsis_score(current_data,model):
    seq_length=6 
    data=fill_null_with_mean(current_data)
    len_columns =data.shape[1] #15列
    end=data.shape[0]
    start =  end - seq_length if end - seq_length >= 0 else 0  
    features_temp = np.zeros((seq_length,len_columns))    
    features = data[start:end]
    index = seq_length - features.shape[0]    
    features_temp[index:] = features
    score =model.predict(features_temp.reshape(1,seq_length,data.shape[1]))
    label = score > 0.40
    return score,label
 
def load_sepsis_model():
   # model =Sequential()
    #model.add(Masking(mask_value=0,input_shape=(6,15)))
   # model.add(LSTM(128, input_shape=(6, 15)))
   # model.add(Dense(1,activation='sigmoid'))
   # model.load_weights('/home/fms/cinc_data/LSTM_model_weight.h5')
    model=load_model("./LSTM_model.h5")
    return model 
