import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna.integration.lightgbm as opt_lgb

import warnings
import sys

#===================================================================

#データ取得
data_tmp = pd.read_csv("./konpe/Shinzo_5/Main_Data/Data/Dummied_0228.csv")

#目的の変数
y = data_tmp["HeartDisease"]

#テストファイル入れる
args = sys.argv

filename = args[1]

data_test = pd.read_csv(f"./{filename}")

#目的の変数
y_data_test = data_test["HeartDisease"]



#テストデータ加工 start==========================

data_test['G_ChestPainType'] = data_test['ChestPainType']
data_test.loc[data_test['G_ChestPainType']=='ASY','G_ChestPainType']='G_ASY'
data_test.loc[data_test['G_ChestPainType']=='NAP','G_ChestPainType']='G_NAP'
data_test.loc[data_test['G_ChestPainType']=='TA','G_ChestPainType']='G_TA+ATA'
data_test.loc[data_test['G_ChestPainType']=='ATA','G_ChestPainType']='G_TA+ATA'

#ChestPain_groupの作成
data_test['K_ChestPain_group'] = data_test['ChestPainType']
data_test.loc[(data_test['K_ChestPain_group'] == 'ASY') , 'K_ChestPain_group'] = 'K_ASY'
data_test.loc[(data_test['K_ChestPain_group'] == 'NAP') , 'K_ChestPain_group'] = 'K_NAP'
data_test.loc[(data_test['K_ChestPain_group'] == 'ATA') | (data_test['K_ChestPain_group'] == 'TA'),
                         'K_ChestPain_group'] = 'K_ATA/TA' #ATAとTAをグループ化

#G_CholCatの作成
data_test['G_CholCat']='chol0'
for i in range(len(data_test)):
    if 200>data_test.loc[i,'Cholesterol']>0:
        data_test.loc[i,'G_CholCat']='chol1'
    elif 320>data_test.loc[i,'Cholesterol']>=200:
        data_test.loc[i,'G_CholCat']='chol2'
    elif data_test.loc[i,'Cholesterol']>=320:
        data_test.loc[i,'G_CholCat']='chol3'
    else:
        pass


#K_Cholesterol_is0の作成
data_test['K_Cholesterol_is0'] = 'Cho_False'
data_test.loc[data_test['Cholesterol'] == 0, 'K_Cholesterol_is0'] = 'Cho_True' #CholesterolがCho_Trueの場合は0、1の場合はCho_False
data_test["K_Cholesterol_is0"] .value_counts()

#K_Oldpeak_minusの作成
data_test['K_Oldpeak_isminus'] = 'Old_False'
data_test.loc[data_test['Oldpeak'] < -0.3, 'K_Oldpeak_isminus'] = 'Old_True' #Oldpeakが-0.3以下の場合はOld_True、-0.3以上の場合はOld_False
#K_Oldpeak_abs4の作成
data_test["K_Oldpeak_abs4"] = data_test["Oldpeak"].abs()
data_test.loc[data_test["K_Oldpeak_isminus"] == 'True', "K_Oldpeak_abs4"] += 4 #Old peakが-0.3以下の場合はOld peakの絶対値に＋4


# G_H_riskの作成
data_test['G_H_risk']='Risk0'
for i in range(len(data_test)):
    if abs(data_test.loc[i,'Oldpeak'])<0.3 and data_test.loc[i,'ST_Slope']!='Up':
        data_test.loc[i,'G_H_risk']='Risk2'
    elif 0.3<=abs(data_test.loc[i,'Oldpeak'])<2.5 and data_test.loc[i,'ST_Slope']=='Up':
        data_test.loc[i,'G_H_risk']='Risk1'
    elif 0.3<=abs(data_test.loc[i,'Oldpeak'])<2.5 and data_test.loc[i,'ST_Slope']!='Up':
        data_test.loc[i,'G_H_risk']='Risk3'
    elif abs(data_test.loc[i,'Oldpeak'])>=2.5:
        data_test.loc[i,'G_H_risk']='Risk4'
    else:
        pass
#data_test['G_H_risk'].value_counts()

#M_ST_Nomal-or-Errorを作成
data_test.loc[((data_test["ST_Slope"] == "Flat") | (data_test["ST_Slope"] == "Down")) \
             & data_test["Oldpeak"] > 0.1, "M_ST_Nomal-or-Error"] = 1
data_test.loc[(data_test["ST_Slope"] == "Up") & (data_test["Oldpeak"] < -0.1), "M_ST_Nomal-or-Error"] = 1
data_test["M_ST_Nomal-or-Error"] = data_test["M_ST_Nomal-or-Error"].fillna(0)
#data_test['M_ST_Nomal-or-Error'].value_counts()

#K_Oldpeak_rangeを作成
data_test['K_Oldpeak_range'] = 'zero'
data_test.loc[(data_test['Oldpeak'] < -0.3), 'K_Oldpeak_range'] = 'minus'
data_test.loc[(data_test['Oldpeak'] >= 1)&(data_test['Oldpeak'] < 2), 'K_Oldpeak_range'] = 'one'
data_test.loc[(data_test['Oldpeak'] >= 2)&(data_test['Oldpeak'] < 3), 'K_Oldpeak_range'] = 'two'
data_test.loc[(data_test['Oldpeak'] >= 3)&(data_test['Oldpeak'] < 4), 'K_Oldpeak_range'] = 'three'
data_test.loc[(data_test['Oldpeak'] >= 4), 'K_Oldpeak_range'] = 'four'

#'K_Peak_Slope_grouped'の範囲を入れる
data_test['K_Peak_Slope_grouped'] = 'higher_risk'# higher risk = oldpeakが-0.3以上で下記以外
data_test.loc[(data_test['K_Oldpeak_range'] == 'zero')&(data_test['ST_Slope'] == 'Up'), 'K_Peak_Slope_grouped'] = 'low_risk'# low risk = oldpeakが0でslopeがUp
data_test.loc[(data_test['K_Oldpeak_range'] == 'one')&(data_test['ST_Slope'] == 'Up'), 'K_Peak_Slope_grouped'] = 'mid_risk'# mid risk = oldpeakが1でslopeがUp
data_test.loc[((data_test['K_Oldpeak_range'] == 'zero') | (data_test['K_Oldpeak_range'] == 'one'))
               & (data_test['ST_Slope'] == 'Flat'), 'K_Peak_Slope_grouped'] = 'high_risk'# high risk = oldpeakが0か1でslopeがFlat
data_test.loc[(data_test['K_Oldpeak_range'] == 'minus') , 'K_Peak_Slope_grouped'] = 'super_high_risk'# super high risk = oldpeakが-0.3以下
#data_test['K_Peak_Slope_grouped'].value_counts()

data_test_SS = data_test.copy()

#連続値の抜き出し
numeric_features=[
                  'Age',
                  'Cholesterol',
                  'RestingBP',
                  'MaxHR',
                  'Oldpeak'
                  ]

SS=StandardScaler()

data_test_SS[['SS_Age',
            'SS_Cholesterol',
            'SS_RestingBP',
            'SS_MaxHR',
            'SS_Oldpeak']]=SS.fit_transform(data_test_SS[numeric_features].values)

#ダミー変数
df_test_dummied = data_test_SS.copy()

CP_Test_dummies = pd.get_dummies(df_test_dummied['ChestPainType'])
RE_Test_dummies = pd.get_dummies(df_test_dummied['RestingECG'])
SS_Test_dummies = pd.get_dummies(df_test_dummied['ST_Slope'])

df_test_dummied = pd.concat([df_test_dummied, CP_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, RE_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, SS_Test_dummies], axis=1)


G_CP_Test_dummies = pd.get_dummies(df_test_dummied['G_ChestPainType'])
K_CP_Test_dummies = pd.get_dummies(df_test_dummied['K_ChestPain_group'])
G_CC_Test_dummies = pd.get_dummies(df_test_dummied['G_CholCat'])
K_CI_Test_dummies = pd.get_dummies(df_test_dummied['K_Cholesterol_is0'])
GHR_Test_dummies = pd.get_dummies(df_test_dummied['G_H_risk'])
KOR_Test_dummies = pd.get_dummies(df_test_dummied['K_Oldpeak_range'])
KPS_Test_dummies = pd.get_dummies(df_test_dummied['K_Peak_Slope_grouped'])

df_test_dummied = pd.concat([df_test_dummied, G_CP_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, K_CP_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, G_CC_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, K_CI_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, GHR_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, KOR_Test_dummies], axis=1)
df_test_dummied = pd.concat([df_test_dummied, KPS_Test_dummies], axis=1)


#ラベルエンコーダー
df_test_labeled = data_test_SS.copy()

le = LabelEncoder()

#ChestPainType
CPT_mapping = {'ASY':3,'TA':2, 'ATA':1, 'NAP':0}
df_test_labeled['ChestPainType'] = df_test_labeled['ChestPainType'].map(CPT_mapping)

G_CPT_mapping = {'G_ASY':2, 'G_TA+ATA':1, 'G_NAP':0}
df_test_labeled['G_ChestPainType'] = df_test_labeled['G_ChestPainType'].map(G_CPT_mapping)

K_CPT_mapping = {'K_NAP':2, 'K_ASY':1, 'K_ATA/TA':0}
df_test_labeled['K_ChestPain_group'] = df_test_labeled['K_ChestPain_group'].map(K_CPT_mapping)

#G_CholCat
G_Chol_mapping = {'chol0':4, 'chol1':0, 'chol2':1,'chol3':2}
df_test_labeled['G_CholCat'] = df_test_labeled['G_CholCat'].map(G_Chol_mapping)

K_Chol_mapping = {'Cho_False':1, 'Cho_True':0,}
df_test_labeled['K_Cholesterol_is0'] = df_test_labeled['K_Cholesterol_is0'].map(K_Chol_mapping)

#RestingECG
G_RECG_mapping = {'ST':2, 'LVH':1, 'Normal':0}
df_test_labeled['RestingECG'] = df_test_labeled['RestingECG'].map(G_RECG_mapping)

#K_Oldpeak_isminus
K_OI_mapping = {'Old_False':1, 'Old_True':0}
df_test_labeled['K_Oldpeak_isminus'] = df_test_labeled['K_Oldpeak_isminus'].map(K_OI_mapping)

#ST_Slope
STS_mapping = {'Up':2, 'Flat':1, 'Down':0}
df_test_labeled['ST_Slope'] = df_test_labeled['ST_Slope'].map(STS_mapping)

#G_H_risk
G_Hrisk_mapping = {'Risk0':0, 'Risk1':1, 'Risk2':2,'Risk3':3,'Risk4':4}
df_test_labeled['G_H_risk'] = df_test_labeled['G_H_risk'].map(G_Hrisk_mapping)

#K_Peak_Slope_grouped
df_test_labeled["K_Oldpeak_range"] = le.fit_transform(df_test_labeled["K_Oldpeak_range"])

K_PS_mapping = {'low_risk':0, 'mid_risk':1, 'high_risk':2,'higher_risk':3,'super_high_risk':4}
df_test_labeled['K_Peak_Slope_grouped'] = df_test_labeled['K_Peak_Slope_grouped'].map(K_PS_mapping)


#df_dummiedのラベルエンコーダー

#ChestPainType
df_test_dummied['ChestPainType'] = df_test_dummied['ChestPainType'].map(CPT_mapping)

df_test_dummied['G_ChestPainType'] = df_test_dummied['G_ChestPainType'].map(G_CPT_mapping)

df_test_dummied['K_ChestPain_group'] = df_test_dummied['K_ChestPain_group'].map(K_CPT_mapping)

#G_CholCat
df_test_dummied['G_CholCat'] = df_test_dummied['G_CholCat'].map(G_Chol_mapping)

df_test_dummied['K_Cholesterol_is0'] = df_test_dummied['K_Cholesterol_is0'].map(K_Chol_mapping)

#RestingECG
df_test_dummied['RestingECG'] = df_test_dummied['RestingECG'].map(G_RECG_mapping)

#K_Oldpeak_isminus
df_test_dummied['K_Oldpeak_isminus'] = df_test_dummied['K_Oldpeak_isminus'].map(K_OI_mapping)

#ST_Slope
df_test_dummied['ST_Slope'] = df_test_dummied['ST_Slope'].map(STS_mapping)

#G_H_risk
df_test_dummied['G_H_risk'] = df_test_dummied['G_H_risk'].map(G_Hrisk_mapping)

#K_Peak_Slope_grouped
df_test_dummied["K_Oldpeak_range"] = le.fit_transform(df_test_dummied["K_Oldpeak_range"])

df_test_dummied['K_Peak_Slope_grouped'] = df_test_dummied['K_Peak_Slope_grouped'].map(K_PS_mapping)

data_test = df_test_dummied.copy()

#テストデータ加工   end===============================================



warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)


#===================================================================
def Svm_set(X_train, y_train, X_test, y_test):

    feature = [
    #Age
            "Age",
    #Sex
            "Sex",
    #ChestPainType
                "ASY",
                "ATA",
                "NAP",
                "TA",
    #RestingBP
            "RestingBP",
    #Cholesterol
    #FastingBS
            "FastingBS",
    #RestingECG
                "LVH",
                "Normal",
                "ST",
    #MaxHR
            "MaxHR",
    #ExerciseAngina
            "ExerciseAngina",
    #Oldpeak
            "Oldpeak",
    #ST_Slope
                "Down",
                "Flat",
                "Up",
    #Oldpeak&ST_Slope
            "M_ST_Nomal-or-Error"
          ]


    X_train = X_train[feature]

    X_test = X_test[feature]

    clf_SVM = make_pipeline(StandardScaler(),
                            SVC(kernel = "rbf", C = 3.98450120759242,
                                gamma = 0.016899785453654257, random_state = 82))

    clf_SVM.fit(X_train,y_train)

    #trainの予測値
    train_pred = clf_SVM.predict(X_train)

    y_pred = clf_SVM.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('SVM Accuracy:', accuracy)


    return (accuracy, y_pred, train_pred)

#===================================================================

def Lr_set(X_train, y_train, X_test, y_test):
    features = ['SS_Age','Sex','SS_RestingBP','FastingBS','SS_MaxHR',
                'ExerciseAngina','chol0','chol1', 'chol2', 'chol3',
                'Risk0', 'Risk1', 'Risk2', 'Risk3', 'Risk4']

    #モデル作成
    clf_LC = LogisticRegression(C= 0.7384006850395533,max_iter=63733,penalty='l2')

    #変数
    X_train = X_train[features]

    X_test = X_test[features]

    clf_LC.fit(X_train, y_train)

    #trainの予測値
    train_pred = clf_LC.predict(X_train)

    y_pred = clf_LC.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('LR Accuracy:', accuracy)


    return (accuracy, y_pred, train_pred)

#===================================================================

def Knn_set(X_train, y_train, X_test, y_test):

    #ラベル特徴量
    feature = ["SS_Age", "Sex", "G_ChestPainType", "SS_RestingBP", "FastingBS", "RestingECG",
                "SS_MaxHR", "ExerciseAngina", "G_H_risk"]

    KNN=KNeighborsClassifier(n_neighbors=18, algorithm ='auto', p= 1,metric='minkowski')

    #変数
    X_train = X_train[feature]

    X_test = X_test[feature]

    KNN.fit(X_train, y_train)

    #trainの予測値
    train_pred = KNN.predict(X_train)

    y_pred = KNN.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('KNN Accuracy:', accuracy)


    return (accuracy, y_pred, train_pred)

#===================================================================

def Rfc_set(X_train, y_train, X_test, y_test):

    #ラベル特徴量
    feature = ["Age", "Sex", "K_ChestPain_group", "RestingBP", "G_CholCat",
               "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "G_H_risk"
              ]

    clf = RFC(n_estimators = 525,
              max_depth = 84,
              min_samples_split = 4,
              min_samples_leaf = 5,
              max_features = "auto",
              random_state=42
              )

    #変数
    X_train = X_train[feature]

    X_test = X_test[feature]

    clf.fit(X_train, y_train)

    #trainの予測値
    train_pred = clf.predict(X_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('RFC Accuracy:', accuracy)


    return (accuracy, y_pred, train_pred)

#===================================================================

def Xgb_set(X_train, y_train, X_test, y_test):

    #ラベル特徴量
    feature = ["Age", "Sex", "RestingBP", "Cholesterol", "FastingBS",
               "RestingECG", "MaxHR", "ExerciseAngina", "G_H_risk"
              ]

    XGB=XGBClassifier(max_depth = 3,learning_rate = 0.621585751698358,
                      reg_lambda = 5.165310706448221)

    #変数
    X_train = X_train[feature]
    X_test = X_test[feature]

    XGB.fit(X_train, y_train)

    #trainの予測値
    train_pred = XGB.predict(X_train)

    y_pred = XGB.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('XGB Accuracy:', accuracy)


    return (accuracy, y_pred, train_pred)

#===================================================================

# 各5つのモデルの正答率を保存するリストの初期化
SVM_accuracies = []
LR_accuracies = []
KNN_accuracies = []
RFC_accuracies = []
XGB_accuracies = []

# 学習のカウンター
loop_counts = 1

# 各クラスの確率（1モデル*5seed*1クラス）
train_probs = pd.DataFrame(np.zeros((len(data_tmp), 5*1*1)))
test_probs = pd.DataFrame(np.zeros((len(data_tmp), 5*1*1)))

X_train = data_tmp
y_train = y
X_test = data_test
y_test = y_data_test

# SVMの訓練を実行
svm_accuracy, svm_prob, svm_train_prob = Svm_set(X_train, y_train,
                                                 X_test, y_test
                                                )

# LRの訓練を実行
lr_accuracy, lr_prob, lr_train_prob = Lr_set(X_train, y_train,
                                             X_test, y_test
                                            )

# KNNの訓練を実行
knn_accuracy, knn_prob, knn_train_prob = Knn_set(X_train, y_train,
                                                 X_test, y_test
                                                )

# RFCの訓練を実行
rfc_accuracy, rfc_prob, rfc_train_prob = Rfc_set(X_train, y_train,
                                                 X_test, y_test
                                                )


# XGBの訓練を実行
xgb_accuracy, xgb_prob, xgb_train_prob = Xgb_set(X_train, y_train,
                                                 X_test, y_test
                                                )


# 実行回数のカウント
loop_counts += 1

# 学習が終わったモデルの正答率をリストに入れておく
SVM_accuracies.append(svm_accuracy)
LR_accuracies.append(lr_accuracy)
KNN_accuracies.append(knn_accuracy)
RFC_accuracies.append(rfc_accuracy)
XGB_accuracies.append(xgb_accuracy)

train_probs.iloc[:, 0] = svm_train_prob[:]
train_probs.iloc[:, 1] = lr_train_prob[:]
train_probs.iloc[:, 2] = knn_train_prob[:]
train_probs.iloc[:, 3] = rfc_train_prob[:]
train_probs.iloc[:, 4] = xgb_train_prob[:]

test_probs.iloc[:, 0] = svm_prob[:]
test_probs.iloc[:, 1] = lr_prob[:]
test_probs.iloc[:, 2] = knn_prob[:]
test_probs.iloc[:, 3] = rfc_prob[:]
test_probs.iloc[:, 4] = xgb_prob[:]



# 予測結果の格納用のnumpy行列を作成
#test_preds = np.zeros((len(y), 5))
#test_preds = []
acc_list = []

X_train_cv = train_probs
y_train_cv = y
X_test_cv = test_probs
y_test_cv = y_data_test

# データを格納する
# 学習用
xgb_train = xgb.DMatrix(X_train_cv, label=y_train_cv)
# テスト用
xgb_test = xgb.DMatrix(X_test_cv, label=y_test_cv)

xgb_params = {
    "objective": "binary:logistic", # 2値分類問題
}


# 学習
evals = [(xgb_train, 'train')] # 学習に用いる検証用データ

evaluation_results = {}                            # 学習の経過を保存する箱
bst = xgb.train(xgb_params,                        # 上記で設定したパラメータ
                xgb_train,                         # 使用するデータセット
                num_boost_round=200,               # 学習の回数
                early_stopping_rounds=10,          # アーリーストッピング
                evals=evals,                       # 学習経過で表示する名称
                evals_result=evaluation_results,   # 上記で設定した検証用データ
                verbose_eval=0                     # 学習の経過の表示(非表示)
                )


y_pred_proba = bst.predict(xgb_test, ntree_limit=bst.best_ntree_limit)
y_pred = np.where(y_pred_proba > 0.5, 1, 0)


# testの予測を保存
#test_preds.extend(y_pred)

acc = accuracy_score(y_test, y_pred)
acc_list.append(acc)

print('テストデータのAccuracy:', acc)
