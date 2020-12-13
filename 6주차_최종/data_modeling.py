# 필요한 라이브러리 불러오기

import warnings
warnings.filterwarnings(action='ignore')

import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot
from scipy.stats import pearsonr
#환경 설정
matplotlib.pyplot.rcdefaults()
matplotlib.pyplot.rcParams["font.family"] = 'Haansoft Dotum'
matplotlib.pyplot.rcParams['axes.unicode_minus'] = False


# 모델링 자동화 코드
def modeling(x_in, x_out):
    start = time.time()  # 시작 시간 저장
    X_train, X_test, y_train, y_test= train_test_split(x_in, x_out, test_size=0.2, random_state=42)

    # xgboost 학습

    model = XGBRegressor(booster="gbtree", objective ='reg:squarederror', n_estimators=3000, learning_rate=0.001 ,
                      max_depth=12, n_jobs = -1,subsample=0.75, reg_lambda=1, colsample_bytree=1, gamma=0, )

    eval_set = [(X_test, y_test)]

    model.fit(X_train,y_train, eval_set=eval_set, verbose=True)

    pred_y = model.predict(X_test)

    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    return model # 생성된 모델 반환

# 모델 성능 시각화 코드
def visual_model(model, x_in, x_out):

    # 시계열 index 생성
    rng = pd.date_range('1/1/2016', periods=18, freq='Q')

    x_out_list = []
    for i in range(len(x_out)):
        n = x_out.values[i][0]
        x_out_list.append(n)
    fig, ax =matplotlib.pyplot.subplots(figsize=(15,5))
    sns.lineplot(x = rng, y = model.predict(x_in),alpha = 0.7,linewidth = 2, ax = ax, label = '모델 예측값')
    sns.lineplot(x = rng, y = x_out_list, alpha = 0.7,linewidth = 2, ax = ax, label = '실제값')
    ax.set_title('폐업률 예측 모델 성능', size = 20)
    ax.set_ylabel('폐업률', size = 13)
    ax.set_xlabel('연도', size = 13)
    # 파란색 : 예측값
    # 주황색 : 실제값
    # 갈색 : 파란색과 주황색이 겹치는 부분

# Feature Importance 시각화 코드
def feature_importance(model) :
    # 중요도 시각화

    feature_important = model.get_booster().get_score(importance_type='weight')
    print(feature_important)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    fig, ax = matplotlib.pyplot.subplots(figsize=(20,7))
    data_50 = pd.DataFrame(data = values, index = keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data_50[:10].plot(kind='barh', ax = ax, label = '')
    ax.set_title('변수 중요도', size = 20)
    ax.set_xlabel('Score', size = 20)
