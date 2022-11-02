import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def logloss_crutch(y_true, y_pred, eps=1e-15):
    # Логистическая функция потерь
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    df = pd.read_csv('./Data/data.csv')
    binary = {'Yes': 1, 'No': 0}

    df['International plan'] = df['International plan'].map(binary)
    df['Voice mail plan'] = df['Voice mail plan'].map(binary)
    df['Churn'] = df['Churn'].astype('int64')

    le = LabelEncoder()
    df['State'] = le.fit_transform(df['State'])

    ohe = OneHotEncoder(sparse=False)

    encoded_state = ohe.fit_transform(df['State'].values.reshape(-1, 1))
    tmp = pd.DataFrame(encoded_state,
                       columns=['state ' + str(i) for i in range(encoded_state.shape[1])])
    df = pd.concat([df, tmp], axis=1)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Делим выборку на train и test, все метрики будем оценивать на тестовом датасете

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

    # Обучаем ставшую родной логистическую регрессию

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    print(lr.score(X_train, y_train))
    # print('Logloss при неуверенной классификации %f' % logloss_crutch(1, 0.5))
    # # >> Logloss при неуверенной классификации 0.693147
    #
    # print('Logloss при уверенной классификации и верном ответе %f' % logloss_crutch(1, 0.9))
    # # >> Logloss при уверенной классификации и верном ответе 0.105361
    #
    # print('Logloss при уверенной классификации и НЕверном ответе %f' % logloss_crutch(1, 0.1))
    # # >> Logloss при уверенной классификации и НЕверном ответе 2.302585

if __name__ == '__main__':
    main()
