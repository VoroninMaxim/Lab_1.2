import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_preprocessing(path):

    df = pd.read_csv(path, encoding='utf-8', sep=',')

    df_data = df.drop('Target_regression', axis=1)

    data_columns_name = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

    standart = StandardScaler()
    standart.fit(df_data)

    stand_transform = standart.transform(df_data[data_columns_name])
    df_stand_transform = pd.DataFrame(stand_transform, columns=data_columns_name)

    scaler = MinMaxScaler()
    scaler.fit(df_stand_transform)

    scaler_transform = scaler.transform(df_stand_transform)
    df_scaler_transform = pd.DataFrame(scaler_transform, columns=data_columns_name)

    df_prepared = pd.concat([df_scaler_transform, df['Target_regression']], axis=1)
    return df_prepared

