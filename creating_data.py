import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

loc, scale = 0, 1

n_samples = 10000
features = {
    'feature_1':np.random.logistic(loc, scale, n_samples),
    'feature_2':np.random.poisson(20, n_samples),
    'feature_3':np.random.rand(n_samples),
    'feature_4':np.random.rand(n_samples),
    'feature_5':np.random.rand(n_samples)
}

features['Target_regression'] = (
    2 * features['feature_1']
    * 3 * features['feature_2']
    * 0.5 * features['feature_3']
    * np.random.normal(loc=0, scale=0.1, size=n_samples)
)

df_train = pd.DataFrame(features).sample(frac=0.9, random_state=42)
df_test = pd.DataFrame(features).drop(df_train.index)

file_path_train = Path('train_data.csv')
file_path_train.parent.mkdir(parents=True, exist_ok=True)

file_path_test = Path('test_data.csv')
file_path_test.parent.mkdir(parents=True, exist_ok=True)

df_train.to_csv(file_path_train, index=False)
df_test.to_csv(file_path_test, index=False)

print("Фаил Тренировочных данных : ", file_path_train)
print("Фаил тестовых данных : ", file_path_test)

