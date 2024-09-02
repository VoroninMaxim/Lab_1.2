from sklearn.model_selection import train_test_split
from model_preprocessing import data_preprocessing
from sklearn.ensemble import RandomForestRegressor
import pickle

clf = RandomForestRegressor(random_state=42)

path = ('train_data.csv')

df = data_preprocessing(path)

X_train, X_test, y_train, y_test = train_test_split(df.drop(
    'Target_regression', axis=1), df['Target_regression'], test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

print('model saved')