import pickle
from model_preprocessing import data_preprocessing

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

path = "test_data.csv"

clf = load_model('model.pkl')

df = data_preprocessing(path)

X_test, y_test = df.drop('Target_regression', axis=1), df['Target_regression']
y_pred = clf.predict(X_test)


print('The testing was successful')

save = input('Saving a prediction ? (y/n): ')
if save == 'y':
    with open('predictions.txt', 'w') as f:
        f.write(str(y_pred))
