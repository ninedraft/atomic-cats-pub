# %%
from joblib import load
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np


def predict(data, predictors):
    names = ['Data'] + list(predictors.keys())
    table = {'Data': data}
    for name, predict in predictors.items():
        table[name] = predict(data)
    return pd.DataFrame(table, columns=names)


def todense(vectors):
    return [np.asarray(item.todense())[0] for item in vectors]

def load_catboost_predictor(name):
    clf = CatBoostClassifier()
    clf.load_model(f"{name}.cbm")
    le = load(f"{name}_le.job")
    vect = load(f"{name}_vect.job")

    def predict(data):
        compressed_data = vect.transform(data)
        vect_data = todense(compressed_data)
        preds = clf.predict(vect_data).flatten().astype('int64')
        return le.inverse_transform(preds).flatten()
    return predict


def get_labels(df, column):
    return [item.lower() for item in df['Тип упаковки'].values.flatten()]


# %%
beer = pd.read_csv('MASTER_DATA_beer_hackaton.csv', sep=';')

data = [str(item) for item in beer[['o.item_name']].values.flatten()]

# labels = get_labels(beer, 'Тип упаковки')
predictors = {
    'Тип упаковки': load_catboost_predictor('beer_container_catboost_400')
}

result = predict(data, predictors)

print(result.head())
result.to_csv('output.csv')
# %%
