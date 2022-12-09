from sklearn.ensemble import RandomForestRegressor
import pandas as pd


"""
This is a file for simple fitting without NLP processing.
Just for exploring, testing and enhancing the data.
"""


if __name__ == '__main__':
    # utf-8 encoding will fail
    df = pd.read_csv("../processed/10496.csv", sep=",", encoding="ANSI")
    df.dropna(axis=0, inplace=True)
    df = df.reset_index(drop=True)

    # drop features related to NLP here
    feat_name = df.columns.drop(["title", "album_name", "artist_terms"])
    features = [df[column] for column in feat_name]
    whole_data = pd.concat(features, axis=1).astype(dtype='float64', copy=False)

    train = []
    to_predict = []
    # some data do not have "hotness" value, save them for prediction,
    # others for training
    for i in range(whole_data.shape[0]):
        if whole_data.loc[i][0] >= 0:
            train.append(whole_data.loc[i])
        else:
            to_predict.append(whole_data.loc[i])

    # generate training and testing dataset
    train = pd.concat(train, axis=1).astype(dtype='float64', copy=False).transpose()
    train = train.dropna(axis=0)
    train_y = train["hotness"]
    train_x = train.drop("hotness", axis=1)

    to_predict = pd.concat(to_predict, axis=1).astype(dtype='float64', copy=False).transpose()
    to_predict = to_predict.drop("hotness", axis=1)
    to_predict = to_predict.dropna(axis=0)

    # training
    my_rf_regressor = RandomForestRegressor(n_jobs=12)
    my_rf_regressor.fit(train_x, train_y)
    score = my_rf_regressor.score(train_x, train_y)
    print("training score:", score)

    # predicting and save the result for future use
    prediction = my_rf_regressor.predict(to_predict)
    print(prediction, min(prediction), max(prediction))

    idx = 0
    for i in range(df.shape[0]):
        if df.loc[i, "hotness"] >= 0:
            continue
        else:
            df.loc[i, "hotness"] = prediction[idx]
            idx += 1

    print(df)
    df.to_csv("./processed/whole_data.csv", index=False, encoding="ANSI")
