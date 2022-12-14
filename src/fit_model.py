from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from torch.utils.data import TensorDataset, DataLoader
from fc_network import FC_Net

import pandas as pd
import numpy as np
import torch
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


class MusicPopularityPredictor:
    def __init__(self, file_path):
        df = pd.read_csv(file_path, sep=",", encoding="ANSI")
        df.dropna(axis=0, inplace=True)
        self.df = df.reset_index(drop=True)
        print("Data loaded!")

        features = self.df.drop(columns=["hotness", "title", "album_name", "artist_terms"], axis=1).values
        labels = self.df["hotness"].values

        # normalization
        col_min = np.min(features, axis=0)
        col_max = np.max(features, axis=0)
        features = (features - col_min) / (col_max - col_min)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(features, labels, test_size=0.1)

        self.train_class_labels = np.asarray(list(map(lambda x: 0 if x < 0.5 else 1, self.y_train)))
        self.test_class_labels = np.asarray(list(map(lambda x: 0 if x < 0.5 else 1, self.y_test)))

    def fit_with_RF(self, n_jobs=12):
        rf_regressor = RandomForestRegressor(n_jobs=n_jobs, n_estimators=1000)

        print("Start fitting with RandomForest!")
        start = time.perf_counter()
        rf_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        print("Training with RandomForest done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = rf_regressor.score(self.x_train, self.y_train)
        test_score = rf_regressor.score(self.x_test, self.y_test)

        prediction = rf_regressor.predict(self.x_test)
        class_pred = list(map(lambda x: 0 if x < 0.5 else 1, prediction))
        test_score_2 = accuracy_score(self.test_class_labels, class_pred)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)
        print("\tClassification Score:", test_score_2)

    def fit_with_XGBR(self, n_jobs=12):
        xgb_regressor = XGBRegressor(n_jobs=n_jobs, n_estimators=1000)

        print("Start fitting with XG Boost!")
        start = time.perf_counter()
        xgb_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        print("Training with XG Boost done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = xgb_regressor.score(self.x_train, self.y_train)
        test_score = xgb_regressor.score(self.x_test, self.y_test)

        prediction = xgb_regressor.predict(self.x_test)
        class_pred = list(map(lambda x: 0 if x < 0.5 else 1, prediction))
        test_score_2 = accuracy_score(self.test_class_labels, class_pred)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)
        print("\tClassification Score:", test_score_2)

    def fit_with_LinearRegression(self, n_jobs=12):
        xgb_regressor = LinearRegression(n_jobs=n_jobs)

        print("Start fitting with LinearRegression!")
        start = time.perf_counter()
        xgb_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        print("Training with LinearRegression done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = xgb_regressor.score(self.x_train, self.y_train)
        test_score = xgb_regressor.score(self.x_test, self.y_test)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)

    def grid_search_for_RF(self, n_jobs=12):
        param_grid = {"n_estimators": [100, 200, 400],
                      "criterion": ["squared_error", "absolute_error", "poisson"]}
        grid_search = GridSearchCV(RandomForestRegressor(n_jobs=n_jobs),
                                   param_grid=param_grid,
                                   return_train_score=True,
                                   cv=2,
                                   verbose=3,
                                   n_jobs=1)
        print("\nStart Grid Searching for RandomForest...")
        start = time.perf_counter()
        grid_search.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start
        print("\tUsed cores:", n_jobs)
        print("\tSearching time: %f seconds" % wall_clock_time)

    def grid_search_for_XGBR(self, n_jobs=12):
        param_grid = {"n_estimators": [100, 200, 400],
                      "max_depth": [3, 6, 9]}
        grid_search = GridSearchCV(RandomForestRegressor(n_jobs=n_jobs),
                                   param_grid=param_grid,
                                   return_train_score=True,
                                   cv=2,
                                   verbose=3,
                                   n_jobs=1)
        print("\nStart Grid Searching for XG Boost...")
        start = time.perf_counter()
        grid_search.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start
        print("\tUsed cores:", n_jobs)
        print("\tSearching time: %f seconds" % wall_clock_time)

    def fit_with_fc_network(self, batch_size=64, epochs=1000):
        # transfer pd DataFrame to torch tensor and to torch DataLoader
        train_dataset = TensorDataset(torch.tensor(self.x_train, dtype=torch.float32),
                                      torch.tensor(self.y_train, dtype=torch.float32))
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        data_num = len(train_data_loader)

        y_test = self.y_test

        fc_network = FC_Net(input_dims=self.x_train.shape[1], output_dims=1)
        # print(fc_network)
        if os.path.exists('fc_latest.pth'):
            fc_network.load_state_dict(torch.load('./models/fc_latest.pth'))
            print("Model Loaded!")

        optimizer = torch.optim.Adam(fc_network.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_function = torch.nn.MSELoss()

        # automatically detect if using GPU and if multi GPUs
        if torch.cuda.is_available():
            print("Pytorch with CUDA available!")
            fc_network = fc_network.cuda()
            if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
                fc_network = torch.nn.DataParallel(fc_network)

        # set up training procedure
        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_loss = 0
            train_score = 0
            train_score_2 = 0
            for iteration, data in enumerate(train_data_loader):
                x, y = data
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                prediction = fc_network(x).squeeze(1)

                # back forwarding
                loss = loss_function(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # use R2 Score to evaluate
                prediction = prediction.cpu().detach().numpy()
                train_score += r2_score(y.cpu().detach().numpy(), prediction)

            train_score /= data_num
            train_score_2 /= data_num
            epoch_time = time.perf_counter() - start_time

            # transfer pd DataFrame to torch tensor
            to_predict = torch.tensor(self.x_test, dtype=torch.float32)
            if torch.cuda.is_available():
                to_predict = to_predict.cuda()
            prediction = fc_network(to_predict).squeeze(1).cpu().detach().numpy()

            test_score = r2_score(y_test, prediction)
            class_pred = list(map(lambda x: 0 if x < 0.5 else 1, prediction))
            test_score_2 = accuracy_score(self.test_class_labels, class_pred)

            # print(prediction[:5])
            # print(y_test[:5])

            print('\tepoch: {} / {}  Validation Score: {}  Validation Score 2: {}  Training Score: {} '
                  .format(epoch+1, epochs, test_score, test_score_2, train_score))
            print("\tEpoch time: %.3f s\tEpoch Loss: %f" % (epoch_time, epoch_loss))

            # save the model
            if epoch % 100 == 0:
                if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
                    torch.save(fc_network.module.state_dict(), './models/fc_latest.pth')
                else:
                    torch.save(fc_network.state_dict(), './models/fc_latest.pth')
                print("Model Saved!")


if __name__ == '__main__':
    my_music_popularity_predictor = MusicPopularityPredictor("./processed/whole_data.csv")
    my_music_popularity_predictor.fit_with_RF(n_jobs=12)
    my_music_popularity_predictor.fit_with_XGBR(n_jobs=12)
    my_music_popularity_predictor.fit_with_LinearRegression(n_jobs=12)

    # my_music_popularity_predictor.grid_search_for_XGBR(n_jobs=12)
    # my_music_popularity_predictor.grid_search_for_RF(n_jobs=12)

    my_music_popularity_predictor.fit_with_fc_network(batch_size=256, epochs=5000)
