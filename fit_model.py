from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from torch.utils.data import TensorDataset, DataLoader
from fc_network import FC_Net

import pandas as pd
import torch
import time
import os


class MusicPopularityPredictor:
    def __init__(self, file_path):
        df = pd.read_csv(file_path, sep=",", encoding="utf-8")
        df.dropna(axis=0, inplace=True)
        self.df = df.reset_index(drop=True)
        print("Data loaded!")

        features = self.df.drop(columns=["hotness", "title", "album_name", "artist_terms"], axis=1)
        labels = self.df["hotness"]

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(features, labels, test_size=0.1)

    def fit_with_RF(self, n_jobs=12):
        rf_regressor = RandomForestRegressor(n_jobs=n_jobs)

        print("Start fitting with RandomForest!")
        start = time.perf_counter()
        rf_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        # 1 - 1140.630814
        # 2 - 611.533856
        # 4 - 355.727361
        # 6 - 275.180256
        # 8 - 246.753700
        # 10 - 205.206498
        # 12 - 191.179174
        print("Training with RandomForest done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = rf_regressor.score(self.x_train, self.y_train)
        test_score = rf_regressor.score(self.x_test, self.y_test)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)

    def fit_with_XGBR(self, n_jobs=12):
        xgb_regressor = XGBRegressor(n_jobs=n_jobs)

        print("Start fitting with XG Boost!")
        start = time.perf_counter()
        xgb_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        # 1 - 249.251679
        # 2 - 132.333319
        # 4 - 78.611960
        # 6 - 60.947870
        # 8 - 53.302667
        # 10 - 50.109146
        # 12 - 45.899494
        print("Training with XG Boost done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = xgb_regressor.score(self.x_train, self.y_train)
        test_score = xgb_regressor.score(self.x_test, self.y_test)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)

    def fit_with_LinearRegression(self, n_jobs=12):
        xgb_regressor = LinearRegression(n_jobs=n_jobs)

        print("Start fitting with LinearRegression!")
        start = time.perf_counter()
        xgb_regressor.fit(self.x_train, self.y_train)
        wall_clock_time = time.perf_counter() - start

        # 1 - 249.251679
        # 2 - 132.333319
        # 4 - 78.611960
        # 6 - 60.947870
        # 8 - 53.302667
        # 10 - 50.109146
        # 12 - 45.899494
        print("Training with LinearRegression done!")
        print("\tUsed cores:", n_jobs)
        print("\tTraining time: %f seconds" % wall_clock_time)
        train_score = xgb_regressor.score(self.x_train, self.y_train)
        test_score = xgb_regressor.score(self.x_test, self.y_test)

        print("\tTraining score:", train_score)
        print("\tTesting score:", test_score)

    def fit_with_fc_network(self, epochs=1000):
        train_dataset = TensorDataset(torch.tensor(self.x_train.to_numpy(), dtype=torch.float32),
                                      torch.tensor(self.y_train.to_numpy(), dtype=torch.float32))
        train_data_loader = DataLoader(train_dataset, batch_size=64)
        data_num = len(train_data_loader)

        fc_network = FC_Net(input_dims=self.x_train.shape[1], output_dims=1)
        # print(fc_network)
        if os.path.exists('fc_latest.pth'):
            fc_network.load_state_dict(torch.load('fc_latest.pth'))
            print("Model Loaded!")

        optimizer = torch.optim.Adam(fc_network.parameters(), lr=1e-5, weight_decay=1e-5)
        loss_function = torch.nn.MSELoss()

        if torch.cuda.is_available():
            fc_network = fc_network.cuda()

        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_loss = 0
            train_score = 0
            for iteration, data in enumerate(train_data_loader):
                x, y = data
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                prediction = fc_network(x).squeeze(1)
                loss = loss_function(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_score += r2_score(y.cpu().detach().numpy(),
                                        prediction.cpu().detach().numpy())
                # if iteration % 10 == 0:
                #     print('\t\tepoch: {}  {} / {}  Training Score: {}'
                #           .format(epoch, iteration, data_num, train_score))

            train_score /= data_num
            epoch_time = time.perf_counter() - start_time
            to_predict = torch.tensor(self.x_test.to_numpy(), dtype=torch.float32).cuda()
            prediction = fc_network(to_predict).squeeze(1).cpu().detach().numpy()
            test_score = r2_score(self.y_test, prediction)
            print('\tepoch: {} / {}  Validation Score: {}  Training Score: {}'
                  .format(epoch+1, epochs, test_score, train_score))
            print("\tEpoch time: %.3f s\tEpoch Loss: %f" % (epoch_time, epoch_loss))
            if epoch % 100 == 0:
                torch.save(fc_network.state_dict(), 'fc_latest_2.pth')
                # torch.save(fc_network.module.state_dict(), 'fc_latest.pth')
                print("Model Saved!")

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


if __name__ == '__main__':
    my_music_popularity_predictor = MusicPopularityPredictor("./processed/enhanced_data.csv")
    # my_music_popularity_predictor.fit_with_RF(n_jobs=12)
    # my_music_popularity_predictor.fit_with_XGBR(n_jobs=12)

    # my_music_popularity_predictor.grid_search_for_XGBR(n_jobs=12)
    # my_music_popularity_predictor.grid_search_for_RF(n_jobs=12)

    my_music_popularity_predictor.fit_with_fc_network(epochs=1000)
