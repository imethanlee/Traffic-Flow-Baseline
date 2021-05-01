import numpy as np
# from statsmodels.tsa.arima import model
# from statsmodels.tsa.vector_ar import var_model
from sklearn.svm import LinearSVR
from data.loader import *
from utils.masked_loss import *


class Helper:
    @staticmethod
    def retrieve_data(n_time, out_time, train_pct, test_pct, v_path, w_path):
        data = TrafficFlowData(torch.device("cpu"), v_path, w_path, train_pct, test_pct, n_time, out_time)
        return data.train_x.numpy(), data.train_y.numpy(), data.test_x.numpy(), data.test_y.numpy()

    @staticmethod
    def metrics(y_pred, y_true, normalize=True):
        if normalize:
            y_pred = Utils.inverse_z_score(y_pred)
            y_true = Utils.inverse_z_score(y_true)

        mae = masked_loss_np(y_pred, y_true, "mae")
        mape = masked_loss_np(y_pred, y_true, "mape")
        rmse = np.sqrt(masked_loss_np(y_pred, y_true, "rmse"))

        print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(mae, mape, rmse))
        return mae, mape, rmse


class HA:
    def __init__(self):
        super(HA, self).__init__()
        self.period = 12 * 24 * 7

    def predict(self, df):
        period = self.period

        n_sample, n_node = df.shape
        n_test = int(round(n_sample * 0.2))
        n_train = n_sample - n_test
        y_true = df[-n_test:]
        y_pred = np.zeros_like(y_true)

        for i in range(n_train, min(n_sample, n_train + period)):
            inds = [j for j in range(i % period, n_train, period)]
            historical = df[inds, :]
            y_pred[i - n_train, :] = historical[historical != 0.].mean()

        for i in range(n_train + period, n_sample, period):
            size = min(period, n_sample - i)
            start = i - n_train
            y_pred[start:start + size, :] = y_pred[start - period: start + size - period, :]

        return y_pred.reshape(-1), y_true.reshape(-1)

    def eval(self, v_path):
        df = pd.read_csv(v_path, header=None).to_numpy()
        y_pred, y_true = self.predict(df)
        Helper.metrics(y_pred, y_true, False)


"""
class ARIMA:
    def __init__(self, df):
        super(ARIMA, self).__init__()
        self.model = model.ARIMA(df, (3, 0, 1))
        self.model.fit()

    def predict(self, df=None):
        return self.model.fit()


class VAR:
    def __init__(self):
        super(VAR, self).__init__()
        self.maxlags = 3
        self.model = None

    def fit(self, df, out_time):
        n_forwards = out_time
        n_sample, n_output = df.shape
        n_test = int(round(n_sample * 0.2))
        n_train = n_sample - n_test
        df_train, df_test = df[:n_train], df[-n_test:]

        # scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
        # data = scaler.transform(df_train.values)
        self.model = var_model.VAR(df)
        var_result = self.model.fit(self.maxlags)
        max_n_forwards = np.max(n_forwards)

        # Do forecasting.
        result = np.zeros(shape=(len(n_forwards), n_test, n_output))
        start = n_train - self.maxlags - max_n_forwards + 1
        for input_ind in range(start, n_sample - self.maxlags):
            prediction = var_result.forecast(df.values[input_ind: input_ind + self.maxlags], max_n_forwards)
            for i, n_forward in enumerate(n_forwards):
                result_ind = input_ind - n_train + self.maxlags + n_forward - 1
                if 0 <= result_ind < n_test:
                    result[i, result_ind, :] = prediction[n_forward - 1, :]

        df_predicts = []
        for i, n_forward in enumerate(n_forwards):
            df_predict = pd.DataFrame(result[i], index=df_test.index, columns=df_test.columns)
            df_predicts.append(df_predict)
        return np.array(df_predicts).reshape(-1), np.array(df_test).reshape(-1)

    def eval(self, out_time, v_path):
        df = pd.read_csv(v_path, header=None)
        print("VAR Fitting...")
        y_pred, y_true = self.fit(df, [out_time])
        print("VAR Fitted!")
        Helper.metrics(y_pred, y_true, normalize=False)
"""


class LSVR:
    def __init__(self):
        super(LSVR, self).__init__()
        self.C = 0.1
        self.n_time = 5
        self.model = LinearSVR(C=self.C)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def eval(self, out_time, v_path, w_path):
        train_x, train_y, test_x, test_y = Helper.retrieve_data(
            n_time=5,
            out_time=out_time,
            train_pct=0.7,
            test_pct=0.2,
            v_path=v_path,
            w_path=w_path
        )
        train_x = np.squeeze(train_x.transpose((0, 2, 1, 3))).reshape(-1, self.n_time)
        test_x = np.squeeze(test_x.transpose((0, 2, 1, 3))).reshape(-1, self.n_time)
        train_y = train_y.reshape(-1)
        test_y = test_y.reshape(-1)
        print("LSVR Fitting...")
        self.model.fit(train_x, train_y)
        print("LSVR Fitted!")
        y_pred = self.model.predict(test_x)
        Helper.metrics(y_pred, test_y)
