from statsmodels.tsa import arima_model
from statsmodels.tsa.vector_ar import var_model
from sklearn.svm import LinearSVR
from data.loader import *


class Helper:
    @staticmethod
    def retrieve_data(n_time, out_time, train_pct, test_pct, v_path, w_path):
        data = TrafficFlowData(torch.device("cpu"), v_path, w_path, train_pct, test_pct, n_time, out_time)
        return data.train_x.numpy(), data.train_y.numpy(), data.test_x.numpy(), data.test_y.numpy()

    @staticmethod
    def metrics(y_pred, y_true):
        y_pred = Utils.inverse_z_score(y_pred)
        y_true = Utils.inverse_z_score(y_true)
        d = np.abs(y_true - y_pred)
        ae = d.tolist()
        ape = (np.divide(d, y_true)).tolist()
        se = (np.power(d, 2)).tolist()
        mae = np.array(ae).mean()
        mape = np.array(ape).mean()
        rmse = np.sqrt(np.array(se).mean())
        print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(mae, mape, rmse))
        return mae, mape, rmse


class HA:
    def __init__(self):
        super(HA, self).__init__()


class ARIMA:
    def __init__(self):
        super(ARIMA, self).__init__()
        self.model = arima_model.ARIMA(None, (3, 0, 1))

    def fit(self, train_x, train_y):
        self.model.fit()

    def predict(self):
        pass


class VAR:
    def __init__(self):
        super(VAR, self).__init__()
        self.maxlags = 3
        self.model = var_model.VAR()

    def fit(self):
        self.model.fit(self.maxlags)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def eval(self, n_time):
        pass


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
        train_x = np.squeeze(train_x.transpose((0, 2, 1, 3))).reshape(-1, 5)
        test_x = np.squeeze(test_x.transpose((0, 2, 1, 3))).reshape(-1, 5)
        train_y = train_y.reshape(-1)
        test_y = test_y.reshape(-1)
        print("LSVR Fitting...")
        self.model.fit(train_x, train_y)
        print("LSVR Fitted!")
        y_pred = self.model.predict(test_x)
        Helper.metrics(y_pred, test_y)



