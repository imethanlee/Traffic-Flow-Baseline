import torch
from torch.utils.data import TensorDataset
from scipy.linalg import fractional_matrix_power
from utils.utils import *


class TrafficFlowData:
    def __init__(self, device: torch.device, v_path: str, w_path: str, train_pct: float, test_pct: float,
                 n_time: int = 12, out_time: int = 3):
        self.device = device
        self.v_path = v_path
        self.w_path = w_path
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.n_time = n_time
        self.out_time = out_time
        self.train = None
        self.val = None
        self.test = None
        self.w_adj_mat = self.get_weighted_adjacency_matrix()
        self.n_node = self.w_adj_mat.shape[1]
        self.gen_data()

    def get_weighted_adjacency_matrix(self, sigma2=0.1, epsilon=0.5, scaling=True):
        df = pd.read_csv(self.w_path, header=None).to_numpy()
        return df

    def get_data(self):
        df = pd.read_csv(self.v_path, header=None)
        len_df = len(df)
        len_train = int(self.train_pct * len_df)
        len_test = int(self.test_pct * len_df)
        len_val = len_df - len_train - len_test
        train_data = df[: len_train].to_numpy()
        val_data = df[len_train: len_train + len_val].to_numpy()
        test_data = df[-len_test:].to_numpy()
        return train_data, test_data, val_data

    def transform_data(self, data: np.ndarray):
        # transform from row data to formatted data
        len_record = len(data)
        num_available_data = len_record - self.n_time - self.out_time

        x = np.zeros([num_available_data, self.n_time, self.n_node, 1])
        y = np.zeros([num_available_data, self.n_node])

        for i in range(num_available_data):
            start = i
            end = i + self.n_time
            x[i, :, :, :] = data[start: end].reshape(self.n_time, self.n_node, 1)
            y[i] = data[end + self.out_time - 1]

        return x, y

    def gen_data(self):
        # generate formatted data
        train_data, val_data, test_data = self.get_data()
        train_data_x, train_data_y = self.transform_data(train_data)
        val_data_x, val_data_y = self.transform_data(val_data)
        test_data_x, test_data_y = self.transform_data(test_data)
        Utils.fit(np.hstack((train_data_x.flatten(), train_data_y.flatten())))
        train_data_x = torch.Tensor(Utils.z_score(train_data_x)).to(self.device)
        train_data_y = torch.Tensor(Utils.z_score(train_data_y)).to(self.device)
        val_data_x = torch.Tensor(Utils.z_score(val_data_x)).to(self.device)
        val_data_y = torch.Tensor(Utils.z_score(val_data_y)).to(self.device)
        test_data_x = torch.Tensor(Utils.z_score(test_data_x)).to(self.device)
        test_data_y = torch.Tensor(Utils.z_score(test_data_y)).to(self.device)

        self.train = TensorDataset(train_data_x, train_data_y)
        self.val = TensorDataset(val_data_x, val_data_y)
        self.test = TensorDataset(test_data_x, test_data_y)

    def get_conv_kernel(self, approx: str):
        if approx == "linear":
            W_wave = np.eye(self.n_node, self.n_node) + self.w_adj_mat
            D_wave = np.diag(np.sum(W_wave, axis=1))
            kernel = np.matmul(
                np.matmul(fractional_matrix_power(D_wave, -0.5), W_wave),
                fractional_matrix_power(D_wave, -0.5)
            )
            # return torch.Tensor(kernel).to(self.device)
            # convert to sparse tensor
            r, c, v = [], [], []
            for i in range(len(kernel)):
                for j in range(len(kernel[0])):
                    if kernel[i, j] != 0:
                        r.append(i)
                        c.append(j)
                        v.append(kernel[i, j])
            index = torch.LongTensor([r, c])
            value = torch.FloatTensor(v)
            sparse_kernel = torch.sparse.FloatTensor(index, value, [self.n_node, self.n_node])
            return sparse_kernel.to(self.device)
        elif approx == "Cheb":
            pass
        else:
            raise ValueError("No such type!")

