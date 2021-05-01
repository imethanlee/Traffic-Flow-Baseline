import pandas as pd
import numpy as np


class Utils:
    mean = 0
    std = 0
    null_val = None

    @staticmethod
    def fit(x: np.ndarray):
        len_x = len(x)
        Utils.mean = np.sum(x) / len_x
        Utils.std = np.std(x)
        Utils.null_val = (0. - Utils.mean) / Utils.std

    @staticmethod
    def z_score(x: np.ndarray):
        z = (x - Utils.mean) / Utils.std
        return z

    @staticmethod
    def inverse_z_score(z: np.ndarray):
        x = z * Utils.std + Utils.mean
        return x

    @staticmethod  # 存档，可能在其他项目用
    def weight_matrix_preprocessing(file_path, normalized_k=0.1):
        dist_mat = pd.read_csv(file_path, header=None).to_numpy()
        std = np.std(dist_mat.flatten())
        adj_mat = np.exp(-np.square(dist_mat / std))
        adj_mat[adj_mat < normalized_k] = 0
        return adj_mat
