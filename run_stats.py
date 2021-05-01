from model.stats_model import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ha")
parser.add_argument('--v_path', type=str, default='./data/v_pems_325 (missing values).csv')
parser.add_argument('--w_path', type=str, default='./data/w_pems_325.csv')
parser.add_argument('--out_time', type=int, default=12)

args = parser.parse_args()

if args.model == "ha":
    ha = HA()
    ha.eval(args.v_path)
if args.model == "arima":
    df = np.array([1, 2, 3])
    arima = ARIMA(df)
    print(arima.predict().summary())
if args.model == "var":
    var = VAR()
    var.eval(args.out_time, args.v_path)
if args.model == "lsvr":
    lsvr = LSVR()
    lsvr.eval(args.out_time, args.v_path, args.w_path)
