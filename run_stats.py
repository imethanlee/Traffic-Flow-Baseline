from model.stats_model import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="lsvr")
parser.add_argument('--v_path', type=str, default='./data/v_pems_228.csv')
parser.add_argument('--w_path', type=str, default='./data/w_pems_228.csv')
parser.add_argument('--out_time', type=int, default=3)

args = parser.parse_args()

if args.model == "ha":
    pass
if args.model == "arima":
    pass
if args.model == "var":
    pass
if args.model == "lsvr":
    lsvr = LSVR()
    lsvr.eval(args.out_time, args.v_path, args.w_path)
