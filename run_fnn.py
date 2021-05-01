from model.fnn import *
from data.loader import *
from utils.early_stop import *
from utils.masked_loss import *
from torch.utils.data import DataLoader
from torchsummary import summary
import torch
import torch.optim as optim
import argparse
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patience', type=int, default=20)

parser.add_argument('--save_path', type=str, default='./model/save/')
parser.add_argument('--v_path', type=str, default='./data/v_pems_228.csv')
parser.add_argument('--w_path', type=str, default='./data/w_pems_228.csv')
parser.add_argument('--train_pct', type=float, default=0.7)
parser.add_argument('--test_pct', type=float, default=0.2)
parser.add_argument('--n_time', type=int, default=12)
parser.add_argument('--out_time', type=int, default=3)

parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()
model_save_path = args.save_path + "FNN_trained_" + args.v_path[-12:-4] + "_" + str(args.out_time)

device = args.device

# Data setup
data = TrafficFlowData(device=device,
                       v_path=args.v_path,
                       w_path=args.w_path,
                       train_pct=args.train_pct,
                       test_pct=args.test_pct,
                       n_time=args.n_time,
                       out_time=args.out_time
                       )
train_iter = DataLoader(data.train, args.batch_size)
val_iter = DataLoader(data.val, args.batch_size)
test_iter = DataLoader(data.test, args.batch_size)

# Model Initialization
model = FNN(args.n_time).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def val():
    model.eval()
    loss_sum, n = 0., 0
    with torch.no_grad():
        for x, y in val_iter:
            if torch.sum(torch.ne(y, Utils.null_val)).item() == 0.:
                continue

            y_pred = model(x).view(len(x), -1)
            loss = masked_mse_loss(y_pred, y, Utils.null_val)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        return loss_sum / n


def train():
    early_stop = EarlyStop(args.patience)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0., 0
        for x, y in tqdm.tqdm(train_iter):
            # TODO: 真实值全是null value导致masked_loss为0，可能导致了梯度出现问题
            #       因此此处直接既不inference，也不back-propagate
            if torch.sum(torch.ne(y, Utils.null_val)).item() == 0.:
                continue

            y_pred = model(x).view(len(x), -1)

            loss = masked_mse_loss(y_pred, y, Utils.null_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()

        val_loss = val()

        if early_stop.check(val_loss):
            break
        if early_stop.save:
            torch.save(model.state_dict(), model_save_path)

        # print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f} | Early Stop: {:02d}'.format(
        #     epoch, optimizer.param_groups[0]['lr'], loss_sum / n, val_loss, early_stop.cnt))

    print("Training Completed!")


def test():
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    loss_sum, n = 0., 0
    with torch.no_grad():
        for x, y in test_iter:
            if torch.sum(torch.ne(y, Utils.null_val)).item() == 0.:
                continue

            y_pred = model(x).view(len(x), -1)
            loss = masked_mse_loss(y_pred, y, Utils.null_val)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        test_loss = loss_sum / n

        ae, ape, se, n = 0., 0., 0., 0
        for x, y in test_iter:
            if torch.sum(torch.ne(y, Utils.null_val)).item() == 0.:
                continue

            batch_size = len(x)
            n += batch_size

            y_true = Utils.inverse_z_score(y.view(batch_size, -1).cpu().numpy())
            y_pred = Utils.inverse_z_score(model(x).view(batch_size, -1).cpu().numpy())

            ae += batch_size * masked_loss_np(y_pred, y_true, "mae")
            ape += batch_size * masked_loss_np(y_pred, y_true, "mape")
            se += batch_size * masked_loss_np(y_pred, y_true, "rmse")

        mae = np.divide(ae, n)
        mape = np.divide(ape, n)
        rmse = np.sqrt(np.divide(se, n))
    print('Test loss {:.6f}'.format(test_loss))
    print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(mae, mape, rmse))


train()
test()

# summary(model, (args.n_time, data.n_node, 1))
