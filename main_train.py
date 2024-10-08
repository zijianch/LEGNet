'''
Step 2: Training and Validation on Dataset B Using k-fold CV
Objective: To train the model and validate its performance reliably.
Process:
    Split dataset B into k folds.
    train the model (configured with the best hyperparameters from Step 1) on k-1 folds and validate on the remaining fold.
    Repeat this process for each fold, creating k different models, each trained on a slightly different subset of data.
After evaluating all k models on their respective validation folds, report the mean and standard deviation (or confidence intervals) 
of the performance metrics across these k folds. This statistical summary provides an understanding of the model's average performance 
and its variability due to different subsets of the training data.
'''

# important note: this script is designed to be run reshuffling the data in each fold. So, after uploading the new assignment file


from dataLoader import *
from mynet.net import *

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import copy
import argparse
import datetime
import os
import io
import sys
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Preparation ######

# parser for quick hyperparameter adjustment outside of the script
# Default values is set based on the pretraining process
# usage: python main_train.py
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=8, help='number of node clusters')
parser.add_argument('--dim0', type=int, default=4, help='number of features per node after E2EBlock')
parser.add_argument('--dim1', type=int, default=8, help='number of features per node after E2N')
parser.add_argument('--dim2', type=int, default=2, help='number of features per node after brainGNNconv')
parser.add_argument('--dim3', type=int, default=8, help='number of upsampled features per graph')
parser.add_argument('--nepoch', type=int, default=150, help='max number of epochs to train')
parser.add_argument('--scheduler_step', type=int, default=20, help='scheduler step size')
parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='scheduler gamma')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--data', type=str, default=1, help='ways of data split')
args = parser.parse_args()

# Prepare the experiment log directory and file name based on current time and input args
experiment_name = f"{args.data}_training_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"
log_dir = os.path.join('./training/log', experiment_name)
model_dir = os.path.join('./training/model', experiment_name)
results_dir = os.path.join('./training/results', experiment_name)
experiment_log_file = os.path.join('./training/experiment_logs', f"{experiment_name}.txt")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.dirname(experiment_log_file), exist_ok=True)

with open(experiment_log_file, 'a') as log_file:
    log_file.write(f"Experiment Name: {experiment_name}\n")
    log_file.write(f"Experiment Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Hyperparameters: k={args.k}, dim0(E2E)={args.dim0}, dim1(E2N)={args.dim1}, dim2(BGconv)={args.dim2}, dim3(FC)={args.dim3}\n")
    log_file.write(f"Scheduler Step Size: {args.scheduler_step}, Scheduler Gamma: {args.scheduler_gamma}. This is now effective\n")
    log_file.write(f"Early Stopping Patience: {args.patience}\n")


# Loss functions
criterion = torch.nn.MSELoss()

# pre-initialize the model just for logging purposes
model = Network(k=args.k, dim0=args.dim0, dim1=args.dim1, dim2=args.dim2, dim3=args.dim3).to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = count_parameters(model)

model_str = io.StringIO()
sys.stdout = model_str
print(model)
sys.stdout = sys.__stdout__  # Reset the standard output to its original value

with open(experiment_log_file, 'a') as log_file:
    log_file.write("\nModel Architecture:\n")
    log_file.write(model_str.getvalue())
    log_file.write("\n")  
    log_file.write(f"Total learnable parameters: {total_params}\n")
    
###### Trainer ######

def train(epoch):

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    
    model.train()
    losses = []

    for batch_idx, (X, Y, pos) in enumerate(train_loader):

        X, Y, pos = X.to(device), Y.to(device), pos.to(device)
        optimizer.zero_grad()

        Xtemp = X[0][0].detach().numpy()
        edge_index, edge_attr = createEdgeIdxAttr(Xtemp)
        batch = torch.zeros(246).to(torch.int64)
        out = model(X, edge_index, batch, edge_attr,pos)


        loss = torch.sqrt(criterion(out, Y))
        loss.backward()
        optimizer.step()

        scheduler.step()
        

        losses.append(loss.item())
        writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader) + batch_idx)

        
    return np.mean(losses)

def validation(epoch):

    model.eval()
    losses = []

    with torch.no_grad():
        for batch_idx, (X, Y, pos) in enumerate(val_loader):

            X, Y, pos = X.to(device), Y.to(device), pos.to(device)

            Xtemp = X[0][0].detach().numpy()
            edge_index, edge_attr = createEdgeIdxAttr(Xtemp)
            batch = torch.zeros(246).to(torch.int64)
            out = model(X, edge_index, batch, edge_attr, pos)

            loss = criterion(out, Y)
            losses.append(loss.item())
            writer.add_scalar('Loss/validation', loss.item(), epoch*len(val_loader) + batch_idx)

            if batch_idx == 0:
                Y_all = Y
                out_all = out
            else:
                Y_all = torch.cat((Y_all, Y), 0)
                out_all = torch.cat((out_all, out), 0)

    return np.mean(losses), Y_all, out_all


###### Training and Validation ######

nepoch = args.nepoch
data_split = args.data

r2_all = np.zeros(10)
pearson_all = np.zeros(10)

r2_all_best = np.zeros(10)
pearson_all_best = np.zeros(10)

val_loss_10folds = np.zeros(10)
val_loss_10folds_best = np.zeros(10)


with open(experiment_log_file, 'a') as log_file:
    log_file.write("\n----------Training Start----------\n")

for fold_idx in range(1, 11):

    print(f"Fold {fold_idx}:")
    with open(experiment_log_file, 'a') as log_file:
        log_file.write(f"Fold {fold_idx}:\n")

    # Load the dataset for the current fold
    train_dataset = DS1dataset(section='train', fold=10, val_index=fold_idx, data_split=data_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = DS1dataset(section='validation', fold=10, val_index=fold_idx, data_split=data_split)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize the model for the current fold
    model = Network(k=args.k, dim0=args.dim0, dim1=args.dim1, dim2=args.dim2, dim3=args.dim3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    # Early stopping
    best_metric = float('inf')
    min_delta = 0.005
    patience = args.patience
    patience_counter = 0

    # Train and validate
    train_loss_all = np.zeros(nepoch)
    val_loss_all = np.zeros(nepoch)

    for epoch in range(nepoch):

        train_loss = train(epoch)
        val_loss, Y_all, out_all = validation(epoch)

        val_loss = np.sqrt(val_loss)

        train_loss_all[epoch] = train_loss
        val_loss_all[epoch] = val_loss

        print(f"    Epoch {epoch}: Train loss: {train_loss}, Validation loss: {val_loss}")

        # Log to file every 5 epochs
        if epoch % 5 == 0:
            with open(experiment_log_file, 'a') as log_file:
                log_file.write(f"    Epoch {epoch}: Train loss: {train_loss}, Validation loss: {val_loss}\n")
        
        # Early stopping
        if best_metric - val_loss > min_delta: # i.e., current val loss reaches a new low (recorded in best_metric)
            best_metric = val_loss

            #------------------- Save the best model -------------------#
            best_model = copy.deepcopy(model)
            best_model_save_path = os.path.join(model_dir, f"best_model_fold_{fold_idx}.pt")
            torch.save(best_model.state_dict(), best_model_save_path)
            best_model_epoch = epoch
            Y_all_best = Y_all
            out_all_best = out_all
            #-----------------------------------------------------------#

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 35: # if the model hasn't improved (or even worsen) for 5 epochs, stop immediately
                print("Early stopping")
                with open(experiment_log_file, 'a') as log_file:
                    log_file.write(f"Early stopping at epoch {epoch}, current validation loss: {val_loss}, current best validation loss: {best_metric}\n")
                    val_loss_10folds_best[fold_idx-1] = best_metric
                    val_loss_10folds[fold_idx-1] = val_loss
                break
        
    # Save the **last observed model**
    model_save_path = os.path.join(model_dir, f"last_observed_model_fold_{fold_idx}.pt")
    torch.save(model.state_dict(), model_save_path)

    # Save losses
    np.save(os.path.join(results_dir, f"train_loss_fold_{fold_idx}.npy"), train_loss_all)
    np.save(os.path.join(results_dir, f"val_loss_fold_{fold_idx}.npy"), val_loss_all)

    #------------------- Evaluate the model on the validation set -------------------#
    # 1. based on the last observed model
    r2 = r2_score(Y_all.cpu().numpy(), out_all.cpu().numpy())
    pearson, _ = pearsonr(Y_all.cpu().numpy().flatten(), out_all.cpu().numpy().flatten())
    # saving the Y and out (together in one file) for future reference (e.g., plotting)
    np.save(os.path.join(results_dir, f"GT_Pred_fold_{fold_idx}.npy"), np.array([Y_all.cpu().numpy(), out_all.cpu().numpy()]))

    r2_all[fold_idx-1] = r2
    pearson_all[fold_idx-1] = pearson

    # 2. based on the best model
    r2_best = r2_score(Y_all_best.cpu().numpy(), out_all_best.cpu().numpy())
    pearson_best, _ = pearsonr(Y_all_best.cpu().numpy().flatten(), out_all_best.cpu().numpy().flatten())
    # saving the Y and out (together in one file) for future reference (e.g., plotting)
    np.save(os.path.join(results_dir, f"GT_Pred_best_fold_{fold_idx}.npy"), np.array([Y_all_best.cpu().numpy(), out_all_best.cpu().numpy()]))

    r2_all_best[fold_idx-1] = r2_best
    pearson_all_best[fold_idx-1] = pearson_best
    #---------------------------------------------------------------------------------#


    print(f"Fold {fold_idx}: R^2: {r2}, Pearson correlation: {pearson}")
    with open(experiment_log_file, 'a') as log_file:
        log_file.write(f"Fold {fold_idx} - R2: {r2}, Pearson: {pearson}\n")
        log_file.write(f"Fold {fold_idx} - Best Model (at epoch {best_model_epoch}) - R2: {r2_best}, Pearson: {pearson_best}\n")


with open(experiment_log_file, 'a') as log_file:
    log_file.write("\n----------Training Done----------\n")

