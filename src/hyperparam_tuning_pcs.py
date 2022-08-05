from functools import partial
from re import A
from tkinter.messagebox import NO
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from statsmodels.stats.weightstats import ztest
import warnings
import statsmodels.api as sm
from scipy.stats import pearsonr
from torch.autograd import grad
from functools import reduce
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ERM (basline) trianing and testing definitons
def erm_train(config, train_data, val_data, trait, num_pcs, checkpoint_dir=None):
    model = Net_pheno_auto(config['units'],num_pcs)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(config["batch_size"]), shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])



    for epoch in range(1, 11):
        for batch_idx, (_, data, target, pcs, ancestry) in enumerate(train_loader):
            data, target, pcs, ancestry = data.to(device), target.to(device).float(), pcs.to(device).float(), ancestry.to(device).long()
            data = torch.unsqueeze(data, 1)
            optimizer.zero_grad()
            output, pheno_out = model(data,pcs)
            if trait == 1:
                loss_pheno = F.mse_loss(pheno_out, target) # predicting pheno!
            else:
                loss_pheno = F.binary_cross_entropy_with_logits(pheno_out, target) # predicting pheno!
            loss_prs = F.mse_loss(output, data) # autoencoder
            loss = loss_prs+loss_pheno
            loss.backward()
            optimizer.step()
        
        _, train_l,  train_R2_og, train_R2_pred, train_prs_corr, train_pheno_corr, train_out, train_zt_med, train_ancs, train_pred_phenos, train_targets, train_prs, deciles_pred_train, deciles_og_train, pcs_train = test_model(model, train_loader, trait)
        _, val_l,  val_R2_og, val_R2_pred, val_prs_corr, val_pheno_corr, val_out, val_zt_med, val_ancs, val_pred_phenos, val_targets, train_prs, deciles_pred_val, deciles_og_val, pcs_val = test_model(model, val_loader, trait)
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_l, R2= val_R2_pred)
    print("Finished Training")






# IRM training / Testin definitions
# Modified compute penalty to include n environments (still needs to be checked for equilance with original implementation)!
def compute_irm_penalty(losses, dummy, num_envs):
    grads = []
    for e in range(int(num_envs)):
        grads.append(grad(losses[e::int(num_envs)].mean(), dummy, create_graph=True)[0])
    return reduce(lambda x, y: x*y, grads).sum()

def irm_train(config, train_datasets, val_data, device, num_envs, trait, num_pcs, checkpoint_dir=None):
    model = Net_pheno_auto(config['units'], num_pcs)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    train_loaders = []
    for e in range(num_envs):
        # train_loaders.append(torch.utils.data.DataLoader(train_datasets[e], batch_size=int(config["batch_size"]), shuffle=True))
        train_loaders.append(torch.utils.data.DataLoader(train_datasets[e], batch_size=32, shuffle=True))
    
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)
    
    train_loaders = [iter(x) for x in train_loaders]

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

    for epoch in range(1, 101):
        batch_idx = 0
        penalty_multiplier = epoch ** int(config["penalty"])
        while True:
            optimizer.zero_grad()
            error = 0
            penalty = 0
            for loader in train_loaders:
                _, data, target, pcs, ancestry = next(loader, (None, None, None, None, None))
                if data is None:
                    break
                data, target, pcs, ancestry = data.to(device), target.to(device).float(), pcs.to(device).float(), ancestry.to(device).long()
                data = torch.unsqueeze(data, 1)
                output, pheno_out = model(data,pcs)
                # print(pheno_out)
                # print('#'*40)
                if trait == 1:
                    loss_erm_pheno = F.mse_loss(pheno_out* dummy_w, target, reduction='none') # predicting pheno!
                else:
                    loss_erm_pheno = F.binary_cross_entropy_with_logits(pheno_out* dummy_w, target, reduction='none') # predicting pheno!
                # print(F.sigmoid(loss_erm_pheno))
                loss_erm_prs = F.mse_loss(output* dummy_w, data, reduction='none')
                loss_erm = loss_erm_prs + loss_erm_pheno
                penalty += compute_irm_penalty(loss_erm, dummy_w, num_envs)
            error += loss_erm.mean()
            break

        (error + penalty_multiplier * penalty).backward(retain_graph=True)
        optimizer.step()
        batch_idx += 1
        
        # train_l,  train_R2_og, train_R2_pred, train_prs_corr, train_pheno_corr, train_out, train_zt_med, train_ancs, train_pred_phenos, train_targets, train_prs = test_model(model, train_loader)
        _, val_l, val_R2_og, val_R2_pred, val_prs_corr, val_pheno_corr, val_out, val_zt_med, val_ancs, val_pred_phenos, val_targets, val_prs,deciles_pred_val, deciles_og_val, pcs = test_model(model, val_loader, trait)
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_l, R2= val_R2_pred)
    print("Finished Training")

def decile_r2s(prs, pheno, ancs):
    comb = np.stack((prs, pheno, ancs)).T
    r2s = []
    # Subset by ancestry
    for anc in np.unique(comb[:,2]):
        per_ancs = comb[np.where(comb[:,2]==anc)]
        thersholds = np.percentile(per_ancs[:,0],np.arange(0, 100, 10))
        # Subset by deciles
        for t in thersholds[1:]:
            per_decile = per_ancs[np.where(per_ancs[:,0]<=t)]
            # Compute R2s
            temp_res = sm.OLS(per_decile[:,1],per_decile[:,0]).fit()
            r2s.append(temp_res.rsquared_adj)            

    return np.asarray(r2s).reshape(len(np.unique(comb[:,2])),9)
## Model testing - reports loss, R2, r2, median Z-test from distribution comparison
def test_model(model, test_loader, trait):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # Bandaid solution for Z-test runtime warning about float point issues 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.eval()
        test_loss = 0
        test_loss_prs = 0
        test_loss_pheno = 0
        outputs = []
        outputs_phenos = []
        targets = []
        datas_corr = []
        ancestries = []
        PRS_pops = []
        ztests = []
        pcs_all = []
        data_out_r2 = []
        ids_out = []

        with torch.no_grad():
            for ids, data, target, pcs, ancestry in test_loader:
                data, target, pcs, ancestry = data.to(device), target.to(device).float(), pcs.to(device).float(), ancestry.to(device).long()
                # Model testing
                data = torch.unsqueeze(data, 1)
                output, pheno_out = model(data,pcs)
                if trait == 1:
                    test_loss_pheno += F.mse_loss(pheno_out, target, reduction='sum').item()  # sum up batch loss - predicting pheno!
                else:
                    test_loss_pheno += F.binary_cross_entropy_with_logits(pheno_out, target, reduction='sum').item()  # sum up batch loss - predicting pheno!
                test_loss_prs += F.mse_loss(output, data, reduction='sum').item()  # sum up batch loss - autoencoder
                # Metrics calculation per batch
                data_out = data.detach().numpy()
                target_out = target.detach().numpy()
                ancestry_out = ancestry.detach().numpy()
                outputs.extend(output.detach().numpy())
                targets.extend(target_out)
                ancestries.extend(ancestry.detach().numpy())
                datas_corr.extend(data_out[:,0])
                outputs_phenos.extend(pheno_out.detach().numpy())
                data_out_a = np.concatenate((data_out,ancestry_out),axis=1)
                pcs_all.extend(pcs.detach().numpy())
                data_out_r2.extend(data.detach().numpy())
                ids_out.extend(ids.detach().numpy())

        # Dataloader metric computation
        test_loss_prs /= len(test_loader.dataset)
        test_loss_pheno /= len(test_loader.dataset)
        test_loss = test_loss_prs + test_loss_pheno
        if trait == 1:
            model_prs_org = sm.OLS(targets,np.concatenate((data_out_r2,pcs_all),axis =1))
            results_og = model_prs_org.fit()
            coef_det_og = results_og.rsquared_adj 
            model_prs_pred = sm.OLS(targets,np.concatenate((outputs,pcs_all),axis =1))
            results_prs_pred = model_prs_pred.fit()
            coef_det_prs_pred = results_prs_pred.rsquared_adj
        else:
            # Using logistic regression from sklearn
            clf_og = LogisticRegression(random_state=0).fit(data_out_r2, targets)
            coef_det_og = roc_auc_score(targets, clf_og.predict_proba(data_out_r2)[:, 1])
            clf_pred = LogisticRegression(random_state=0).fit(outputs, targets)
            coef_det_prs_pred = roc_auc_score(targets, clf_pred.predict_proba(outputs)[:, 1])
            # sigmoid = nn.Sigmoid()
            # coef_det_prs_pred = roc_auc_score(targets, sigmoid(torch.tensor(outputs_phenos)))
            
        prs_corr,_ = pearsonr(datas_corr, np.squeeze(outputs)) #prs corr
        pheno_corr,_ = pearsonr(np.squeeze(targets), outputs_phenos) #pheno corr
        deciles_pred = decile_r2s(prs=np.squeeze(outputs),pheno=np.squeeze(targets),ancs=np.squeeze(ancestries))
        deciles_og = decile_r2s(prs=np.squeeze(datas_corr),pheno=np.squeeze(targets),ancs=np.squeeze(ancestries))

        return ids_out, test_loss, coef_det_og, coef_det_prs_pred, prs_corr, pheno_corr, outputs, 10, ancestries, outputs_phenos, targets, datas_corr, deciles_pred, deciles_og, pcs


def tuner(train_data, test_data, all_train_data=None, irm=True, config = None, num_samples=10, max_num_epochs=10, num_envs =None, trait = 1, num_pcs = 8 ,device='cpu', gpus_per_trial=0):
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "units": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "penalty": tune.uniform(0.5, 1.5),
    }
    # config = {
    #     "lr": tune.grid_search([0.01,0.001,0.0001,0.00001]),
    #     "units": tune.grid_search((2 ** np.arange(3, 10)).tolist()),
    #     "penalty": tune.grid_search(np.arange(0.7, 1.3, 0.1).tolist()),
    #     # 'batch_size': tune.grid_search([16,32,64]),
    # }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss",'R2' ,"training_iteration"])
    if irm:
        result = tune.run(
            partial(irm_train, train_datasets=train_data, val_data=test_data, device= device, num_envs=num_envs, trait = trait, num_pcs=num_pcs),
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)
    else:
        result = tune.run(
            partial(erm_train, train_data=train_data, val_data=test_data, trait = trait, num_pcs=num_pcs),
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation R2: {}".format(
        best_trial.last_result['R2']))

    best_trained_model = Net_pheno_auto(best_trial.config["units"], num_pcs=num_pcs)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    if irm:
        train_loader = torch.utils.data.DataLoader(all_train_data, batch_size=len(all_train_data), shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    ids_out_train, loss_train, R2_og_train, R2_pred_train, prs_corr_train, pheno_corr_train, outputs_train, ztest_med_train, ancestries_train, outputs_phenos_train, targets_train, prs_og_train, decile_r2s_pred_train, decile_r2s_og_train, pcs_train = test_model(best_trained_model, train_loader, trait)
    ids_out_val, loss_val, R2_og_val, R2_pred_val, prs_corr_val, pheno_corr_val, outputs_val, ztest_med_val, ancestries_val, outputs_phenos_val, targets_val, prs_og_val, decile_r2s_pred_val, decile_r2s_og_val, pcs_val  = test_model(best_trained_model, val_loader, trait)
    ids_out_test, loss_test, R2_og_test, R2_pred_test, prs_corr_test, pheno_corr_test, outputs_test, ztest_med_test, ancestries_test, outputs_phenos_test, targets_test, prs_og_test, decile_r2s_pred_test, decile_r2s_og_test, pcs_test = test_model(best_trained_model, test_loader, trait)

    print("Best trial test set R2: {}".format(R2_pred_test))

    #return dict with results
    results_ret = {
        'metrics':{
            'R2_og_train': R2_og_train,
            'R2_og_val': R2_og_val,
            'R2_og_test': R2_og_test,
            'R2_pred_train':R2_pred_train,
            'R2_pred_val':R2_pred_val,
            'R2_pred_test':R2_pred_test,
            'r2_prs_train':prs_corr_train,
            'r2_prs_val':prs_corr_val,
            'r2_prs_test':prs_corr_test,
            'r2_pheno_train':pheno_corr_train,
            'r2_pheno_val':pheno_corr_val,
            'r2_pheno_test':pheno_corr_test,
            'loss_train':loss_train,
            'loss_val':loss_val,
            'loss_test':loss_test
        },
        'data':{
            'ids_out_train':ids_out_train,
            'ids_out_val':ids_out_val,
            'ids_out_test':ids_out_test,
            'PRS_og_train':prs_og_train,
            'PRS_og_val':prs_og_val,
            'PRS_og_test':prs_og_test,
            'PRS_pred_train':outputs_train,
            'PRS_pred_val':outputs_val,
            'PRS_pred_test':outputs_test,
            'Pheno_og_train':targets_train,
            'Pheno_og_val':targets_val,
            'Pheno_og_test':targets_test,
            'Pheno_pred_train':outputs_phenos_train,
            'Pheno_pred_val':outputs_phenos_val,
            'Pheno_pred_test':outputs_phenos_test,
            'ancs_train':ancestries_train,
            'ancs_val':ancestries_val,
            'ancs_test':ancestries_test,
            'pcs_train':pcs_train,
            'pcs_val':pcs_val,
            'pcs_test':pcs_test,
            'deciles_r2s_pred_train':decile_r2s_pred_train,
            'deciles_r2s_pred_val':decile_r2s_pred_val,
            'deciles_r2s_pred_test':decile_r2s_pred_test,
            'deciles_r2s_og_train':decile_r2s_og_train,
            'deciles_r2s_og_val':decile_r2s_og_val,
            'deciles_r2s_og_test':decile_r2s_og_test,
        },
        'hyperparams':{
            'lr':best_trial.config["lr"],
            'pen_mult':best_trial.config["penalty"],
            'units':best_trial.config["units"]
        }
    }
    return results_ret

# Encoder-decoder-mlp structures  
class encoder(nn.Module):
    def __init__(self, units, num_pcs):
        super(encoder, self).__init__()
        # self.num_classes = 11
        # self.embed_ancs = nn.Embedding(self.num_classes, embedding_dim=1)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(1+num_pcs,units))
        # if len(units)>1:
            # for i in range(len(units)-1):
            #     self.linears.append(nn.Linear(units, units))

    def forward(self, x, a):
        # emb_ancs = torch.squeeze(self.embed_ancs(a),dim=-1)
        # x = torch.cat((x,emb_ancs),1)
        x = torch.cat((x,a),1)
        for i in range(len(self.linears)):
            x = F.relu(self.linears[i](x))
        return x 
    
class decoder(nn.Module):
    def __init__(self, units):
        super(decoder, self).__init__()
        self.linears = nn.ModuleList()
        # if len(units)>1:
        #     for i in range(len(units)-1):
        #         self.linears.append(nn.Linear(units, units))
        self.linears.append(nn.Linear(units,1))
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.dropout(x)
        logits = self.linears[-1](x)
        return logits 
    
class mlp(nn.Module):
    def __init__(self, units):
        super(mlp, self).__init__()
        self.linears = nn.ModuleList()
        # if len(units)>1:
        #     for i in range(len(units)-1):
        #         self.linears.append(nn.Linear(units[i], units[i+1]))
        self.linears.append(nn.Linear(units,1))
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.dropout(x)
        logits = self.linears[-1](x)
        return logits 
    
class Net_pheno_auto(nn.Module):
    def __init__(self, units, num_pcs):
        super(Net_pheno_auto, self).__init__()
        self.encoder = encoder(units, num_pcs)
        self.decoder = decoder(units)
        self.mlp = mlp(units)

    def forward(self, x,a):
        x = self.encoder(x,a)
        prs = self.decoder(x)
        pheno = self.mlp(x)
        return prs, pheno