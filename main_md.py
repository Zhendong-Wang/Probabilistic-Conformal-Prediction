import os
import sys
import torch
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

from pcp.utils import seed_everything, \
                    evaluate_predictions_baselines_md, evaluate_predictions_baselines_md_cd, \
                    evaluate_predictions_pcp_md
from pcp.pcp import PCP
from pcp.models.gan import GAN
from cde.density_estimator import KernelMixtureNetwork, MixtureDensityNetwork, NormalizingFlowEstimator
from chr.black_boxes import QNet, QRF
from other_baselines.cd_split import CDSplit
from pcp.vis import visual_baseline, visual_pcp, visual_baseline_cd, visual_data
from dataset import GetDataset, Data_Sampler_MD
from sklearn.preprocessing import StandardScaler
import signal


'''
import json
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

from copulatools.preprocessing_utilities import (
    learn_scalers,
    apply_scalers,
    get_continuous_arrays,
    get_targets_arrays
)
from copulatools.models import (
    two_model_mlp,
    fit_per_target,
    predict_per_target
)
from copulatools.conformal_utilities import (
    aggregate_metrics_across_folds,
    aggregate_across_folds
)
from copulatools.nonconformity_predictor import (
    multi_target_nn_nonconformity,
    single_target_rf_nonconformity
)
'''

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

def run(seed, res_table, gj_table, args, caltype = 'uniform', gamma = 1):

    # Set random seed
    seed_everything(seed)
    random_state = seed
    print(f"running seed {seed} ......")

    X, Y = GetDataset(dataset_name, base_dataset_path, seed = seed)
    print(X)
    #scaler = StandardScaler()
    #Y = scaler.fit_transform(Y)
    #Y += 1e-6*np.random.normal(size=Y.shape)     # Add noise to response

    if X.shape[0] <= 2*n_cal + test_size:
        raise ValueError("X doesn't have the correct shape")

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal, random_state=random_state)

    visual_data(X_test, Y_test, num = 200)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test_origin = X_test
    X_test = scaler.transform(X_test)

    '''
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_calib = scaler.transform(Y_calib)
    Y_test = scaler.transform(Y_test)
    '''

    n_train = X_train.shape[0]
    assert(n_cal == X_calib.shape[0])
    n_test = X_test.shape[0]

    """
    Train Probabilistic Model
    """
    x_dim, y_dim = X_train.shape[1], Y_train.shape[1]
    #x_dim, y_dim = X_train.shape[1], 1
    train_dataset = Data_Sampler_MD(X_train, Y_train, device=device, seed = seed)
    if args.md_type == 'gan':
        model = GAN(x_dim, y_dim, args.z_dim, args.H, args.batch_size, device,
                    lr=args.lr,
                    adam_b1=args.b1,
                    adam_b2=args.b2)
        model.train(train_dataset, args.batch_size, args.n_epochs)
        model.eval()
        pcp = PCP(model=model, base="torch", device=args.device, alpha=args.alpha, sample_K=args.K)
    elif args.md_type == 'cgan':
        model = CGAN(x_dim, y_dim, args.z_dim, args.H, args.batch_size, device,
                     lr=args.lr,
                     adam_b1=args.b1,
                     adam_b2=args.b2)
    elif args.md_type == 'gauss':
        model = Gauss_Model(x_dim, y_dim, args.z_dim, args.H, args.batch_size, device,
                            lr=args.lr)
    elif args.md_type == 'cvae':
        model = CVAE_Model(x_dim, y_dim, args.z_dim, args.H, args.batch_size, device,
                           lr=args.lr)
    elif args.md_type == 'mixd':
        model = MixtureDensityNetwork(name="MIXD" +str(seed)+caltype,hidden_sizes=(16, 16), ndim_x=x_dim, ndim_y=y_dim, n_training_epochs= args.n_epochs, random_seed = seed)
        model.fit(X=X_train, Y=Y_train)
        pcp = PCP(model=model, base="numpy", device=device, alpha=args.alpha, sample_K=args.K, fr = args.fr)
    elif args.md_type == 'kmn':
        model = KernelMixtureNetwork(name="kmn" +str(seed)+caltype, ndim_x=x_dim, ndim_y=y_dim, n_training_epochs= args.n_epochs, random_seed = seed)
        model.fit(X=X_train, Y=Y_train)
        pcp = PCP(model=model, base="numpy", device=device, alpha=args.alpha, sample_K=args.K, fr = args.fr)
    elif args.md_type == 'nfe':
        model = NormalizingFlowEstimator(name="NFE", ndim_x=x_dim, ndim_y=y_dim, hidden_sizes=(100, 100))
    elif args.md_type == 'qrf':
        n_estimators = 100
        min_samples_leaf = 50
        grid_quantiles = np.arange(0.01, 1.0, 0.01)
        model = QRF(grid_quantiles, n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=n_jobs, verbose=verbose)
        model.fit(X=X_train, Y=Y_train)
        pcp = PCP(model=model, base="numpy", device=args.device, alpha=args.alpha, sample_K=args.K)
    #train_dataset = Data_Sampler(X_train, Y_train, device=device)
    # calib_dataset = Data_Sampler(X_calib, Y_calib, device=args.device)
    # test_dataset  = Data_Sampler(X_test, Y_test, device=args.device)

    if caltype == 'cov':
        cov = np.linalg.inv(np.cov(Y_train.T))
        pcp.calibrate_md(X_calib, Y_calib, caltype = caltype, gamma = gamma, cov = cov)
    elif caltype == 'covcond':
        cov = np.array([np.linalg.pinv(i) for i in model.covariance(X_calib)])
        pcp.calibrate_md(X_calib, Y_calib, caltype = caltype, gamma = gamma, cov = cov)
    elif caltype == 'filtered':
        cov = None
        '''
        fr_grid = [0, 0.01, 0.05, 0.1, 0.2]
        areafr = []
        for fr in fr_grid:
            pcp.calibrate_md(X_calib, Y_calib, caltype = caltype, gamma = gamma, cov = cov, fr = fr)
            pred, density = pcp.predict_md(X_calib, caltype = caltype, gamma = gamma, fr = fr)
            res = evaluate_predictions_pcp_md(pred, pcp.qt, Y_calib, X=X_calib, caltype = caltype, density = density, cov = cov)
            area_thisfr = res['Area'].values[0]
            areafr.append(area_thisfr)
        print(areafr)
        fr = fr_grid[np.argmin(areafr)]
        print(f'selected fr is {fr}')
        '''
        fr = 0.05
        pcp.calibrate_md(X_calib, Y_calib, caltype = caltype, gamma = gamma, cov = cov, fr = fr)
    else:
        cov = None
        fr = 0
        pcp.calibrate_md(X_calib, Y_calib, caltype = caltype, gamma = gamma, cov = cov)
    print('radius:')
    print(pcp.qt)
    # test
    X_test = X_test
    # Compute prediction on test data
    pred, density = pcp.predict_md(X_test, caltype = caltype, gamma = gamma, fr = fr)
    # Evaluate results
    if caltype == 'covcond':
        cov = np.array([np.linalg.inv(i) for i in model.covariance(X_test)])
    res = evaluate_predictions_pcp_md(pred, pcp.qt, Y_test, X=X_test, caltype = caltype, density = density, cov = cov)
    '''
    if dataset_name == 'taxi':
        gj, neighbor_list = taxi_subgroup_pcp(pred, pcp.qt, Y_test, X_test=X_test_origin, caltype = caltype, density = density)
        gj['seed'] = seed
        visual_subgroup(gj, neighbor_list, pred, pcp.qt, Y_test, num = 10, seed = 0, caltype = caltype, density = density)
    else:
        gj = None
    if dataset_name == 'taxi':
        gj['method'] = 'pcp'
        gj_table = gj_table.append(gj)
    '''
    # Add information about this experiment
    print(res)
    res['Dataset'] = dataset_name
    res['Method'] = "PCP_" + caltype
    res['seed'] = seed
    res['Nominal'] = 1-alpha
    res['n_train'] = n_train
    res['n_cal'] = n_cal
    res['n_test'] = n_test

    # Add results to the list
    res_table = res_table.append(res)
    visual_pcp(pred, pcp.qt, Y_test,  X_test = X_test_origin, num=200,seed = seed, caltype = caltype, density = np.array(density))
    return res_table, gj_table



def run_baselines(seed, res_table, gj_table, args):
    """
    a baseline where regress each y separately.
    """
    # Set random seed
    seed_everything(seed)
    random_state = seed
    print(f"running seed {seed} ......")

    X, Y = GetDataset(dataset_name, base_dataset_path, seed = seed)
    #print(X)
    #Y += 1e-6*np.random.normal(size=Y.shape)     # Add noise to response
    #scaler = StandardScaler()
    #Y = scaler.fit_transform(Y)

    if X.shape[0] <= 2*n_cal + test_size:
        raise ValueError("X doesn't have the correct shape")

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test_origin = X_test
    X_test = scaler.transform(X_test)

    '''
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_calib = scaler.transform(Y_calib)
    Y_test = scaler.transform(Y_test)
    '''

    n_train = X_train.shape[0]
    assert(n_cal == X_calib.shape[0])
    n_test = X_test.shape[0]

    """
    Train Probabilistic Model
    """

    print(Y_train)
    print(X_train)
    n_features = X_train.shape[1]
    epochs = args.n_epochs
    lr = 0.0005
    batch_size = args.batch_size
    dropout = 0.1
    grid_quantiles = np.arange(0.01,1.0,0.01)

    '''
    bbox_nn_y0 = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                   dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                   verbose=verbose, random_state=seed)
    bbox_nn_y1 = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                   dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                   verbose=verbose, random_state=seed)
    print("Training black box model NNet...")
    bbox_nn_y0.fit(X_train, Y_train[:,0])
    bbox_nn_y1.fit(X_train, Y_train[:,1])

    print(bbox_nn_y0.predict(X_train))
    print(bbox_nn_y1.predict(X_train))
    '''
    x_dim = X_train.shape[1]
    #'''
    if args.md_type == 'mixd':
        model_y1 = MixtureDensityNetwork(name="MIXDy1" +str(seed)+caltype, ndim_x=x_dim, ndim_y=1, n_training_epochs= args.n_epochs, random_seed = seed)
        model_y1.fit(X=X_train, Y=Y_train[:,0])
        model_y2 = MixtureDensityNetwork(name="MIXDy2" +str(seed)+caltype, ndim_x=x_dim, ndim_y=1, n_training_epochs= args.n_epochs, random_seed = seed)
        model_y2.fit(X=X_train, Y=Y_train[:,1])
    elif args.md_type == 'kmn':
        model_y1 = MixtureDensityNetwork(name="KMNy1" +str(seed)+caltype, ndim_x=x_dim, ndim_y=1, n_training_epochs= args.n_epochs, random_seed = seed)
        model_y1.fit(X=X_train, Y=Y_train[:,0])
        model_y2 = MixtureDensityNetwork(name="KMNy2" +str(seed)+caltype, ndim_x=x_dim, ndim_y=1, n_training_epochs= args.n_epochs, random_seed = seed)
        model_y2.fit(X=X_train, Y=Y_train[:,1])
    #'''
    methods = {
        #'CHR-NNet'    : [CHR(bbox_nn_y0, ymin=np.min(Y_train[:,0]), ymax=np.max(Y_train[:,0]), y_steps=1000, randomize=True), \
        #                    CHR(bbox_nn_y1, ymin=np.min(Y_train[:,1]), ymax=np.max(Y_train[:,1]), y_steps=1000, randomize=True)],
        'CDSplit'     : [CDSplit(X_train, Y_train[:,0], ymin=np.min(Y_train[:,0]), ymax=np.max(Y_train[:,0]), name = 'y0', seed = seed, model = model_y1), \
                            CDSplit(X_train, Y_train[:,1], ymin=np.min(Y_train[:,1]), ymax=np.max(Y_train[:,1]), name = 'y1', seed = seed, model = model_y2)],
        #'DistSplit'   : [DistSplit(bbox_nn_y0, ymin=np.min(Y_train[:,0]), ymax=np.max(Y_train[:,0])), \
        #                    DistSplit(bbox_nn_y1, ymin=np.min(Y_train[:,1]), ymax=np.max(Y_train[:,1]))],
        #'DCP'         : [DCP(bbox_nn_y0, ymin=np.min(Y_train[:,0]), ymax=np.max(Y_train[:,0])), \
        #                    DCP(bbox_nn_y1, ymin=np.min(Y_train[:,1]), ymax=np.max(Y_train[:,1]))],
        #'CQR'         : [CQR(bbox_nn_y0), CQR(bbox_nn_y1)],
        #'CQR2'        : [CQR2(bbox_nn_y0), CQR2(bbox_nn_y1)],
    }


    for method_name in methods:
        print(method_name)
        # Apply the conformalization method
        method_y0 = methods[method_name][0]
        method_y1 = methods[method_name][1]
        method_y0.calibrate(X_calib, Y_calib[:,0], alpha/2)
        method_y1.calibrate(X_calib, Y_calib[:,1], alpha/2)
        if method_name == 'CDSplit':
            pred_y0 = method_y0.predict(X_test, clear_session = False)
            pred_y1 = method_y1.predict(X_test)
        else:
            pred_y0 = method_y0.predict(X_test)
            pred_y1 = method_y1.predict(X_test)
        if method_name == 'CDSplit':
            res = evaluate_predictions_baselines_md_cd(pred_y0, pred_y1, Y_test, X=X_test)
            if dataset_name == 'taxi':
                visual_baseline_cd(pred_y0, pred_y1, Y_test, X_test = X_test_origin, num = 200, seed = seed)
        else:
            # first evaluate the coverage in each dimension
            res = evaluate_predictions_baselines_md(pred_y0, pred_y1, Y_test, X=X_test)
            if dataset_name == 'taxi':
                visual_baseline(pred_y0, pred_y1, Y_test,  X_test = X_test_origin, num = 200,seed = seed)
        # Add information about this experiment
        print(res)
        res['Dataset'] = dataset_name
        res['Method'] = method_name
        res['seed'] = seed
        res['Nominal'] = 1-alpha
        res['n_train'] = n_train
        res['n_cal'] = n_cal
        res['n_test'] = n_test
        '''
        if dataset_name == 'taxi':
            if method_name == 'CDSplit':
                gj, neighbor_list = taxi_subgroup_baselines_cd(pred_y0, pred_y1, Y_test, X_test=X_test_origin)
                visual_subgroup_cd(gj, neighbor_list, pred_y0, pred_y1,Y_test)
            else:
                gj, neighbor_list = taxi_subgroup_baselines(pred_y0, pred_y1, Y_test, X_test=X_test_origin)
                visual_subgroup_baselines(gj, neighbor_list, pred_y0, pred_y1, Y_test)
            # Add results to the list
            gj['method']  = method_name
            gj_table = gj_table.append(gj)
        '''
        res_table = res_table.append(res)
    return res_table, gj_table



def run_copula(seed, res_table, gj_table, args):
    """
    a baseline where regress each y separately.
    """
    # Set random seed
    seed_everything(seed)
    random_state = seed
    print(f"running seed {seed} ......")

    X, Y = GetDataset(dataset_name, base_dataset_path, seed = seed)
    #print(X)
    #Y += 1e-6*np.random.normal(size=Y.shape)     # Add noise to response
    #scaler = StandardScaler()
    #Y = scaler.fit_transform(Y)

    if X.shape[0] <= 2*n_cal + test_size:
        raise ValueError("X doesn't have the correct shape")

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal, random_state=random_state)

    n_val = int(X_train.shape[0]*0.1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=n_val, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test_origin = X_test
    X_test = scaler.transform(X_test)

    '''
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_calib = scaler.transform(Y_calib)
    Y_test = scaler.transform(Y_test)
    '''

    n_train = X_train.shape[0]
    assert(n_cal == X_calib.shape[0])
    n_test = X_test.shape[0]

    """
    Train Probabilistic Model
    """

    n_features = X_train.shape[1]
    epochs = 4000
    lr = 0.0005
    batch_size = args.batch_size
    dropout = 0.1
    grid_quantiles = np.arange(0.01,1.0,0.01)

    #'''
    bbox_nn_y0 = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                   dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1,
                   verbose=verbose, random_state=seed)
    bbox_nn_y1 = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                   dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1,
                   verbose=verbose, random_state=seed)
    print("Training black box model NNet...")
    bbox_nn_y0.fit(X_train, Y_train[:,0])
    bbox_nn_y1.fit(X_train, Y_train[:,1])
    #'''
    x_dim = X_train.shape[1]

    checkpoint = ModelCheckpoint(
        "mlp.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=10, verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]


    model, model_repr = two_model_mlp(
        n_continuous=X_train.shape[1],
        n_outputs=Y_train.shape[1],
        learning_rate=0.0001,
        layer_size=1024,
        embed_size=64,
        dropout_rate=0.3,
        n_layers=5,
    )
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs= 1000,
        verbose=2,
        callbacks=callbacks_list,
        batch_size=256,
    )

    # predict
    Y_test_pred = model.predict(X_test)
    Y_cal_pred = model.predict(X_calib)

    (
        independent_nn_conf_levels[i],
        independent_nn_conf_interval[i],
        gumbel_nn_conf_levels[i],
        gumbel_nn_conf_interval[i],
        empirical_nn_conf_levels[i],
        empirical_nn_conf_interval[i]
    ) = multi_target_nn_nonconformity(
        model,
        X_train_cont,
        X_val_cont,
        Y_train,
        Y_val,
        X_cal_cont,
        X_test_cont,
        Y_cal,
        Y_test,
        Y_cal_pred,
        Y_test_pred,
        epsilon_values,
        n_cont,
        n_out,
        nb_epoch,
        layer_size,
        dropout_rate,
        batch_size,
    )

    for method_name in methods:
        print(method_name)
        # Apply the conformalization method
        method_y0 = methods[method_name][0]
        method_y1 = methods[method_name][1]
        method_y0.calibrate(X_calib, Y_calib[:,0], alpha/2)
        method_y1.calibrate(X_calib, Y_calib[:,1], alpha/2)
        if method_name == 'CDSplit':
            pred_y0 = method_y0.predict(X_test, clear_session = False)
            pred_y1 = method_y1.predict(X_test)
            print(pred_y0)
            print(pred_y1)            
        else:
            pred_y0 = method_y0.predict(X_test)
            pred_y1 = method_y1.predict(X_test)
            print(pred_y0)
            print(pred_y1)
            print(Y_test)
        if method_name == 'CDSplit':
            res = evaluate_predictions_baselines_md_cd(pred_y0, pred_y1, Y_test, X=X_test)
            if dataset_name == 'taxi':
                visual_baseline_cd(pred_y0, pred_y1, Y_test, num = 20, seed = seed)
        else:
            # first evaluate the coverage in each dimension
            res = evaluate_predictions_baselines_md(pred_y0, pred_y1, Y_test, X=X_test)
            if dataset_name == 'taxi':
                visual_baseline(pred_y0, pred_y1, Y_test, num = 20,seed = seed)
        # Add information about this experiment
        print(res)
        res['Dataset'] = dataset_name
        res['Method'] = method_name
        res['seed'] = seed
        res['Nominal'] = 1-alpha
        res['n_train'] = n_train
        res['n_cal'] = n_cal
        res['n_test'] = n_test
        '''
        if dataset_name == 'taxi':
            if method_name == 'CDSplit':
                gj, neighbor_list = taxi_subgroup_baselines_cd(pred_y0, pred_y1, Y_test, X_test=X_test_origin)
                visual_subgroup_cd(gj, neighbor_list, pred_y0, pred_y1,Y_test)
            else:
                gj, neighbor_list = taxi_subgroup_baselines(pred_y0, pred_y1, Y_test, X_test=X_test_origin)
                visual_subgroup_baselines(gj, neighbor_list, pred_y0, pred_y1, Y_test)
            # Add results to the list
            gj['method']  = method_name
            gj_table = gj_table.append(gj)
        '''
        res_table = res_table.append(res)
    return res_table, gj_table

if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=1, type=int)  # Experiment ID
    parser.add_argument('--device', type=str, default="cpu", help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
    parser.add_argument("--dataset", default="taxi", type=str)  # dataset name
    parser.add_argument('--n_runs', type=int, default=5, help='num of runs')
    # Training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--md_type", type=str, default='kmn', help='[cgan, gan, cvae, gauss, mixd]')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--z_dim', type=int, default=15, metavar='N',
                        help='dimensionality of z (default: 50)')
    parser.add_argument('--H', type=int, default=100, metavar='N',
                        help='dimensionality of feature x_feat (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for Adam')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    # PCP parameters
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ncal", type=int, default=2000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--method", type=str, default='baseline')
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--caltype", type=str, default='uniform')
    parser.add_argument("--fr", type=float, default = 0.2)

    args = parser.parse_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Default arguments
    alpha = args.alpha
    test_size = args.ntest
    n_cal = args.ncal
    base_dataset_path = './data/'
    n_jobs = 1
    verbose = False
    out_dir = './results_real'
    method = args.method
    gamma = args.gamma
    caltype = args.caltype

    dataset_name = args.dataset
    experiment = args.exp_id

    print(f"data: {dataset_name}")

    # Determine output file

    res_table = pd.DataFrame()
    gj_table = pd.DataFrame()
    res_table_baseline = pd.DataFrame()
    res_table_copula = pd.DataFrame()

    final_gj = pd.DataFrame()
    final_gj_baseline = pd.DataFrame()
    final_result = pd.DataFrame()
    final_result_baseline = pd.DataFrame()
    final_result_copula = pd.DataFrame()
    for i in np.arange(0, args.n_runs):
        if method == 'pcp':
            #res_table, gj_table = run(i, res_table, gj_table, args, caltype = 'uniform')
            #final_result = final_result.append(res_table)
            #final_gj = final_gj.append(gj_table)
            res_table, gj = run(i, res_table, gj_table, args, caltype = caltype, gamma = gamma)
            final_gj = final_gj.append(gj_table)
            final_result = res_table
            final_gj = final_gj.append(gj)
            out_file = out_dir + f"/{args.dataset}" + "_pcpmd_" + "n_runs_" + str(args.n_runs) + f"_{args.md_type}_{caltype}.txt"
            outgj_file = out_dir + f"/{args.dataset}" + "_pcpmd_subgroup_" + "n_runs_" + str(args.n_runs) + f"_{args.md_type}_{caltype}.txt"
            out_file_detailed = out_dir + f"/{args.dataset}" + "_pcpmddt_" + "n_runs_" + str(args.n_runs) + f"_{args.md_type}_{caltype}.txt"
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            final_result.to_csv(out_file_detailed)
            agg_result_mean = final_result.groupby(['Method']).mean()
            agg_result_std = final_result.groupby(['Method']).std() / np.sqrt(i)
            agg_result = agg_result_mean.append(agg_result_std)
            agg_result.to_csv(out_file, index=True, float_format="%.4f")
            final_gj.to_csv(outgj_file)
            print("Updated summary of results on\n {}".format(out_file))
            print(agg_result)
        elif method == 'baseline':
            res_table_baseline, gj_table = run_baselines(i, res_table_baseline, gj_table, args)
            final_gj = final_gj.append(gj_table)
            final_result_baseline = res_table_baseline
            out_file = out_dir + f"/{args.dataset}" + "_baseline_" + "n_runs_" + str(args.n_runs) + ".txt"
            outgj_file = out_dir + f"/{args.dataset}" + "_baseline_subgroup_" + "n_runs_" + str(args.n_runs) + f"_{args.md_type}.txt"
            out_file_detailed = out_dir + f"/{args.dataset}" + "_baselinedt_" + "n_runs_" + str(args.n_runs) + ".txt"
            # Write results on output files
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            final_result_baseline.to_csv(out_file_detailed)
            final_result_baseline.Area = final_result_baseline.Area.astype(float)
            final_result_baseline['Area cover'] = final_result_baseline['Area cover'].astype(float)
            agg_result_mean = final_result_baseline.groupby(['Method'])[['Coverage', 'Conditional coverage',\
                                'Area', 'Area cover', 'Dataset', 'Nominal',  'n_train', 'n_cal',  'n_test']].mean()
            agg_result_std = final_result_baseline.groupby(['Method'])[['Coverage', 'Conditional coverage',\
                                'Area', 'Area cover', 'Dataset', 'Nominal',  'n_train', 'n_cal',  'n_test']].std() / np.sqrt(i)
            agg_result = agg_result_mean.append(agg_result_std)
            agg_result.to_csv(out_file, index=True, float_format="%.4f")
            final_gj.to_csv(outgj_file)
            print("Updated summary of results on\n {}".format(out_file))
            print(agg_result)
        elif method == 'copula':
            res_table_copula, gj_table = run_copula(i, res_table_copula, gj_table, args)
            final_gj = final_gj.append(gj_table)
            final_result_baseline = res_table_copula
            out_file = out_dir + f"/{args.dataset}" + "_copula_" + "n_runs_" + str(args.n_runs) + ".txt"
            outgj_file = out_dir + f"/{args.dataset}" + "_copula_subgroup_" + "n_runs_" + str(args.n_runs) + f"_{args.md_type}.txt"
            out_file_detailed = out_dir + f"/{args.dataset}" + "_copuladt_" + "n_runs_" + str(args.n_runs) + ".txt"
            # Write results on output files
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            final_result_baseline.to_csv(out_file_detailed)
            final_result_baseline.Area = final_result_baseline.Area.astype(float)
            final_result_baseline['Area cover'] = final_result_baseline['Area cover'].astype(float)
            agg_result_mean = final_result_baseline.groupby(['Method'])[['Coverage', 'Conditional coverage',\
                                'Area', 'Area cover', 'Dataset', 'Nominal',  'n_train', 'n_cal',  'n_test']].mean()
            agg_result_std = final_result_baseline.groupby(['Method'])[['Coverage', 'Conditional coverage',\
                                'Area', 'Area cover', 'Dataset', 'Nominal',  'n_train', 'n_cal',  'n_test']].std() / np.sqrt(i)
            agg_result = agg_result_mean.append(agg_result_std)
            agg_result.to_csv(out_file, index=True, float_format="%.4f")
            final_gj.to_csv(outgj_file)
            print("Updated summary of results on\n {}".format(out_file))
            print(agg_result)



