import os
import sys
import torch
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

""" Load baseline Methods """
from chr.black_boxes import QNet, QRF
from chr.methods import CHR
from other_baselines.cqr import CQR, CQR2
from other_baselines.dist_split import DistSplit
from other_baselines.dcp import DCP
from other_baselines.cd_split import CDSplit

from chr.utils import evaluate_predictions

""" Load PCP Methods """
from pcp.pcp import PCP
from pcp.models.gan import GAN
from pcp.models.sivi import SIVI
from cde.density_estimator import KernelMixtureNetwork, MixtureDensityNetwork
from pcp.utils import evaluate_predictions_pcp

""" Load dataset methods """
from dataset import GetDataset, Data_Sampler
from sklearn.preprocessing import StandardScaler
import time

import multiprocessing


def _run(args, out_dir, seed):

    # Default arguments
    alpha = args.alpha
    base_dataset_path = './data/'
    n_jobs = 1
    verbose = False

    dataset_name = args.dataset
    print(f"data: {dataset_name}")

    # Set random seed
    random_state = seed
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    X, Y = GetDataset(dataset_name, base_dataset_path)
    Y += 1e-6*np.random.normal(size=Y.shape)  # Add noise to response
    y_min = min(Y)
    y_max = max(Y)

    out_file = out_dir + f"/{dataset_name}_alpha_{alpha}" + f"_seed_{seed}" + ".txt"
    print(out_file)

    results = pd.DataFrame()

    # Split the data
    n_total = X.shape[0]
    n_test = 2000  # min(2000, int(n_total * 0.2))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test, random_state=random_state)
    n_cal = 2000  # min(2000, int(X_train.shape[0] * 0.1))
    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    n_train = X_train.shape[0]
    assert(n_cal == X_calib.shape[0])
    n_test = X_test.shape[0]

    if len(X.shape) == 1:
        n_features = 1
    else:
        n_features = X.shape[1]

    # Training models for CHR, CQR ...
    """ NNet """
    epochs = args.n_epochs
    lr = 0.0005
    batch_size = args.batch_size
    dropout = 0.1
    grid_quantiles = np.arange(0.01,1.0,0.01)

    bbox_nn = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                   dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1,
                   verbose=verbose, random_state=seed)
    print("Training black box model NNet...")
    bbox_nn.fit(X_train, Y_train)
    """ QRF """
    n_estimators = 100
    min_samples_leaf = 50
    grid_quantiles = np.arange(0.01,1.0,0.01)
    bbox_rf = QRF(grid_quantiles, n_estimators=n_estimators,
                  min_samples_leaf=min_samples_leaf, random_state=seed,
                  n_jobs=n_jobs, verbose=verbose)
    print("Training black box model RF...")
    bbox_rf.fit(X_train, Y_train)

    """
    Train Probabilistic Model: GAN, SIVI, MixDensityNetwork, KernelMixtureNetwork
    """
    x_dim, y_dim = X_train.shape[1], 1
    z_dim = min(10, x_dim // 2)
    Y_train = Y_train.reshape(-1, 1)
    train_dataset = Data_Sampler(X_train, Y_train, device=args.device)

    gan_model = GAN(x_dim, y_dim, z_dim, args.H, args.batch_size, args.device,
                    adam_b1=args.b1,
                    adam_b2=args.b2)
    gan_model.train(train_dataset, args.batch_size, args.n_epochs)
    gan_model.eval()

    sivi_model = SIVI(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, H=128, batch_size=args.batch_size,
                      device=args.device, lr=args.lr)
    sivi_model.train(train_dataset, args.batch_size, args.n_epochs)

    mixd_model = MixtureDensityNetwork(name=f"MixD-{seed}", ndim_x=x_dim, ndim_y=y_dim, hidden_sizes=(100, 100),
                                       dropout=0.1, random_seed=seed)
    mixd_model.fit(X=X_train, Y=Y_train)

    kmn_model = KernelMixtureNetwork(name=f"KMN-{seed}", ndim_x=x_dim, ndim_y=y_dim, hidden_sizes=(100, 100),
                                     dropout=0.1, random_seed=seed)
    kmn_model.fit(X=X_train, Y=Y_train)

    pcp_sivi = PCP(model=sivi_model, base="torch", device=args.device, alpha=args.alpha, sample_K=args.K)
    pcp_gan = PCP(model=gan_model, base="torch", device=args.device, alpha=args.alpha, sample_K=args.K)
    pcp_qrf = PCP(model=bbox_rf, base="numpy", device=args.device, alpha=args.alpha, sample_K=args.K)
    pcp_mixd = PCP(model=mixd_model, base="numpy", device=args.device, alpha=args.alpha, sample_K=args.K)
    pcp_kmn = PCP(model=kmn_model, base="numpy", device=args.device, alpha=args.alpha, sample_K=args.K)

    hd_pcp_mixd = PCP(model=mixd_model, base="numpy", device=args.device, alpha=args.alpha,
                      sample_K=int(args.K / (1. - args.fr)), fr=args.fr, cal_type='filtered')
    hd_pcp_kmn = PCP(model=kmn_model, base="numpy", device=args.device, alpha=args.alpha,
                     sample_K=int(args.K / (1. - args.fr)), fr=args.fr, cal_type='filtered')

    # Define list of methods to use in experiments
    methods = {
        'PCP-SIVI'    : pcp_sivi,
        'PCP-GAN'     : pcp_gan,
        'PCP-QRF'     : pcp_qrf,
        'PCP-MixD'    : pcp_mixd,
        'PCP-KMN'     : pcp_kmn,
        'HD-PCP-MixD' : hd_pcp_mixd,
        'HD-PCP-KMN'  : hd_pcp_kmn,
        'CHR-NNet'    : CHR(bbox_nn, ymin=y_min, ymax=y_max, y_steps=1000, randomize=True),
        'CHR-RF'      : CHR(bbox_rf, ymin=y_min, ymax=y_max, y_steps=1000, randomize=True),
        'DistSplit'   : DistSplit(bbox_nn, ymin=y_min, ymax=y_max),
        'CDSplit-KMN' : CDSplit(X_train, Y_train.reshape(-1), ymin=y_min, ymax=y_max, seed=seed, alpha=args.alpha, model=kmn_model),
        'CDSplit-MixD': CDSplit(X_train, Y_train.reshape(-1), ymin=y_min, ymax=y_max, seed=seed, alpha=args.alpha, model=mixd_model),
        'DCP'         : DCP(bbox_nn, ymin=y_min, ymax=y_max),
        'CQR'         : CQR(bbox_nn),
        'CQR2'        : CQR2(bbox_nn)
    }

    for method_name in methods:
        t_start = time.time()
        print(method_name)
        # Apply the conformalization method
        method = methods[method_name]

        method.calibrate(X_calib, Y_calib, alpha)
        pred = method.predict(X_test)
        del method # for saving memory

        # Evaluate results
        if "PCP" in method_name or "CDSplit" in method_name:
            res = evaluate_predictions_pcp(pred, Y_test, X=X_test, cc_delta=0.4, cc_split=0.75)
        else:
            res = evaluate_predictions(pred, Y_test, X=X_test)
        # Add information about this experiment
        t_end = time.time()
        t_run = round(t_end - t_start, 2)
        print(f"time for {method_name} run: {t_run}")
        print(res)
        res['Dataset'] = dataset_name
        res['Method'] = method_name
        res['Nominal'] = 1-alpha
        res['run_time'] = t_run
        res['n_train'] = n_train
        res['n_cal'] = n_cal
        res['n_test'] = n_test

        # Add results to the list
        results = results.append(res)

        results.to_csv(out_file, index=False, float_format="%.4f")
        print("Updated summary of results on\n {}".format(out_file))
        sys.stdout.flush()


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
    parser.add_argument("--dataset", default="facebook_1", type=str)  # dataset name
    # Training parameters
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--n_parallel", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=250)
    # Implicit model training parameters
    parser.add_argument('--z_dim', type=int, default=15, metavar='N',
                        help='dimensionality of z (default: 50)')
    parser.add_argument('--H', type=int, default=100, metavar='N',
                        help='dimensionality of feature x_feat (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for Adam')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    # PCP parameters
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fr", type=float, default=0.2, help='Sample Filtering Ratio')
    parser.add_argument("--exp", type=str, default="0")

    args = parser.parse_args()
    #device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device("cpu")

    out_dir = f'./results_{args.exp}'
    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.n_runs // args.n_parallel):
        # parallel running
        p_s, p_e = i * args.n_parallel, (i+1) * args.n_parallel
        processes = []
        for sd in range(p_s, p_e):
            p = multiprocessing.Process(target=_run, args=(args, out_dir, sd))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
