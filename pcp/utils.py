import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

from shapely.geometry import Point, Polygon

from pcp import coverage as conditional_coverage
from tqdm.autonotebook import tqdm
from scipy.spatial.distance import mahalanobis

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_predictions_pcp(pred, Y, X=None, cc_delta=0.1, cc_split=0.75):
    # Extract lower and upper prediction bands
    # evaluations
    coverages = []
    lengths = []
    n_interval = []
    bands = pred
    y_te = Y.reshape(-1,1)
    for i_test in (range(len(y_te))):
        band_i = bands[i_test]
        y_i = y_te[i_test]
        n_interval.append(len(band_i))
        coverage = 0
        length = 0
        for interval_i in band_i:
            length += (interval_i[1] - interval_i[0])
            if ((y_i>interval_i[0]) & (y_i<interval_i[1])):
                coverage = 1
        coverages.append(coverage)
        lengths.append(length)

    marg_coverage = np.mean(coverages)
    # to do
    wsc_coverage = conditional_coverage.wsc_unbiased(X, Y, pred, M=100, delta=cc_delta, test_size=cc_split)
    length = np.mean(lengths)
    # to do
    idx_cover = np.where(coverages)[0]
    length_cover = np.mean([lengths for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Length': [length], 'Length cover': [length_cover]})
    return out


def dumb_area_counting(points_origin, qt, grid_size = 100, caltype = 'uniform', density = None, cov = None):
    points_origin = np.array(points_origin)
    # get a bounding box
    if caltype == 'uniform' or caltype == 'filtered':
        qtmax = np.sqrt(qt)
        x_min = np.min(points_origin[:,0] - qtmax)
        x_max = np.max(points_origin[:,0] + qtmax)
        y_min = np.min(points_origin[:,1] - qtmax)
        y_max = np.max(points_origin[:,1] + qtmax)
    elif caltype == 'density':
        # loop to find max round
        density = np.array(density)
        qtmax = qt * max(density)
        x_min = np.min(points_origin[:,0] - qtmax)
        x_max = np.max(points_origin[:,0] + qtmax)
        y_min = np.min(points_origin[:,1] - qtmax)
        y_max = np.max(points_origin[:,1] + qtmax)
    elif caltype == 'cov':
        x_ = np.min(points_origin[:,0])
        xsize = (np.max(points_origin[:,0])-np.min(points_origin[:,0]))/100
        while mahalanobis(np.array([np.min(points_origin[:,0]),0]), np.array([x_,0]), cov) < 2 * qt:
            x_ -= xsize
        x_min = x_
        x_ = np.max(points_origin[:,0])
        while mahalanobis(np.array([np.max(points_origin[:,0]),0]), np.array([x_,0]), cov) < 2 * qt:
            x_ += xsize
        x_max = x_

        y_ = np.min(points_origin[:,1])
        ysize = (np.max(points_origin[:,1])-np.min(points_origin[:,1]))/100
        while mahalanobis(np.array([np.min(points_origin[:,0]),0]), np.array([y_,0]), cov) < 2 * qt:
            y_ -= ysize
        y_min = y_
        y_ = np.max(points_origin[:,1])
        while mahalanobis(np.array([np.max(points_origin[:,1]),1]), np.array([y_,1]), cov) < 2 * qt:
            y_ += ysize
        y_max = y_
    area = (x_max - x_min) * (y_max - y_min)
    # get grid inside
    xgrid = np.linspace(x_min, x_max, grid_size)
    ygrid = np.linspace(y_min, y_max, grid_size)
    xg, yg = np.meshgrid(xgrid, ygrid)
    total_count = 0
    area_count = 0

    xygrid = np.array(list(zip(xg.reshape(-1), yg.reshape(-1))))

    if caltype == 'uniform':
        if_in = [np.sum((i - xygrid)**2, 1) < qt for i in points_origin]
        areacover = np.any(np.array(if_in), 0).mean() * area
    elif caltype == 'density':
        if_in = [np.sum((i - xygrid)**2, 1) < qt * j for i, j in zip(points_origin, density)]
        areacover = np.any(np.array(if_in), 0).mean() * area
    elif caltype == 'cov':
        from scipy.spatial.distance import cdist
        if_in = [(cdist(i, xygrid, metric = 'mahalanobis', VI = cov) < qt) for i in points_origin]
        areacover = np.any(np.array(if_in), 0).mean() * area
    return areacover

def evaluate_predictions_pcp_md(pred, qt, Y, X=None, caltype = 'uniform', density = None, cov = None, grid_size = 100):
    # Extract lower and upper prediction bands
    # evaluations
    coverages = []
    lengths = []
    n_interval = []
    points = np.array(pred)
    y_te = Y
    density = np.array(density)

    for i_test in tqdm(range(len(y_te))):
        point_i = points[i_test]
        y_i = y_te[i_test]
        if caltype == 'density':
            density_i = density[i_test]
        coverage = 0
        length = 0
        for j, sample in enumerate(point_i):
            if caltype == 'uniform' or caltype == 'filtered':
                qt_i = qt
                dist = sum((sample - y_i)**2)
            elif caltype == 'density':
                qt_i = qt * density_i[j]
                dist = sum((sample - y_i)**2)
            elif caltype == 'cov':
                qt_i = qt
                dist = mahalanobis(sample, y_i, cov)
            elif caltype == 'covcond':
                qt_i = qt
                dist = mahalanobis(sample, y_i, cov[i_test])
            if dist < qt_i:
                coverage = 1
                break
        if caltype == 'uniform' or caltype == 'filtered':
            length = dumb_area_counting(points[i_test], qt, grid_size = grid_size)
        elif caltype == 'density':
            length = dumb_area_counting(points[i_test], qt, grid_size = 100, caltype = caltype, density = density[i_test])
        elif caltype == 'cov':
            length = dumb_area_counting(points[i_test], qt, grid_size = 100, caltype = caltype, cov = cov)
        elif caltype == 'covcond':
            length = dumb_area_counting(points[i_test], qt, grid_size = 100, caltype = 'cov', cov = cov[i_test])
        coverages.append(coverage)
        lengths.append(length)

    marg_coverage = np.mean(coverages)
    print(marg_coverage)
    # to do
    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_md(X, Y, pred, qt, M=100, mdtype = 'pcp', caltype = caltype, cov = cov)

    #wsc_coverage = 1
    length = np.mean(lengths)
    # to do
    idx_cover = np.where(coverages)[0]
    length_cover = np.mean([lengths for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out


def taxi_subgroup_pcp(pred, qt, Y, X_test = None, caltype = 'uniform', density = None):
    coverages = []
    lengths = []
    n_interval = []
    points = np.array(pred)
    y_te = Y
    density = np.array(density)

    import geopandas as gpd
    gj = gpd.read_file('./mh.geojson')

    neighbor_list = []
    for i in range(X_test.shape[0]):
        loc = Point(X_test[i,-2],X_test[i,-1])
        iffind = 0
        for j in range(gj.shape[0]):
            if gj.iloc[j]['geometry'].contains(loc):
                neighbor_list.append(j)
                iffind = 1
        if iffind == 0:
            neighbor_list.append(-1)

    gj['coverage_count'] = 0
    gj['total_count'] = 0
    for i_test in tqdm(range(len(y_te))):
        if neighbor_list[i_test] != -1:
            gj.loc[neighbor_list[i_test],'total_count'] += 1
        point_i = points[i_test]
        y_i = y_te[i_test]
        if caltype == 'density':
            density_i = density[i_test]
        coverage = 0
        length = 0
        for j, sample in enumerate(point_i):
            if caltype == 'uniform' or caltype == 'filtered':
                qt_i = qt
            elif caltype == 'density':
                qt_i = qt * density_i[j]
            dist = sum((sample - y_i)**2)
            if dist < qt_i:
                if neighbor_list[i_test] != -1:
                    gj.loc[neighbor_list[i_test],'coverage_count'] += 1
                break
        if caltype == 'uniform' or caltype == 'filtered':
            length = dumb_area_counting(points[i_test], qt)
        elif caltype == 'density':
            length = dumb_area_counting(points[i_test], qt, grid_size = 100, caltype = caltype, density = density[i_test])
        coverages.append(coverage)
        lengths.append(length)

    marg_coverage = np.mean(coverages)
    gj['coverage'] = gj['coverage_count'] / gj['total_count']
    return gj , neighbor_list

def taxi_subgroup_baselines(pred_y0, pred_y1, Y_test, X_test = None):
    coverages = []
    lengths = []
    n_interval = []
    y_te = Y_test

    import geopandas as gpd
    gj = gpd.read_file('./mh.geojson')

    neighbor_list = []
    for i in range(X_test.shape[0]):
        loc = Point(X_test[i,-2],X_test[i,-1])
        iffind = 0
        for j in range(gj.shape[0]):
            if gj.iloc[j]['geometry'].contains(loc):
                neighbor_list.append(j)
                iffind = 1
        if iffind == 0:
            neighbor_list.append(-1)

    gj['coverage_count'] = 0
    gj['total_count'] = 0


    pred_l0 = np.min(pred_y0,1)
    pred_h0 = np.max(pred_y0,1)
    pred_l1 = np.min(pred_y1,1)
    pred_h1 = np.max(pred_y1,1)
    # Marginal coverage for each group
    sgcover = []
    for sg in range(gj.shape[0]):
        cover = np.mean([i and j for i,j,k in zip(((Y_test[:,0]>=pred_l0)*(Y_test[:,0]<=pred_h0)),((Y_test[:,1]>=pred_l1)*(Y_test[:,1]<=pred_h1)),neighbor_list) if k==sg])
        sgcover.append(cover)
    gj['sgcoverage'] = sgcover
    return gj , neighbor_list


def taxi_subgroup_baselines_cd(pred_y0, pred_y1, Y_test, X_test = None):
    coverages = []
    lengths = []
    n_interval = []
    y_te = Y_test

    import geopandas as gpd
    gj = gpd.read_file('./mh.geojson')

    neighbor_list = []
    for i in range(X_test.shape[0]):
        loc = Point(X_test[i,-2],X_test[i,-1])
        iffind = 0
        for j in range(gj.shape[0]):
            if gj.iloc[j]['geometry'].contains(loc):
                neighbor_list.append(j)
                iffind = 1
        if iffind == 0:
            neighbor_list.append(-1)

    gj['coverage_count'] = 0
    gj['total_count'] = 0

    sgcover = []
    for sg in range(gj.shape[0]):
        cover = []
        for i in range(len(pred_y0)):
            # iterate over all points
            if neighbor_list[i] != sg:
                continue
            x_in0 = 0
            x_in1 = 0
            for x_ in pred_y0[i]:
                if (Y_test[i,0] > x_[0]) and (Y_test[i,0] < x_[1]):
                    x_in0 = 1
                    break
                x_in0 = 0
            for x_ in pred_y1[i]:
                if (Y_test[i,1] > x_[0]) and (Y_test[i,1] < x_[1]):
                    x_in1 = 1
                    break
                x_in1 = 0
            cover.append((x_in0 and x_in1))
        sgcover.append(np.mean(cover))
    marg_coverage = np.mean(cover)
    gj['sgcoverage'] = sgcover
    return gj , neighbor_list

def evaluate_predictions_baselines_md(pred_y0, pred_y1, Y_test, X=None):
    # Extract lower and upper prediction bands
    pred_l0 = np.min(pred_y0,1)
    pred_h0 = np.max(pred_y0,1)
    pred_l1 = np.min(pred_y1,1)
    pred_h1 = np.max(pred_y1,1)
    # Marginal coverage
    cover = [i and j for i,j in zip(((Y_test[:,0]>=pred_l0)*(Y_test[:,0]<=pred_h0)),((Y_test[:,1]>=pred_l1)*(Y_test[:,1]<=pred_h1)))]
    marg_coverage = np.mean(cover)

    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_md(X, Y_test, [pred_y0, pred_y1], 1, M=100, mdtype = 'md')

    # Marginal length
    lengths = (pred_h0-pred_l0) * (pred_h1-pred_l1)
    length = np.mean(lengths)
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out


def dumb_area_counting_cd(pred_y0, pred_y1, grid_size = 100):
    points_origin = np.array(points_origin)
    # get a bounding box
    x_min = np.min(points_origin[:,0])
    x_max = np.max(points_origin[:,0])
    y_min = np.min(points_origin[:,1])
    y_max = np.max(points_origin[:,1])
    area = (x_max - x_min) * (y_max - y_min)
    # get grid inside
    xgrid = np.linspace(x_min, x_max, grid_size)
    ygrid = np.linspace(y_min, y_max, grid_size)
    xg, yg = np.meshgrid(xgrid, ygrid)
    total_count = 0
    area_count = 0
    for xc, yc in zip(xg.reshape(-1), yg.reshape(-1)):
        # whether this
        total_count += 1
        for i in points_origin:
            if sum((i-np.array([xc,yc]))**2) < qt:
                area_count += 1
                break
    return (area_count / total_count)*area


def evaluate_predictions_baselines_md_cd(pred_y0, pred_y1, Y_test, X=None):
    # pred_y0, pred_y1 is list of bands
    # Extract lower and upper prediction bands
    # Marginal coverage
    cover = []
    for i in range(len(pred_y0)):
        # iterate over all points
        for x_ in pred_y0[i]:
            if (Y_test[i,0] > x_[0]) and (Y_test[i,0] < x_[1]):
                x_in0 = 1
                break
            x_in0 = 0
        for x_ in pred_y1[i]:
            if (Y_test[i,1] > x_[0]) and (Y_test[i,1] < x_[1]):
                x_in1 = 1
                break
            x_in1 = 0
        cover.append((x_in0 and x_in1))
    marg_coverage = np.mean(cover)

    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_md(X, Y_test, [pred_y0, pred_y1], 1, M=100, mdtype = 'cd')

    # Marginal length
    lengths = []
    for i in range(len(pred_y0)):
        area = sum([l[1]-l[0] for l in pred_y0[i]]) * sum([l[1]-l[0] for l in pred_y1[i]])
        lengths.append(area)
    length = np.mean(lengths)
    print(lengths[0])
    print(lengths[1])
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths[i] for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out

def evaluate_predictions_baselines_mmd(pred_ys, Y_test, X=None):
    # Extract lower and upper prediction bands
    pred_ys = np.array(pred_ys)
    pred_l = pred_ys[:,:,0]
    pred_h = pred_ys[:,:,1]
    cover_cond = (Y_test[:,0] < pred_h[:,0]) * (Y_test[:,0] > pred_l[:,0])
    for i in range(Y_test.shape[1]):
        cover_cond = cover_cond * (Y_test[:,i] < pred_h[:,i]) * (Y_test[:,i] > pred_l[:,i])
    cover = np.mean(cover_cond)
    marg_coverage = cover

    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_mmd(X, Y_test, pred_ys, 1, M=100, mdtype = 'md')
        #wsc_coverage = wsc_unbiased_mmd(X, Y_test, pred_ys, 1, M=100, mdtype = 'md')
    # Marginal volume
    lengths = np.prod(pred_h-pred_l, axis = 1)
    length = lengths.mean()
    # Length conditional on coverage
    idx_cover = np.where(cover_cond)[0]
    length_cover = np.mean([lengths for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out



def evaluate_predictions_baselines_mmd_cd(pred_ys, Y_test, X_test=None):
    # pred_y0, pred_y1 is list of bands
    # Extract lower and upper prediction bands
    # Marginal coverage
    cover = []
    for i in range(len(pred_ys)):
        # iterate over all points
        allhit = 1
        for d in range(len(pred_ys[i])):
            thisdvalue = Y_test[i,d]
            thisdhit = 0
            for x_ in pred_ys[i][d]:
                l, h = x_
                if thisdvalue <= h and thisdvalue >= l:
                    thisdhit = 1
                    break
            allhit = allhit * thisdhit
            if allhit == 0:
                break
        cover.append(allhit)
    marg_coverage = np.mean(cover)

    if X_test is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_mmd(X_test, Y_test, pred_ys, 1, M=100, mdtype = 'cd')

    # Marginal length
    lengths = []
    for i in range(len(pred_ys)):
        area = 1
        for d in range(len(pred_ys[i])):
            thisd = 0
            for x_ in pred_ys[i][d]:
                l, h = x_
                thisd += h-l
            area = area * thisd
        lengths.append(area)
    length = np.mean(lengths)
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths[i] for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out


def dumb_area_counting_mmd(points_origin, qt, grid_size = 100, caltype = 'uniform', density = None):
    points_origin = np.array(points_origin)
    # get a bounding box
    if caltype == 'uniform':
        qtmax = qt
    elif caltype == 'density':
        # loop to find max round
        density = np.array(density)
        qtmax = qt * max(density)


    xsmin = np.min(points_origin,0) - np.sqrt(qtmax)
    xsmax = np.max(points_origin,0) + np.sqrt(qtmax)

    area = (xsmax[1] - xsmin[1]) * (xsmax[0] - xsmin[0])
    # get grid inside
    # get random sample
    total_count = 0
    area_count = 0
    num_trial = 1000
    for j in range(num_trial):
        random_sample = [np.random.uniform(xsmin[i], xsmax[i]) for i in [pairind1, pairind2]]
        total_count += 1
        for ind, i in enumerate(points_origin):
            if caltype == 'uniform':
                qt_i = qt
            elif caltype == 'density':
                qt_i = qt * density[ind]
            if sum((i-np.array(random_sample))**2) < qt_i:
                area_count += 1
                break
    return (area_count / total_count)*area

def evaluate_predictions_pcp_mmd(pred, qt, Y_test, X_test=None, caltype = 'uniform', density = None):
    # Extract lower and upper prediction bands
    # evaluations
    coverages = []
    lengths = []
    n_interval = []
    points = np.array(pred)
    density = np.array(density)

    pairind1 = 0
    pairind2 = 1
    for i_test in tqdm(range(len(Y_test))):
        point_i = points[i_test]
        y_i = Y_test[i_test]
        if caltype == 'density':
            density_i = density[i_test]
        coverage = 0
        length = 0
        for j, sample in enumerate(point_i):
            if caltype == 'uniform':
                qt_i = qt
            elif caltype == 'density':
                qt_i = qt * density_i[j]
            dist = sum((sample - y_i)**2)
            if dist < qt_i:
                coverage = 1
                break
        #point_i = point_i[:,[pairind1, pairind2]]
        if caltype == 'uniform':
            length = dumb_area_counting(point_i, qt, grid_size = 100, caltype = caltype)
        elif caltype == 'density':
            length = dumb_area_counting_mmd(point_i, qt, grid_size = 100, caltype = caltype, density = density[i_test])
        coverages.append(coverage)
        lengths.append(length)

    print(np.mean(coverages))
    print(np.mean(lengths))
    marg_coverage = np.mean(coverages)
    # to do
    if X_test is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = conditional_coverage.wsc_unbiased_mmd(X_test, Y_test, pred, qt, M=100, mdtype = 'pcp', caltype = caltype)

    #wsc_coverage = 1
    length = np.mean(lengths)
    # to do
    idx_cover = np.where(coverages)[0]
    length_cover = np.mean([lengths for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Area': [length], 'Area cover': [length_cover]})
    return out
