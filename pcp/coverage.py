import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
from scipy.spatial.distance import mahalanobis

def pcp_cover(pred, Y):
    coverages = []
    bands = pred
    y_te = Y.reshape(-1, 1)
    for i_test in range(len(y_te)):
        band_i = bands[i_test]
        y_i = y_te[i_test]
        coverage = 0
        for interval_i in band_i:
            if ((y_i > interval_i[0]) & (y_i < interval_i[1])):
                coverage = 1
        coverages.append(coverage)
    return np.array(coverages)

def pcp_cover_md(pred, qt, Y, caltype = None, cov = None):
    coverages = []
    lengths = []
    n_interval = []
    points = np.array(pred)
    y_te = Y
    for i_test in range(len(y_te)):
        point_i = points[i_test]
        y_i = y_te[i_test]
        coverage = 0
        length = 0
        for sample in point_i:
            if caltype == 'uniform' or caltype == 'filtered':
                dist = sum((sample - y_i)**2)
                if dist < qt:
                    coverage = 1
                    break
            elif caltype == 'cov':
                dist = mahalanobis(sample, y_i, cov)
                if dist < qt:
                    coverage = 1
                    break
            elif caltype == 'covcond':
                dist = mahalanobis(sample, y_i, cov[i_test])
                if dist < qt:
                    coverage = 1
                    break
        coverages.append(coverage)
    return np.array(coverages)

def cover_md(pred_y0, pred_y1, Y):
    # unimode
    pred_l0 = np.min(pred_y0,1)
    pred_h0 = np.max(pred_y0,1)
    pred_l1 = np.min(pred_y1,1)
    pred_h1 = np.max(pred_y1,1)
    # Marginal coverage
    coverages = [i and j for i,j in zip(((Y[:,0]>=pred_l0)*(Y[:,0]<=pred_h0)),((Y[:,1]>=pred_l1)*(Y[:,1]<=pred_h1)))]
    return np.array(coverages)


def cd_cover_md(pred_y0, pred_y1, Y):
    cover = []
    for i in range(len(pred_y0)):
        # iterate over all points
        for x_ in pred_y0[i]:
            if (Y[i,0] > x_[0]) and (Y[i,0] < x_[1]):
                x_in0 = 1
                break
            x_in0 = 0
        for x_ in pred_y1[i]:
            if (Y[i,1] > x_[0]) and (Y[i,1] < x_[1]):
                x_in1 = 1
                break
            x_in1 = 0
        cover.append((x_in0 and x_in1))
    return np.array(cover)

def cover_mmd(pred_ys, Y_test):
    pred_l = pred_ys[:,:,0]
    pred_h = pred_ys[:,:,1]
    cover_cond = (Y_test[:,0] < pred_h[:,0]) * (Y_test[:,0] > pred_l[:,0])
    for i in range(Y_test.shape[1]):
        cover_cond = cover_cond * (Y_test[:,i] < pred_h[:,i]) * (Y_test[:,i] > pred_l[:,i])
    return cover_cond


def pcp_cover_mmd(pred_ys, qt, Y_test, caltype = None):
    points = np.array(pred_ys)
    coverages = []
    for i_test in (range(len(Y_test))):
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
        coverages.append(coverage)
    return np.array(coverages)


def cd_cover_mmd(pred_ys, Y_test):
    cover = []
    for i in range(len(pred_ys)):
        # iterate over all points
        allhit = 1
        for d in range(len(pred_ys[i])):
            thisdvalue = Y_test[i,d]
            thisdhit = 0
            for x_ in pred_ys[i][d]:
                l, h = x_
                if thisdvalue < h and thisdvalue > l:
                    thisdhit = 1
                    break
            allhit = allhit * thisdhit
            if allhit == 0:
                break
        cover.append(allhit)
    return np.array(cover)

def wsc(X, y, pred, delta=0.1, M=1000, verbose=False):
    # Extract lower and upper prediction bands

    def wsc_v(X, y, pred, delta, v):
        n = len(y)
        cover = pcp_cover(pred, y)
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = np.mean(cover)
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased(X, y, pred, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False):
    def wsc_vab(X, y, pred, v, a, b):
        n = len(y)
        cover = pcp_cover(pred, y)
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(X, y, pred, test_size=test_size,
                                                                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, pred_train, delta=delta, M=M, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, pred_test, v_star, a_star, b_star)
    return coverage



def wsc_md(X, y, pred, qt, delta=0.1, M=1000, verbose=False, mdtype = 'pcp', caltype = None, cov = None):
    # Extract lower and upper prediction bands

    def wsc_v(X, y, pred, delta, v):
        n = len(y)
        if mdtype == 'pcp':
            cover = pcp_cover_md(pred, qt, y, caltype, cov)
        elif mdtype == 'cd':
            pred_y0, pred_y1 = pred[0], pred[1]
            cover = cd_cover_md(pred_y0, pred_y1, y)
        else:
            pred_y0, pred_y1 = pred[0], pred[1]
            cover = cover_md(pred_y0, pred_y1, y)
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = np.mean(cover)
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star



def wsc_mmd(X, y, pred, qt, delta=0.1, M=1000, verbose=False, mdtype = 'pcp', caltype = None):
    # Extract lower and upper prediction bands

    def wsc_v(X, y, pred, delta, v):
        n = len(y)
        if mdtype == 'pcp':
            cover = pcp_cover_mmd(pred, qt, y, caltype = caltype)
        elif mdtype == 'cd':
            cover = cd_cover_mmd(pred, y)
        else:
            cover = cover_mmd(pred, y)
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = np.mean(cover)
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pred, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased_md(X, y, pred, qt, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False, mdtype = 'pcp', caltype = None, cov = None):
    def wsc_vab_md(X, y, pred, v, a, b, caltype = None, cov = None):
        n = len(y)
        if mdtype == 'pcp':
            cover = pcp_cover_md(pred, qt, y, caltype, cov)
        elif mdtype == 'cd':
            pred_y0, pred_y1 = pred[0], pred[1]
            cover = cd_cover_md(pred_y0, pred_y1, y)
        else:
            pred_y0, pred_y1 = pred[0], pred[1]
            cover = cover_md(pred_y0, pred_y1, y)
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    if mdtype == 'pcp':
        X, y, pred = np.array(X), np.array(y), np.array(pred)
        X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(X, y, pred, test_size=test_size,
                                                                         random_state=random_state)
    else:
        pred_y0, pred_y1 = pred[0], pred[1]
        X_train, X_test, y_train, y_test, predy0_train, predy0_test, predy1_train, predy1_test = train_test_split(X, y, pred[0], pred[1], test_size=test_size,
                                                                         random_state=random_state)
        pred_train = [predy0_train, predy1_train]
        pred_test = [predy0_test, predy1_test]
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc_md(X_train, y_train, pred_train, qt, delta=delta, M=M, verbose=verbose, mdtype = mdtype, caltype = caltype, cov = cov)
    # Estimate coverage
    coverage = wsc_vab_md(X_test, y_test, pred_test, v_star, a_star, b_star, caltype, cov = cov)
    return coverage

def wsc_unbiased_mmd(X, y, pred_ys, qt, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False, mdtype = 'pcp', caltype = None):
    def wsc_vab_mmd(X, y, pred_ys, v, a, b, caltype = caltype):
        n = len(y)
        if mdtype == 'pcp':
            cover = pcp_cover_mmd(pred_ys, qt, y, caltype = caltype)
        elif mdtype == 'cd':
            cover = cd_cover_mmd(pred_ys, y)
        else:
            cover = cover_mmd(pred_ys, y)
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    if mdtype == 'pcp':
        X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(X, y, pred_ys, test_size=test_size,
                                                                         random_state=random_state)
    else:
        X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(X, y, pred_ys, test_size=test_size,
                                                                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc_mmd(X_train, y_train, pred_train, qt, delta=delta, M=M, verbose=verbose, mdtype = mdtype, caltype = caltype)
    # Estimate coverage
    coverage = wsc_vab_mmd(X_test, y_test, pred_test, v_star, a_star, b_star, caltype = caltype)
    return coverage
