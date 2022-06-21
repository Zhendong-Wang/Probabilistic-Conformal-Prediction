import torch
import numpy as np
import pandas as pd
import pcp.datasets as datasets

from sklearn.preprocessing import MinMaxScaler


def one_hot(df, cols):
  """Returns one-hot encoding of DataFrame df including columns in cols."""
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df


def GetDataset(name, base_path, seed = 0, rho=0.5):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

    """
    np.random.seed(seed)
    if name == "meps_19":
        df = pd.read_csv(base_path + 'meps_19_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        #        y = np.log(1 + y - min(y))
        X = df[col_names].values

    if name == "meps_20":
        df = pd.read_csv(base_path + 'meps_20_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        #        y = np.log(1 + y - min(y))
        X = df[col_names].values

    if name == "meps_21":
        df = pd.read_csv(base_path + 'meps_21_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        #        y = np.log(1 + y - min(y))
        X = df[col_names].values

    if name=="facebook_1":
        df = pd.read_csv(base_path + 'facebook_1.csv')
        y = df.iloc[:,53].values
#        y = np.log(1 + y - min(y))
        X = df.iloc[:,0:53].values

    if name=="facebook_2":
        df = pd.read_csv(base_path + 'facebook_2.csv')
        y = df.iloc[:,53].values
#        y = np.log(1 + y - min(y))

        X = df.iloc[:,0:53].values

    if name=="bio":
        #https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values

    if name=='blog_data':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:,0:280].values
        y = df.iloc[:,-1].values

    if name == "temperature":
        df = pd.read_csv(base_path + 'Bias_correction_ucl.csv')
        df = df.drop(columns=['station', 'Date', 'Next_Tmax'])
        df = df.dropna()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    if name=="bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df=pd.read_csv(base_path + 'bike_train.csv')

        # # seperating season as per values. this is bcoz this will enhance features.
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)

        # # # now can drop weather and season.
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})

        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values
        y = df['count'].values

    if name == 'taxi':
        nsamples = 10000
        dataset = datasets.NCYTaxiDropoffPredict(n_samples=nsamples, seed = seed)
        data = dataset.get_df()
        X, y = data[['pickup_time_day_of_week_sin', 'pickup_time_day_of_week_cos', 'pickup_time_of_day_sin', 'pickup_time_of_day_cos','pickup_loc_lat','pickup_loc_lon']].values, \
                data[['dropoff_loc_lat', 'dropoff_loc_lon']].values
    if name == 'energy':
        dataset = datasets.Energy()
        data = dataset.get_df()
        X, y = data[['X1','X2','X3','X4','X5','X6','X7','X8']].values, \
                data[['Y1', 'Y2']].values
    if name == 'rf1':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'rf1.arff')
        df = pd.DataFrame(df[0])
        #df = df.dropna()
        X, y = df.iloc[:,:-8].values, df.iloc[:,-8:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'rf2':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'rf2.arff')
        df = pd.DataFrame(df[0])
        #df = df.dropna()
        X, y = df.iloc[:,:-8].values, df.iloc[:,-8:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'scm1d':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'scm1d.arff')
        df = pd.DataFrame(df[0])
        df = df.dropna()
        X, y = df.iloc[:,:-16].values, df.iloc[:,-16:].values
    if name == 'scm20d':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'scm20d.arff')
        df = pd.DataFrame(df[0])
        df = df.dropna()
        X, y = df.iloc[:,:-16].values, df.iloc[:,-16:].values
        #from sklearn.impute import SimpleImputer
        #imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        #X = imputer.fit_transform(X)
    if name == 'osales':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'osales.arff')
        df = pd.DataFrame(df[0])
        X, y = df.iloc[:,:-12].values, df.iloc[:,-12:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'scpf':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'scpf.arff')
        df = pd.DataFrame(df[0])
        X, y = df.iloc[:,:-3].values, df.iloc[:,-3:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'sf2':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'sf2.arff')
        df = pd.DataFrame(df[0])
        X, y = df.iloc[:,:-3].values, df.iloc[:,-3:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'sgemm':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'dataset.arff')
        df = pd.DataFrame(df[0])
        df = df.sample(n=10000, replace = False, random_state = seed)
        X, y = df.iloc[:,:-4].values, df.iloc[:,-4:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'stock':
        from scipy.io import arff
        df = arff.loadarff(base_path + 'dataset_2209_stock.arff')
        df = pd.DataFrame(df[0])
        X, y = df.iloc[:,:-3].values, df.iloc[:,-3:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
    if name == 'mmdsyn':
        np.random.seed(seed)
        nsamples = 10000
        ndim = 5
        X = np.random.normal(size=(nsamples, ndim))
        # linear regression coefficient
        alpha = 5
        rho = rho
        coefx1 = np.random.normal(size = ndim + 1) * alpha
        coefx2 = np.random.normal(size = ndim + 1) * alpha
        coefy1 = np.random.normal(size = ndim + 1) * alpha
        coefy2 = np.random.normal(size = ndim + 1) * alpha

        cluster = np.random.randint(2, size=nsamples).reshape(-1,1)
        mux1 = np.sum(X*coefx1[:(ndim)],1) + coefx1[ndim]
        mux2 = np.sum(X*coefx2[:(ndim)],1) + coefx2[ndim]
        muy1 = np.sum(X*coefy1[:(ndim)],1) + coefy1[ndim]
        muy2 = np.sum(X*coefy2[:(ndim)],1) + coefy2[ndim]

        mu1 = np.concatenate((mux1.reshape(-1,1),muy1.reshape(-1,1)),axis=1)
        mu2 = np.concatenate((mux2.reshape(-1,1),muy2.reshape(-1,1)),axis=1)
        mu = mu1 * cluster + mu2 * (1-cluster)
        # diagonal binormal
        y = np.array([np.random.multivariate_normal(mean = i, cov = [[10,rho],[rho,10]]) for i in mu])

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X, y


class Data_Sampler(object):
    def __init__(self, x, y, device, scale_y=False):
        self.n_samples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.scale_y = scale_y
        if scale_y:
            self.y_scaler = MinMaxScaler().fit(y.reshape(-1, 1))
            self.y = torch.from_numpy(self.y_scaler.transform(y.reshape(-1, 1))).float()
        else:
            self.y = torch.from_numpy(y).float()

        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(low=0, high=self.n_samples, size=(batch_size,))
        return self.x[ind].to(self.device), self.y[ind].view(-1, 1).to(self.device)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y.view(-1, 1)


class Data_Sampler_MD(object):
    def __init__(self, x, y, device, seed = 0):
        self.n_samples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        torch.manual_seed(seed)
        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(low=0, high=self.n_samples, size=(batch_size,))
        ind = ind.long()
        return self.x[ind].to(self.device), self.y[ind].to(self.device)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
