#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-

# Functions used in the analysis
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.io as spio
from scipy import stats, signal
from collections import namedtuple
from multiprocessing import Pool
from functools import partial
import time

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def save_hdf(filn, df):
    df.to_hdf(filn, key='df', complib='blosc:zstd', complevel=9)
    print('%s is saved.' % filn)


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def matdToNtup(matdata):
    '''
    converts matdata to namedtuple variable
    '''
    for k in list(matdata.keys()):
        if k.startswith('_'):
            matdata.pop(k)
            
    return _convert(matdata), list(matdata.keys())

def _convert(dictionary):
    for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = _convert(value)
            if isinstance(value, list):
                dictionary[key] = [_convert(i) for i in value]
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def climdictToDf(dictdata, length='short'):
    '''converts dictionary of climate data to dataframe.
        
    length option is available (long or short)
    
    Parameters
    ----------
    dictdata: dict
        dictionary data of climate drivers
    length: str
        optional length of output array (long or short)
        *long = array during all available periods
        *short = array during the longest overlapping records
        
    Returns
    -------
    df: dataframe
        dataframe of climate drivers
    
    '''
    
# =============================================================================
#     # revision setting ------------------------------------------------------ #
#     import gf
#     length = 'short'
#     dictdata = gf.loadmat(os.path.join('data', 'lscidx.mat'))
#     for k in list(dictdata.keys()):
#         if not k.endswith('Linr'):
#             dictdata.pop(k)
#     for k in list(dictdata.keys()):
#         dictdata[k.replace('Linr', '')] = dictdata.pop(k)
# =============================================================================

    # build a dataframe
    d = dict()
    name = list(dictdata.keys())
    yrmo = np.empty([len(dictdata),2], 'datetime64[M]')     # [dt_start, dt_end]
    for i in range(len(dictdata)):
        temp = dictdata[name[i]]
        period = pd.period_range('{:04d}-{:02d}'.format(int(temp[0,0]),
                              int(temp[0,1])), periods=temp.shape[0], freq='M')
        d.update({name[i]: pd.Series(temp[:,2], index=period)})
        yrmo[i,0] = '{:04d}-{:02d}'.format(int(temp[0,0]), int(temp[0,1]))
        yrmo[i,1] = '{:04d}-{:02d}'.format(int(temp[-1,0]), int(temp[-1,1]))
    df = pd.DataFrame(d)
        
    # size of output array
    if length is 'short':
        period = pd.period_range(yrmo.max(0)[0], yrmo.min(0)[1], freq='M')
        df = df.loc[np.in1d(df.index.to_timestamp(), period.to_timestamp())]
    elif length is 'long':
        pass
    else:
        raise RuntimeError('the length option is not available')
        
    return df        

# =============================================================================
#     # get periods of records
#     strdict = list(dictdata.keys())
#     ndict = len(strdict)
#     yrmo = np.empty([len(dictdata),2], 'datetime64[M]')     # [dt_start, dt_end]
#     for i in range(ndict):
#         temp = dictdata[strdict[i]]
#         yrmo[i,0] = '{:04d}-{:02d}'.format(int(temp[0,0]), int(temp[0,1]))
#         yrmo[i,1] = '{:04d}-{:02d}'.format(int(temp[-1,0]), int(temp[-1,1]))
#         
#     # size of output array
#     if length is 'long':
#         dtdict = np.arange(yrmo.min(0)[0], yrmo.max(0)[1]+1)
#     elif length is 'short':
#         dtdict = np.arange(yrmo.max(0)[0], yrmo.min(0)[1]+1)
#     else:
#         raise RuntimeError('the length option is not available')
#     nt = len(dtdict)
#     
#     # insert data to output array
#     array = np.full([nt, ndict], np.nan)
#     for i in range(ndict):
#         iarr = np.in1d(dtdict, np.arange(yrmo[i][0], yrmo[i][1]+1))
#         idct = np.in1d(np.arange(yrmo[i][0], yrmo[i][1]+1), dtdict)
#         array[iarr,i] = dictdata[strdict[i]][idct,2]
# =============================================================================
    
#    return strdict, dtdict, array
    
    
#%%

def corr2d1d(x, y, alpha):
    '''
    Returns 1D Pearson's correlation of 2D array to 1D array.
    
    x       - 2D ndarray (nrow, tim)
    y       - 1D ndarray (1, tim)
    alpha   - significance level
    corr    - 1D ndarray of Pearson's correlation
    sign    - 1D boolean array of Two-sided T-Test result (1:sign, 0:none)
    '''
#    x = np.array([[1,2,3], [4,5,6], [7,8,9], [12,11,10], [13,15,17]])
#    y = np.array([1,2,3])[:,None]
    
    if y.ndim == 1:
        y = y[:,None]
    if x.ndim != 2 or y.ndim != 2:
        sys.exit('Array dimensions are not correct.')
    # DOF
    n = len(y)
    dof = n - 1
    # zscore
    xz = (x - x.mean(1)[:,None])/np.std(x, ddof=1, axis=1)[:,None]
    yz = (y - y.mean(0))/np.std(y, ddof=1)
    # Correlation
    corr = xz.dot(yz)/dof
    
    # Two-sided T-test
    tstat = corr*np.sqrt(n-2)/np.sqrt(1-corr**2)
    sign = np.abs(tstat) > stats.t.ppf(1-alpha/2, n-2)
    
    return corr, sign

# from http://gestaltrevision.be/wiki/python/simplestats
#def p_corr(df1, df2):
#    """
#    Computes Pearson correlation and its significance (using a t
#    distribution) on a pandas.DataFrame.
# 
#    Ignores null values when computing significance. Based on
#    http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Testing_using_Student.27s_t-distribution
# 
#    Args:
#        df1 (pandas.DataFrame): one dataset
#        df2 (pandas.DataFrame): another dataset
# 
#    Returns:
#        corr (float): correlation between the two datasets
#        t (float): an associated t-value
#        p (float): one-tailed p-value that the two datasets differ
#    """
#    corr = df1.corr(df2)
#    N = np.sum(df1.notnull())
#    t = corr*np.sqrt((N-2)/(1-corr**2))
#    p = 1-scipy.stats.t.cdf(abs(t),N-2)  # one-tailed    
#t, p = scipy.stats.ttest_rel(data1, data2, axis=0)
#    return corr, t, p


def corr1d1d_wrapper(args):
    x, y = args
    return corr1d1d_nan(x, y)

def corr1d1d_nan(x, y, alpha=0.05):
    '''
    Returns 1D Pearson's correlation of 1D array with NaN values to 1D array.
    
    Parameters:
    ----------
    x       - 1D ndarray (1, tim)
    y       - 1D ndarray (1, tim)
    alpha   - significance level
    
    Returns:
    --------
    corr    - 1D ndarray of Pearson's correlation
    sign    - 1D boolean array of Two-sided T-Test result (1:sign, 0:none)
    '''
#    x = np.array([[1,2,3], [4,5,6], [7,8,9], [12,11,10], [13,15,17]])
#    y = np.array([1,2,3])[:,None]

    if x.ndim != 1 or y.ndim != 1:
        sys.exit('Array dimensions are not correct.')
        
    # Exclude missing values
    dpx = np.isnan(x) | np.isnan(y)
    x = x[~dpx]
    y = y[~dpx]
    
    # DOF
    n = len(y)
    dof = n - 1
    # zscore
    xz = (x - x.mean())/np.std(x, ddof=1)
    yz = (y - y.mean())/np.std(y, ddof=1)
    # Correlation
    corr = xz.dot(yz)/dof
    
    # Two-sided T-test
    tstat = corr*np.sqrt(n-2)/np.sqrt(1-corr**2)
    sign = np.abs(tstat) > stats.t.ppf(1-alpha/2, n-2)
    
    return corr, sign


def yrmo(tim):
    '''
    Returns 2D array of [[year][month]]
    
    tim     - [year_start, year_end]
    '''
    year = range(tim[0], tim[1]+1)
    return np.vstack((np.repeat(year, 12),np.tile(range(1,13), len(year)))).T


def detrend_nan(x, y):
    '''detrends 1d time-series having nan values
    
    Parameters
    ----------
    x: ndarray
        time-index of the time series
    y: ndarray
        values of the time-series to be detrended
    
    Returns
    -------
    detrend_y: ndarray
        detrended values of the time-series
    
    '''
# =============================================================================
#     # Revision setting ------------------------------------------------------ #
#     # create data
#     x = np.linspace(0, 2*np.pi, 500)*10 + 100
#     y = np.random.normal(0.3*x, np.random.rand(len(x)))
#     drops = np.random.rand(len(x))
#     y[drops>.95] = np.NaN # add some random NaNs into y
#     plt.plot(x, y)
# =============================================================================

    # Find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind],y[not_nan_ind])
    detrend_y = y - (m*x + b)
#    plt.plot(x, detrend_y)
    
    return detrend_y


#%% corrClimGrid
def corrClimGrid(tim, flow, climMat, maxLeadMonth, alpha=0.05, flagProc=True):
    '''calculate lag-correlations of gridded streamflow with climate index
    
    Parameters
    -----------
    tim: list
        years of the start and end, [year_start, year_end]
    flow: ndarray
        gridded streamflow time-series, [grid_index, 12 months]
    climMat: ndarray
        climate index time-series in climatological form, 
        [time, [year, 12 months]]
    maxLeadMonth: int
        the number of lead months
    alpha: int
        confidence level of correlation, default is 0.05
    flagProc: bool
        flag of printing process, default is True
        
    Returns
    -------
    sheadMaxCorr: ndarray
        gridded maximum correlation value, [grid_index, 12 months]
    sheadMaxLead: ndarray
        gridded corresponding lead-time, [grid_index, 12 months]
    sheadMaxSign: ndarray
        gridded significance of the correlation, [grid_index, 12 months]
    
    '''
# =============================================================================
#     # REVISION SETTING
#     import os
#     import numpy as np
#     from scipy import signal 
#     import gf
#     tim = [1958, 2000]
#     maxLeadMonth = 8              # The maximum lead-month
#     stdFlag = 1; alpha = 0.05
#     flow = np.load(os.path.join('data', 'flow.npy'))
#     cidx = gf.loadmat(os.path.join('data', 'lscidx.mat'))
#     cidx, _ = gf.matdToNtup(cidx)
#     climMat = cidx.nao
#     flagProc = 1
# #    from attribute_corr import tim, flow, alpha, climMat, maxLeadMonth, flagProc
# =============================================================================

    # Initialize variables -------------------------------------------------- #
    yrFlow = np.arange(tim[0],tim[1]+1)
    dtFlow = np.arange(str(yrFlow[0])+'-01', str(yrFlow[-1]+1)+'-01', 
                       dtype='datetime64[M]')
    if dtFlow.shape[0] != len(flow[0]):
        raise RuntimeError('Time dimensions are not correct.')
    
    # Load climate index from 1 year prior to the first year of streamflow
    climMat = climMat[np.in1d(climMat[:,0], np.arange(tim[0]-1,tim[1]+1)),:]
    yrClim = climMat[:,0].astype(int)
    nyrClim = len(yrClim) 
    dtClim = np.arange(str(yrClim[0])+'-01', str(yrClim[-1]+1)+'-01', 
                       dtype='datetime64[M]')    

    # Log-transform, Detrend, Remove annual cycle --------------------------- #
    # Log-transformation
    flow[flow <= 0] = 0.001
    flow = np.log(flow)
    
    # Remove seasonal trends
    # *The long-term trends and annual cycles are eliminated while removing 
    #   seasonal trends
    # Streamflow
    for i in range(12):    
        flow[:,i::12] = signal.detrend(flow[:,i::12], axis=-1, type='linear')
    
    # Climate driver
    clim = signal.detrend(climMat[:,1::], axis=0, type='linear')
    clim = np.reshape(clim, [nyrClim*12, 1])
    
    
    # Computing lag-correlation between concurrent and maxLeadMonth --------- #
    if flagProc:
        print('Calculating grid-scale correlation at each month.')
    
    # Initialize correlation array including concurrent month
    gridCorr = np.full([len(flow[:,0]), (maxLeadMonth+1)*12], np.nan)   # Concurrent - maxLeadMonth
    gridSign = np.zeros(gridCorr.shape, dtype=bool)
    
    for iMon in range(12):
        for iLag in range(maxLeadMonth+1):
            # Define time-index
            dxFlow = dtFlow[iMon::12]                   # Time-index for Flow
            dxClim = dxFlow - np.timedelta64(iLag,'M')  # Time-index for Clim
            
            # Caculate correlation
            gridCorr[:,iMon+(iLag)*12][:,None], gridSign[:,iMon+(iLag)*12][:,None] = \
                corr2d1d(flow[:,np.in1d(dtFlow, dxFlow)], 
                    clim[np.in1d(dtClim, dxClim)], alpha)
            np.seterr(divide='ignore', invalid='ignore')
    
        if flagProc:
            print('Correlations are caculated (Mon:{:02}/12)'.format(iMon+1))
        
    
    # Maximum season-ahead correlation rank --------------------------------- #
    if flagProc:
        print('Identifying season-ahead maximum correlation and lead-time.')
    
    sheadMaxCorr = np.full([flow.shape[0],12],np.nan)
    sheadMaxLead = sheadMaxCorr.copy()
    sheadMaxSign = sheadMaxCorr.copy()
    sheadCorr = gridCorr[:,12*3::]      # Considering only lag3-maxLeadMonth
    sheadSign = gridSign[:,12*3::]      # Considering only lag3-maxLeadMonth
    
    for iMon in range(12):
        # Find season-ahead maximum correlation and its lead-time per each month
        # (*starts with lead-month to avoid to select absoute values used to 
        # select maximum values)
        sheadCorrTemp = sheadCorr[:,iMon::12]
        sheadSignTemp = sheadSign[:,iMon::12]
        maxlead = np.nanargmax(abs(sheadCorrTemp), axis=1)
        sheadMaxLead[:,iMon] = maxlead + 3
        sheadMaxCorr[:,iMon] = sheadCorrTemp[np.arange(flow.shape[0]), maxlead]
        sheadMaxSign[:,iMon] = sheadSignTemp[np.arange(flow.shape[0]), maxlead]
        
    return sheadMaxCorr, sheadMaxLead, sheadMaxSign


#%% corrClimPont
def corrClimPont(ista, mo3FlowMatAll, climMat, maxLeadMonth, alpha):
    '''calculate lag-correlations of station streamflow with climate index
    
    This function is called by "corrClimStat" for multiprocessing
    
    Parameters
    -----------
    ista: int
        index for multiprocessing iteration
    mo3FlowMatAll: ndarray
        station streamflow time-series in climatological form, 
        (sta, [sta_no, nyr, yr, 12 months])
    climMat: ndarray
        climate index time-series in climatological form, 
        (time, [year, 12 months])
    maxLeadMonth: int
        the number of lead months
    alpha: int
        confidence level of correlation, default is 0.05
        
    Returns:
    -------
    sheadMaxCorr: list
        maximum correlation value, [12 months]
    sheadMaxLead: list
        corresponding lead-time, [12 months]
    sheadMaxSign: list
        corresponding significance, [12 months]
    
    '''    
    # station list
    _, idx = np.unique(mo3FlowMatAll[:,0], return_index=True)
    staList = mo3FlowMatAll[np.sort(idx),0].astype(int)
    sta_no = staList[ista]
    
    # Currently, we exclude streamflow records before the 1+start year of 
    # climate index. For example, in case of ONI (starts 1870), streamflow 
    # before 1871 is excluded.
    fdx = (np.in1d(mo3FlowMatAll[:,0],sta_no)) & \
        (mo3FlowMatAll[:,2]>climMat[0,0])
    flow = mo3FlowMatAll[fdx,3::]
    yrFlow = mo3FlowMatAll[fdx,2].astype(int)
    dtFlow = np.arange(str(yrFlow[0])+'-01', str(yrFlow[-1]+1)+'-01', 
                       dtype='datetime64[M]')
    if dtFlow.shape[0] != flow.shape[0] * flow.shape[1]:
        raise RuntimeError('Time dimensions are not correct.')
    
    # Load climate index from 1 year prior to the first year of streamflow
    tempClimMat = climMat[np.in1d(climMat[:,0], np.arange(yrFlow[0]-1,
                                  yrFlow[-1]+1)),:]
    yrClim = tempClimMat[:,0].astype(int)
    nyrClim = len(yrClim) 
    dtClim = np.arange(str(yrClim[0])+'-01', str(yrClim[-1]+1)+'-01', 
                       dtype='datetime64[M]')
    
    
    # Log-transform, Detrend, Remove annual cycle ----------------------- #
    # Log-transformation
    flow[flow <= 0] = 0.001
    flow = np.log(flow)
    
    # Remove seasonal trends
    # *The long-term trends and annual cycles are eliminated while removing 
    #   seasonal trends
    # Streamflow
    for i in range(12):
        if sum(np.isnan(flow[:,i])) != len(yrFlow):
            flow[:,i] = detrend_nan(yrFlow, flow[:,i])
    flow = flow.flatten()
    # Climate driver
    clim = signal.detrend(tempClimMat[:,1::], axis=0, type='linear')
    clim = np.reshape(clim, [nyrClim*12, 1])
    
    
    # Computing lag-correlation between concurrent and maxLeadMonth ----- #
    # Initialize correlation array including concurrent month
    corrTable = np.full([maxLeadMonth+1, 12], np.nan)
    signTable = corrTable.copy()
    for iMon in range(12):
        for iLag in range(maxLeadMonth+1):
            # Define time-index
            dxFlow = dtFlow[iMon::12]                   # time-index for Flow
            subFlow = flow[np.in1d(dtFlow, dxFlow)]
            dxFlow = dxFlow[~np.isnan(subFlow)]         # exclude NaN values
            dxClim = dxFlow - np.timedelta64(iLag,'M')  # time-index for Clim
            
            # Caculate correlation
            corrTable[iLag, iMon], signTable[iLag, iMon] = \
                corr2d1d(flow[np.in1d(dtFlow, dxFlow)][:,None].T,
                clim[np.in1d(dtClim, dxClim)],alpha)
            np.seterr(divide='ignore', invalid='ignore')
    
    
    # Maximum season-ahead correlation rank ----------------------------- #
    # considering only lag3-maxLeadMonth
    # (*starts with lead-month to avoid to select absoute values used to select 
    #   maximum values)
    sheadMaxLead = np.argmax(abs(corrTable[3::,:]), axis=0) + 3
    sheadMaxCorr = corrTable[sheadMaxLead.astype(int), np.arange(12)]
    sheadMaxSign = signTable[sheadMaxLead.astype(int), np.arange(12)]
    
    return sheadMaxCorr, sheadMaxLead, sheadMaxSign


#%% corrClimStat
def corrClimStat(mo3FlowMatAll, climMat, maxLeadMonth, alpha=0.05, flagProc=True):
    '''calculate lag-correlations of station streamflow with climate index
    
    This function calls "corrClimPont" for multiprocessing.
    
    Parameters
    -----------
    mo3FlowMatAll: ndarray
        station streamflow time-series in climatological form, 
        (sta, [sta_no, nyr, yr, 12 months])
    climMat: ndarray
        climate index time-series in climatological form, 
        (time, [year, 12 months])
    maxLeadMonth: int
        the number of lead months
    alpha: int
        confidence level of correlation, default is 0.05
    flagProc: bool
        flag of printing process, default is True
        
    Returns
    -------
    sheadMaxCorr: ndarray
        maximum correlation value, [sta, 12 months]
    sheadMaxLead: ndarray
        corresponding lead-time, [sta, 12 months]
    sheadMaxSign: ndarray
        corresponding significance, [sta, 12 months]
    
    '''
# =============================================================================
#     # REVISION SETTING ------------------------------------------------------ #
#     import os
#     import numpy as np
#     from scipy import signal, stats
#     import gf
#     from multiprocessing import Pool
#     from functools import partial
#     import time
# # =============================================================================
# #     from tqdm import tqdm
# # =============================================================================
#     np.warnings.filterwarnings("ignore", category=RuntimeWarning)
#     maxLeadMonth = 8              # The maximum lead-month
#     alpha = 0.05
#     matdata = gf.loadmat(os.path.join('data', 'gsff.mat'))
#     gsffList = matdata['gsffList'].astype(int); ngsff = len(gsffList)
#     mo3FlowMatAll = matdata['mo3FlowMatAll']
#     cidx = gf.loadmat(os.path.join('data', 'lscidx.mat'))
#     cidx, _ = gf.matdToNtup(cidx)
#     climMat = cidx.nao
#     flagProc = 1
# =============================================================================
    
    # station list
    _, idx = np.unique(mo3FlowMatAll[:,0], return_index=True)
    staList = mo3FlowMatAll[np.sort(idx),0].astype(int)
    
    # initialize variables -------------------------------------------------- #
    nsta = len(staList)
    sheadMaxCorr = np.full([nsta, 12], np.nan)
    sheadMaxLead = sheadMaxCorr.copy()
    sheadMaxSign = sheadMaxCorr.copy()
    print('Multiprocessing is started..')
    start_time = time.time()
# =============================================================================
#     pbar = tqdm(total = nsta)
# =============================================================================
    
    # multiprocessing setting
    pool = Pool(processes=4)
    corrClimPont2=partial(corrClimPont, mo3FlowMatAll=mo3FlowMatAll, climMat=climMat, 
                          maxLeadMonth=maxLeadMonth, alpha=alpha)
    results = pool.map(corrClimPont2, range(nsta))
# =============================================================================
#     with tqdm(total=nsta) as t:
#         for _ in pool.imap_unordered(corrStat_x, range(nsta)):
#             t.update(1)
# =============================================================================
    pool.close()    
    pool.join()
# =============================================================================
#     pbar.close()
# =============================================================================
    # organizing output from multiprocessing
    for i in range(nsta):
        sheadMaxCorr[i,:] = results[i][0]
        sheadMaxLead[i,:] = results[i][1]
        sheadMaxSign[i,:] = results[i][2]
    
    print(time.strftime("Multiprocessing took %H:%M:%S.", 
                        time.gmtime(time.time() - start_time)))

    return sheadMaxCorr, sheadMaxLead, sheadMaxSign


#%% corrAutoGrid
def corrAutoGrid(tim, flow, alpha=0.05):
    '''calculate monthly auto-correlations of gridded streamflow
    
    This function calls "corr1d1d_wrapper" for multiprocessing.
    
    Parameters
    -----------
    tim: list
        years of the start and end, [year_start, year_end]
    flow: ndarray
        gridded streamflow time-series, [grid_index, 12 months]
    alpha: int
        confidence level of correlation, default is 0.05
    flagProc: bool
        flag of printing process, default is True
        
    Returns
    -------
    sheadAutoCorr: ndarray
        gridded shead auto-correlation, [grid_index, 12 months]
    sheadAutoSign: ndarray
        gridded significance of shead auto-correlation, [grid_index, 12 months]
    
    '''
# =============================================================================
#     # REVISION SETTING ------------------------------------------------------ #
#     import os
#     import numpy as np
#     from scipy import signal 
#     import gf
#     from multiprocessing import Pool
#     from functools import partial
#     tim = [1958, 2000]
#     alpha = 0.05
#     flow = np.load(os.path.join('data', 'flow.npy'))
#     flagProc = 1
# =============================================================================

    # Initialize variables -------------------------------------------------- #
    yrFlow = np.arange(tim[0],tim[1]+1)
    dtFlow = np.arange(str(yrFlow[0])+'-01', str(yrFlow[-1]+1)+'-01', 
                       dtype='datetime64[M]')
    if dtFlow.shape[0] != len(flow[0]):
        raise RuntimeError('Time dimensions are not correct.')
    np.seterr(divide='ignore', invalid='ignore')
    
    # Log-transform, Detrend, Remove annual cycle --------------------------- #
    # Log-transformation
    flow[flow <= 0] = 0.001
    flow = np.log(flow)
    
    # Remove seasonal trends
    # *The long-term trends and annual cycles are eliminated while removing 
    #   seasonal trends
    # Streamflow
    for i in range(12):    
        flow[:,i::12] = signal.detrend(flow[:,i::12], axis=-1, type='linear')
    
    # Computing lag-correlation between concurrent and maxLeadMonth --------- #
    print('Calculating grid-scale auto-correlation at each month.')
    
    # Initialize correlation array
    sheadAutoCorr = np.full([len(flow[:,0]), 12], np.nan)
    sheadAutoSign = np.zeros(sheadAutoCorr.shape, dtype=bool)
    for iMon in range(12):
        
        # Define time-index
        dxFlow1 = dtFlow[iMon::12]                  # Time-index for Flow1
        dxFlow0 = dxFlow1 - np.timedelta64(3,'M')   # 1 season (3-mon) ahead
        
        # Remove data prior to the start of flow data
        rdx = dxFlow0 < dtFlow[0]
        flow1 = flow[:,np.in1d(dtFlow, dxFlow1[~rdx])]  # (nrow, nmon)
        flow0 = flow[:,np.in1d(dtFlow, dxFlow0[~rdx])]  # (nrow, nmon)
                   
        # multiprocessing - calculate correlation at each grid
        pool = Pool(processes=4)

        # set each matching item into a tuple
        job_args = []
        for i in range(len(flow0)):
            job_args.append((flow0[i,:], flow1[i,:]))
        
        results = pool.map(corr1d1d_wrapper, job_args)
        pool.close()
        pool.join()
        
        # organizing output from multiprocessing
        for i in range(len(results)):
            sheadAutoCorr[i, iMon] = results[i][0]
            sheadAutoSign[i, iMon] = results[i][1]
        
        print('Auto-correlations are calculated (Mon:{:02}/12).'.format(iMon+1))
        
    return sheadAutoCorr, sheadAutoSign


#%% corrAutoPont
def corrAutoPont(ista, mo3FlowMatAll, alpha=0.05):
    '''Calculate monthly auto-correlation of station streamflow
    
    This function is called by "corrAutoStat" for multiprocessing
    
    Parameters:
    -----------
    ista: int
        index for multiprocessing iteration
    mo3FlowMatAll: ndarray
        station streamflow time-series in climatological form, 
        (sta, [sta_no, nyr, yr, 12 months])
    alpha: int
        confidence level of correlation, default is 0.05
        
    Returns:
    --------
    autoCorr: list
        auto-correlation value, [12 months]
    autoSign: ndarray
        corresponding significance, [12 months]
    
    '''
    
# =============================================================================
#     # revision setting ------------------------------------------------------ #
#     ista = 2
#     matdata = gf.loadmat(os.path.join('data', 'gsff.mat'))
#     gsffList = matdata['gsffList'].astype(int); ngsff = len(gsffList)
#     mo3FlowMatAll = matdata['mo3FlowMatAll']
#     alpha = 0.05    
# =============================================================================
    
    # station list
    _, idx = np.unique(mo3FlowMatAll[:,0], return_index=True)
    staList = mo3FlowMatAll[np.sort(idx),0].astype(int)
    
    # initialize variables -------------------------------------------------- #
    sta_no = staList[ista]
    fdx = (np.in1d(mo3FlowMatAll[:,0],sta_no))
    flow = mo3FlowMatAll[fdx,3::]
    yrFlow = mo3FlowMatAll[fdx,2].astype(int)
    dtFlow = np.arange(str(yrFlow[0])+'-01', str(yrFlow[-1]+1)+'-01', 
                       dtype='datetime64[M]')
    if dtFlow.shape[0] != flow.shape[0] * flow.shape[1]:
        raise RuntimeError('Time dimensions are not correct.')
    
    # log-transform, detrend, remove annual cycle --------------------------- #
    # log-transformation
    flow[flow <= 0] = 0.001
    flow = np.log(flow)
    
    # remove seasonal trends
    # *the long-term trends and annual cycles are eliminated while removing 
    #   seasonal trends
    for i in range(12):
        if sum(np.isnan(flow[:,i])) != len(yrFlow):
            flow[:,i] = detrend_nan(yrFlow, flow[:,i])
    flow = flow.flatten()
    
    # computing auto-correlation -------------------------------------------- #
    # initialize correlation array
    autoCorr = np.full(12, np.nan)
    autoSign = np.zeros(12, dtype=bool)
    for iMon in range(12):
        
        # define time-index
        dxFlow1 = dtFlow[iMon::12]                  # Time-index for Flow1
        dxFlow0 = dxFlow1 - np.timedelta64(3,'M')   # 1 season (3-mon) ahead
        
        # remove data prior to the start of flow data
        rdx = dxFlow0 < dtFlow[0]
        flow1 = flow[np.in1d(dtFlow, dxFlow1[~rdx])]
        flow0 = flow[np.in1d(dtFlow, dxFlow0[~rdx])]
        
        # auto-colleation
        autoCorr[iMon], autoSign[iMon] = corr1d1d_nan(flow0, flow1, alpha)
    
    return autoCorr, autoSign
    

#%% corrAutoStat
def corrAutoStat(mo3FlowMatAll, alpha=0.05):
    '''calculate monthly auto-correaltion of station streamflow
    
    This function calls "corrAutoPont" for multiprocessing.
    
    Parameters
    -----------
    mo3FlowMatAll: ndarray
        station streamflow time-series in climatological form, 
        (sta, [sta_no, nyr, yr, 12 months])
    alpha: int
        confidence level of correlation, default is 0.05
        
    Returns
    -------
    sheadAutoCorr: list
        auto-correlation value, [sta, 12 months]
    sheadAutoSign: ndarray
        corresponding significance, [sta, 12 months]
        
    '''
    
# =============================================================================
#     # REVISION SETTING ------------------------------------------------------ #
#     import os
#     import numpy as np
#     from scipy import signal, stats
#     import gf
#     from multiprocessing import Pool
#     from functools import partial
#     import time
#     np.warnings.filterwarnings("ignore", category=RuntimeWarning)
#     alpha = 0.05
#     matdata = gf.loadmat(os.path.join('data', 'gsff.mat'))
#     mo3FlowMatAll = matdata['mo3FlowMatAll']
# =============================================================================

    # station list
    _, idx = np.unique(mo3FlowMatAll[:,0], return_index=True)
    staList = mo3FlowMatAll[np.sort(idx),0].astype(int)
    
    # initialize variables -------------------------------------------------- #
    nsta = len(staList)
    sheadAutoCorr = np.full([nsta, 12], np.nan)
    sheadAutoSign = np.zeros([nsta, 12], dtype=bool)
    print('Multiprocessing is started..')
    start_time = time.time()
# =============================================================================
#     pbar = tqdm(total = nsta)
# =============================================================================
    
    # multiprocessing setting
    pool = Pool(processes=4)
    corrAutoPont2=partial(corrAutoPont, mo3FlowMatAll=mo3FlowMatAll)
    results = pool.map(corrAutoPont2, range(nsta))
# =============================================================================
#     with tqdm(total=nsta) as t:
#         for _ in pool.imap_unordered(corrStat_x, range(nsta)):
#             t.update(1)
# =============================================================================
    pool.close()    
    pool.join()
# =============================================================================
#     pbar.close()
# =============================================================================
    # organizing output from multiprocessing
    for i in range(nsta):
        sheadAutoCorr[i,:] = results[i][0]
        sheadAutoSign[i,:] = results[i][1]
    
    print(time.strftime("Multiprocessing took %H:%M:%S.", 
                        time.gmtime(time.time() - start_time)))

    return sheadAutoCorr, sheadAutoSign


