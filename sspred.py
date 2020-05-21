#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-
'''
    This script contains algorithms of season-ahead prediction using multiple
    predictors and one-season ahead autocorrelation.

    File name: sspred.py
    Date created: 08/01/2018
    Date last modified: 11/25/2018
'''
__version__ = "1.0"
__author__ = "Donghoon Lee"
__maintainer__ = "Donghoon Lee"
__email__ = "dlee298@wisc.edu"


#import os, sys
#from pathlib import Path
#import gf
#import matplotlib.pyplot as plt
from itertools import compress, product
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
pd.options.mode.chained_assignment = None
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, PredefinedSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import time


def treatMissPeriod(xPred, leadPred, targIndx):
    '''
    This function treats missing periods of predictors.
    For each lead-month of each predictor,
    - If n_missing < 33%, interpolate using climatology
    - If 33% <= n_missing, exclude that lead-month or predictor

    Parameters
    ----------
    xPred:          DataFrame of predictors
    leadPred:       List of ndarrays of lead-months
    targIndx:       Target PeriodIndex

    Returns
    ----------
    xPred:          Input xPred with treated data
    leadPred:       List of ndarrays of treated lead-months
    '''

    xPred = xPred.copy()
    leadPred = leadPred.copy()

    # Thresholds of the percentage of missing periods
    prctFill = 0.33

    for ip in range(xPred.shape[1]):
        # Per each predictor
        tPred = xPred[xPred.columns[ip]]
        mlist = leadPred[ip]
        mlistBool = np.full(mlist.shape, True)
        for im in range(len(mlist)):
            # Per each lead-month
            temp = tPred.reindex(targIndx - MonthEnd(mlist[im]))

            # Missing value contrl
            prctNull = np.sum(temp.isnull()) / len(targIndx)
            if (prctNull < prctFill):
                # If prctNull < 33%, interpolate using climatology
#                print(temp[temp.isnull()])
                temp[temp.isnull()] = temp.mean()
                # Re-assign interpolated data
                tPred.loc[targIndx - MonthEnd(mlist[im])] = temp
            else:
                # If 33% <= prctNull, exclude that lead-month
                mlistBool[im] = False

            # Monotonic value control
            if temp.is_monotonic:
                mlistBool[im] = False

        # Re-assign list of lead-months
        mlist = mlist[mlistBool]
        leadPred[ip] = mlist
        # Re-assign data
        xPred[xPred.columns[ip]] = tPred

    return xPred, leadPred


def _matlab_percentile(in_data, percentiles):
    """
    Calculate percentiles in the way IDL and Matlab do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    in_data: numpy.ndarray
        input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values

    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """
    data = np.sort(in_data)
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def _detrend(sr):
    '''detrends 1D Series

    *this function deals with missing values.

    Parameters
    ----------
    sr: Series
        time-series to be detrended

    Returns
    -------
    sr: Series
        detrended values of the time-series

    '''
    nni = ~sr.isnull()
    if False:
        # Considering NaN values
        x = np.array((sr.index - sr.index[0])/12 + 1, dtype='int')
        m, b, r_val, p_val, std_err = stats.linregress(x[nni], sr[nni])
        sr_detrend = sr - (m*x +b)
    else:
        # Ignoring NaN values
        from scipy import signal
        sr_detrend = sr.copy()
        sr_detrend[nni] = pd.Series(signal.detrend(sr[nni]), index = sr[nni].index)

    return sr_detrend


def _corr2d1d_nan(df, sr, alpha=0.05):
    '''
    Returns Pearson's correlation between Dataframe and Series.
    This function deals with missing (NaN) values.

    Parameters
    ----------
    df      - Dataframe (DateTimeIndex, Columns)
    sr      - Series (DateTimeIndex, Value)
    alpha   - significance level

    Returns
    -------
    corr    - 1D ndarray of Pearson's correlation
    sign    - 1D boolean array of Two-sided T-Test result (1:sign, 0:none)
    '''

    # missing values control
    nan_ind = np.array(pd.isna(sr)) | np.sum(np.array(df.isnull()), axis=1).astype('bool')
    x = df.loc[~nan_ind]
    y = sr.loc[~nan_ind]
    # dof
    n = len(y)
    dof = n - 1
    # zscore
    x = np.array(x.T)
    xz = (x - x.mean(1)[:,None])/np.std(x, ddof=1, axis=1)[:,None]
    yz = (y - y.mean(0))/np.std(y, ddof=1)
    # Correlation
    corr = xz.dot(yz)/dof

    # Two-sided T-test
    tstat = corr*np.sqrt(n-2)/np.sqrt(1-corr**2)
    sign = np.abs(tstat) > stats.t.ppf(1-alpha/2, n-2)

    return corr, sign


def _corr1d1d_nan(sr1, sr2, alpha=0.05):
    '''
    Returns Pearson's correlation between two Series
    This function deals with missing (NaN) values.

    Parameters
    ----------
    sr1     - Series (DateTimeIndex, Value)
    sr2     - Series (DateTimeIndex, Value)
    alpha   - significance level

    Returns
    -------
    corr    - 1D ndarray of Pearson's correlation
    sign    - 1D boolean array of Two-sided T-Test result (1:sign, 0:none)
    '''

    # missing values control
    nan_ind = np.array(pd.isna(sr1)) | np.array(pd.isna(sr2))
    x = sr1.loc[~nan_ind]
    y = sr2.loc[~nan_ind]
    # dof
    n = len(y)
    dof = n - 1
    # zscore
    x = np.array(x.T)
    xz = (x - x.mean())/np.std(x, ddof=1)
    yz = (y - y.mean())/np.std(y, ddof=1)
    # Correlation
    corr = xz.dot(yz)/dof

    # Two-sided T-test
    tstat = corr*np.sqrt(n-2)/np.sqrt(1-corr**2)
    sign = np.abs(tstat) > stats.t.ppf(1-alpha/2, n-2)

    return corr, sign


def _LeaveBlockOut(n,bl):
    """Leave-Block-out
    Parameters:
    ----------
    n:      number of sample
    bl:     block size to be excluded in each interation
    """
    
    # bl should be odd number
    if bl % 2 is 0: raise ValueError('bl should be an odd number')
    
    # PredefinedSplit.split() is replaced
    def split(self, X=None, y=None, groups=None):
        """Modified from the original PredefinedSplit.split()
        """
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]
            train_index = train_index[~np.in1d(train_index, 
                       np.arange(test_index-(bl-1)/2,test_index+(bl-1)/2+1))]
            yield train_index, test_index
    # Create test_fold
    test_fold = np.squeeze(np.hstack((np.ones([1, int((bl-1)/2)])*-1, 
                  np.arange(1,n-bl+2)[None,:], np.ones([1, int((bl-1)/2)])*-1)))
    # Replace PredefinedSplit.split() to split()
    PredefinedSplit.split = split
    return PredefinedSplit(test_fold)


def _pcr(xTran, yTran, npc):
    '''
    Returns PCR regression object.
    *please see "note_pcr.py" for more details

    Parameters
    ----------
    xTran: ndarray
        ndarray to be used to transformed
    npc: int
        number of principal components to be used to regression
    Returnes
    --------
    regr: regression object
        linear regression module with transformed coefficient
    '''
    pca = PCA()
    Z = pca.fit_transform(xTran)
    A = pca.components_
    regr = LinearRegression()
    regr.fit(Z[:,:npc], yTran)
    gamma = regr.coef_
    beta = np.dot(A[:npc,:].T, gamma.flatten())

    # regression object to be returned
    regr = LinearRegression(fit_intercept=False)
    regr.fit(xTran, np.dot(xTran, beta))
    regr.coef_ = beta

    return regr


def _sscore(yClim, yObsd, yFcst):
    '''
    Returns skill scores for "test" period
    *yClim means yObsd during "train" period
    '''


    def _gss(yClim, yObsd, yFcst):
        # Gerrity skill score (GSS)
# =============================================================================
#         # revision setting -------------------------------------------------- #
#         yClim, yObsd, yFcst = yTran, yTest, yTestHat
# =============================================================================
        # - classify forecasts into 3x3 ANB matrix
        n = len(yFcst)
        obsdFcst = np.vstack((yObsd, yFcst)).T
#        terc = np.percentile(yClim, np.array([1/3, 2/3])*100)
        terc = _matlab_percentile(yClim, np.array([1/3, 2/3])*100)
        idx01 = (obsdFcst <= terc[0])
        idx02 = (terc[0] < obsdFcst) & (obsdFcst <= terc[1])
        idx03 = (terc[1] < obsdFcst)
        obsdFcst[idx01] = 0; obsdFcst[idx02] = 1; obsdFcst[idx03] = 2
        data = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                data[i,j] = np.sum( (obsdFcst[:,0] == j) & (obsdFcst[:,1] == i) )
        no = 3
        joinDist = data/n
        margDist = np.sum(data,0)/n

        # control of category with zero count
        if np.sum(margDist == 0) > 0:
            nzero = np.sum(margDist == 0)
            margDist[margDist == 0] = 0.001
            margDist[np.argmax(margDist)] = margDist[np.argmax(margDist)] - 0.001*nzero
# =============================================================================
#         # revision setting -------------------------------------------------- #
#         temp = gf.loadmat('./data/table_8.3.mat')
#         n = temp['n']
#         data = temp['data']
#         no = data.shape[0]
#         joinDist = data/n
#         margDist = np.sum(data, 0)/n
# =============================================================================
        # J-1 odds ratios
        d = np.cumsum(margDist[:-1])
        d = (1-d)/d
        invd = d**-1
        # scoring weights
        scoreWeight = np.zeros(joinDist.shape)
        for j in range(no):
            # correct forecasts
            if j > 0:
                scoreWeight[j,j] = scoreWeight[j,j] + np.sum(invd[:j])
            if j < no-1:
                scoreWeight[j,j] = scoreWeight[j,j] + np.sum(d[j:no-1])
            # incorrect forecasts
            for i in range(no):
                if i < j:
                    if i > 0:
                        scoreWeight[i,j] = scoreWeight[i,j] + np.sum(invd[:i])
                    if j < no-1:
                        scoreWeight[i,j] = scoreWeight[i,j] + np.sum(d[j:no-1])
                    scoreWeight[i,j] = scoreWeight[i,j] - (j-i)

        scoreWeight = scoreWeight + np.triu(scoreWeight, 1).T
        scoreWeight = scoreWeight/(no-1)

        # Gandin-Murphy Skill Scores (GMSS)
        gmss = np.sum(joinDist*scoreWeight)

        return gmss


    def _msess(yClim, yObsd, yFcst):
       # Mean Squared Error Skill Score (MSESS) (%)
       # MSESS = (1 - MSE_pred/MSE_clim)*100
       #    MSE_pred = 1/n*sum((y-yHat).^2)
       #    MSE_clim = 1/n*sum((y-yMean).^2)
       #
       mse_pred = mean_squared_error(yObsd, yFcst)
       mse_clim = mean_squared_error(yObsd, np.ones(yObsd.shape)*yClim.mean())
       msess = (1 - mse_pred/mse_clim)*100

       return msess

    gss = _gss(yClim, yObsd, yFcst)
    msess = _msess(yClim, yObsd, yFcst)

    return gss, msess


def predict(iargs):
    '''
    Predicts seasonal streamflow using climate drivers and auto-correlation
    at single point (grid or stat).

    1) Calculates correlations between seasonal streamflow and large-scale
    climate drivers. Only significantly correlated variables are considered as
    potential predictors. One season-ahead autocorrelation is also calculated.

    2) If a single predictor exists, it applies the Linear Regression (LR).
    If multiple predictors exist, it applies the Principial Component
    Regression (PCR). If climate driver(s) is exist, the Leave-one-out Cross
    Validation (LOOCV) is applied to find optimal lead-months of climate
    drivers.
    - (A) single autocorr predictor:    LR model
    - (B) single climate predictor:     LR model with LOOCV
    - (C) multiple predictors:          PCR model with LOOCV

    Parameters(iargs)
    ----------
    dfFlow  - Pandas Series (DateTimeIndex, Value)
    dfPred  - Pandas DataFrame (DateTimeIndex, Values)

        Lead-predictors are supposed to have no missing data

    sta_no  - Station number (int)
    idx     - Index (int)

    Returns
    -------
    ybox    - Dictionary consists of results for 12 months


    Status
    -------
    100     - Flow data is less than 10 years
    200     - Flow data is monotonic (possible missing records)
    300     - No significantly correlated predictors

    '''


# =============================================================================
#     #%% Revision setting ---------------------------------------------------- #
#     # Load data
#     # Predictors
#     dfPred = pd.read_pickle('dfPred.data')
#     if True:
#         # Target point
#         point_no = 4118410
#         # Streamflow data (Station)
#         dfFlow = pd.read_pickle('dfFlowStat.data')
#         dfFlow = dfFlow[point_no]
#         dfPred['flow'] = dfFlow
#         # PeakMonth (Station)
#         pmpoint = np.load('pmStat.npz')['pmpoint']
#         tarMonth = pmpoint[np.in1d(pmpoint[:,0], point_no),1][0]
#
#     else:
#         # Target point
#         point_no = 1
#         # Streamflow data (Station)
#         dfFlow = pd.read_pickle('dfFlowGrid.data')
#         dfFlow = dfFlow[point_no]
#         dfPred['flow'] = dfFlow
#         # PeakMonth (Station)
#         pmpoint = np.load('pmGrid.npz')['pmpoint']
#         tarMonth = pmpoint[np.in1d(pmpoint[:,0], point_no),1][0]
#
#     # Change input data structure
#     maxLead = np.array([8,8,8,8,3])
#     iargs = (dfFlow, dfPred, maxLead, tarMonth, point_no)
#     # End of revision setting ----------------------------------------------- #
#     #%%
# =============================================================================

    #TODO: ERROR CODE LISTING


    stim = time.time()
    # Input variable control ------------------------------------------------ #
    dfFlow, dfPred, leadMat, tarMonth, point_no = iargs
    assert dfPred.shape[1] == leadMat.shape[1]

    # Parameters
    prct = 0.3              # percentage of test period
    ybox = dict()

    # Flow data control
    # Remove missing years at the front/back of the flow data
    # *This is for tailoring predictors periods. Actual missing value control
    #   will be applied at monthly running.
    alive = dfFlow[~dfFlow.isnull()].index.year
    index = dfFlow.index
    dfFlow = dfFlow.reindex(index[(index.year >= alive.min()) & (index.year <= alive.max())])
    # Flow starts from 1901
    fcon1 = dfFlow.index.year >= 1901
    dfFlow = dfFlow[fcon1]
    # Log-transformation
    # *Temporarily, flow less than or equal to 0 is converted to 0.01
    dfFlow[dfFlow <= 0] = 0.001
    dfFlow = np.log(dfFlow)
    # If "flow" exists in dfPred, replace it to "dfFlow"
    if bool(np.sum(dfPred.columns == 'flow')):
       dfTemp = dfPred['flow'].copy()
       dfTemp[dfTemp <= 0] = 0.01
       dfPred['flow'] = np.log(dfTemp)

    # Predictor data control
    # Predictor starts 1 year earlier than flow data
    pcon1 = dfPred.index >= (dfFlow.index[0] - MonthEnd(12))
    # Predictor ends at the end of flow data
    pcon2 = dfPred.index <= dfFlow.index[-1]
    # Predictor ends at 2015-12
    pcon3 = dfPred.index <= '2015-12'
    dfPred = dfPred[pcon1 & pcon2 & pcon3]

    #TODO: UPDATE MONTH FROM 1 TO 12
    # Target Month
    if tarMonth == 13:
        tarRange = range(12)
    else:
        tarRange = range(tarMonth-1,tarMonth)
            
    # Main algorithms
    for i in tarRange:
        # Initialize the monthly box
        mbox = {'point_no': point_no, 'status': 0, 'month': i+1}
        ybox.update({'m%02d'%(i+1): mbox})
        xPred = dfPred.copy()
        nPred = xPred.shape[1]
        strPredOrig = list(xPred.columns)
        strRemoved = []

        # Flow data control
        # Select flow in the current month
        ipf = dfFlow.index[dfFlow.index.month == i+1]   # Initial period of flow
        y = dfFlow.copy()
        y = y.reindex(ipf)                              # Original flow
        y = y.drop(y[y.isnull()].index)                 # Drop missing years
        # STATUS: If number of years less than 10
        if len(y) < 10:
            mbox['status'] = 110                        # STATUS CODE
            ybox.update({'m%02d'%(i+1): mbox})
            continue
        # STATUS: If all values are the same (possible missing records)
        if y.is_monotonic:
            mbox['status'] = 120                        # STATUS CODE
            ybox.update({'m%02d'%(i+1): mbox})
            continue
        # Detrending flow
        yOrign = y.copy()
        y = _detrend(y) + y.mean()                      # Detrended flow
        yTrend = yOrign - y                             # Saved trend 

        # Predictor data control
        # Lead-months of potential predictors
        leadPred = []
        for ip in range(nPred):
            leadPred.append(np.arange(leadMat[1,ip],leadMat[0,ip]+1))
        # Missing period control (treats only possible lead-months)
        xPred, leadPred = treatMissPeriod(xPred, leadPred, y.index)
        # Refine predictor data (naming with "drop")
        nolead = np.in1d(list(map(np.size, leadPred)), 0)
        strRemoved = list(compress(strPredOrig, nolead))
        leadPredDrop = list(compress(leadPred, ~nolead))
        xPredDrop = xPred.drop(columns = strRemoved)
        strDrop = list(xPredDrop.columns)
        nPredDrop = len(strDrop)


        ## 1) Lag-correlations ---------------------------------------------- #
        # This considers only remained lead-months
        maxLeadPred = list(map(np.max, leadPredDrop))
        corr = np.full([np.max(maxLeadPred)-2, nPredDrop], np.nan)
        sign = corr.copy()
        for j in range(nPredDrop):
            # Dataframe of each predictor
            jPred = xPredDrop[strDrop[j]]
            jLead = leadPredDrop[j]
            for k in jLead:
                # Dataframe of each lead-month of each predictor
                kPred = jPred.reindex(y.index - MonthEnd(k))        # Original
                kPred_detrend = _detrend(kPred) + kPred.mean(0)     # Detrended
                # Correlation and significance
                corr[k-3,j], sign[k-3,j] = _corr1d1d_nan(kPred_detrend, y)

        lead = np.nanargmax(np.abs(corr), axis=0)
        maxcorr = corr[lead, np.arange(corr.shape[1])]
        maxsign = sign[lead, np.arange(corr.shape[1])].astype('bool')
        maxlead = lead + 3
        # Update monthly box
        mbox.update({'maxPred': list(strDrop),'maxcorr': maxcorr, 'maxsign': maxsign,
                     'maxlead': maxlead, 'strRemoved': strRemoved})

        # Post-processing (Renaming)
        # *Only significantly correlrated predictors are used to predict
        strPred = list(compress(strDrop, maxsign))
        xPred = xPredDrop[strPred]
        nPred = xPred.shape[1]
        leadPred = list(compress(leadPredDrop, maxsign))
        xLeadCorr = maxlead[maxsign]
        # Identify auto- and lead-predictors
        isAuto = np.in1d(strPredOrig, strPred)
        isAuto = ((leadMat[0,isAuto] - leadMat[1,isAuto]) == 0)
        nAuto = np.sum(isAuto)
        nLead = np.sum(~isAuto)
        strAuto = list(compress(strPred, isAuto))
        strLead = list(compress(strPred, ~isAuto))
        # STATUS: If no potential predictors
        if nPred == 0:
            mbox['status'] = 300                        # STATUS CODE
            ybox.update({'m%02d'%(i+1): mbox})
            continue
        # Update monthly box
        mbox.update({'strPred': strPred, 'nPred': nPred, 'xLeadCorr': list(xLeadCorr),
                     'strAuto': strAuto, 'strLead': strLead, 'nAuto': nAuto,
                     'nLead': nLead})

        ## 2) Prediction ---------------------------------------------------- #
        # (A) Single-lead(s)  (LR or PCR)
        # (B) Multi-leads     (LR or PCR with LOOCV)

        # Number of lead months of predictors
        nleadPred = np.array(list(map(np.size, leadPred)))

        # (A) Single-lead(s)  (LR or PCR)
        if (nPred > 0) & (nPred == np.sum(nleadPred)):
            # Split data to train/test time period
            xTemp = xPred.reindex(y.index - MonthEnd(k))
            xTran, xTest, yTran, yTest = train_test_split(xTemp, y, test_size=prct,
                                                          shuffle=False)
            # Nomalize before prediction
            scale_x = StandardScaler().fit(xTran)
            scale_x.scale_ = np.std(xTran.values, axis=0, ddof=1)  # sample STDEV
            xTran = scale_x.transform(xTran)
            xTest = scale_x.transform(xTest)
            scale_y = StandardScaler().fit(yTran[:,None])
            scale_y.scale_ = np.std(yTran, axis=0, ddof=1)  # sample STDEV
            yTran = scale_y.transform(yTran[:,None])
            yTest = scale_y.transform(yTest[:,None])

            # Regression
            if nPred == 1:
                # Single predictor (LR)
                regr = LinearRegression()
                regr.fit(xTran, yTran)
            else:
                # Multiple predictors (PCR)
                # *Currently, only the last PC is truncated
                regr = _pcr(xTran, yTran, xTran.shape[1]-1)
            yTranHat = regr.predict(xTran)
            yTestHat = regr.predict(xTest)


        # (B) Multi-leads (LR or PCR with LOOCV)
        if (nPred > 0) & (nPred <= np.sum(nleadPred)):
            # Define combinations of lead months of predictors
            comb = list(product(*leadPred))
            ncomb = len(comb)           # Number of combinations

            # Find optimal lead-times --------------------------------------- #
            mse = np.zeros(ncomb)
            for ic in np.arange(ncomb):
                # Load predictors in the current combination
                # "ndarray" is converted to "DataFrame"
                xTemp = np.zeros([len(y), nPred])
                for ip in range(nPred):
                    temp = xPred[strPred[ip]].reindex(y.index - MonthEnd(comb[ic][ip]))
                    # Detrending
                    xTemp[:,ip] = _detrend(temp) + temp.mean()
                xComb = pd.DataFrame(data=xTemp, index=y.index, columns=strPred)

                # split training and test data sets
                xTran, xTest, yTran, yTest = train_test_split(xComb, y,
                                              test_size=prct, shuffle=False)

# =============================================================================
#                 # Automatic Cross-Validation -------------------------------- #
#                 # Nomalize before prediction
# #                scale_x = StandardScaler().fit(xTran)
# #                scale_x.scale_ = np.std(xTran.values, axis=0, ddof=1)  # Sample STDEV
# #                xTran = scale_x.transform(xTran)
#                 scale_y = StandardScaler().fit(yTran[:,None])
#                 scale_y.scale_ = np.std(yTran, axis=0, ddof=1)  # Sample STDEV
#                 yTran = scale_y.transform(yTran[:,None])
# 
#                 # Leave-One-Out Cross Validation (LOOCV)
#                 lbo = _LeaveBlockOut(n=len(xTran), bl=1)        # bl = block size
#                 #TODO: REPLACE "f_normalize" to perform stadardization
#                 #TODO: Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#                 if nPred == 1:
#                     # (B) Single predictor (LR with LOOCV)
#                     regr = LinearRegression()
#                     regr.fit(xTran, yTran)
#                     regr.normalize, regr.fit_intercept = True, True
#                     yPred = cross_val_predict(regr, xTran, yTran, cv=lbo)
#                 else:
#                     # (C) Multiple predictors (PCR with LOOCV)
#                     # *Currently, only the last PC is truncated
#                     regr = _pcr(xTran, yTran, xTran.shape[1]-1)
#                     regr.normalize, regr.fit_intercept = True, True
#                     yPred = cross_val_predict(regr, xTran, yTran, cv=lbo)
# 
#                 # Evaluating statiscis: MSE
#                 #TODO: REDUCE THE LENGTH OF yTRAN and yPRED
#                 yTran = scale_y.inverse_transform(yTran).flatten()
#                 yPred = scale_y.inverse_transform(yPred).flatten()
#                 mse[ic] = mean_squared_error(yTran, yPred)
# =============================================================================


                # Manual Cross-Validation ----------------------------------- #
                nyTran = len(yTran)
                bl = 1                                  # Block size
                # Sample index out of block
                sampIdx = np.full([nyTran, nyTran-bl+1], False, dtype=bool)
                yPred = np.zeros([nyTran-bl+1, 1])
                for ip in range(nyTran-bl+1):
                    # Regression for each point (ip) to be cross-validated
                    sampIdx[:,ip] = ~np.in1d(np.arange(0,nyTran), np.arange(ip,ip+bl))

                    # Nomalize before prediction
                    yTemp = yTran.loc[sampIdx[:,ip]]
                    scale_y = StandardScaler().fit(yTemp[:,None])
                    scale_y.scale_ = np.std(yTemp, axis=0, ddof=1)          # Sample STDEV
                    yTemp = scale_y.transform(yTemp[:,None])
                    yTempVald = yTran.iloc[int((bl-1)/2+ip)]
                    yTempVald = scale_y.transform(yTempVald)
                    xTemp = xTran.loc[sampIdx[:,ip]]
                    scale_x = StandardScaler().fit(xTemp)
                    scale_x.scale_ = np.std(xTemp.values, axis=0, ddof=1)   # Sample STDEV
                    xTemp = scale_x.transform(xTemp)
                    xTempVald = xTran.iloc[int((bl-1)/2+ip)].values
                    xTempVald = scale_x.transform(xTempVald.reshape(1,-1)).flatten()

                    if nPred == 1:
                        # (B) Single predictor (LR with LOOCV)
                        regr = LinearRegression()
                        regr.fit(xTemp, yTemp)
                    else:
                        # (C) Multiple predictors (PCR with LOOCV)
                        # *Currently, only the last PC is truncated
                        regr = _pcr(xTemp, yTemp, xTemp.shape[1]-1)
                    yPredTemp = regr.predict(xTempVald.reshape(1,-1))
                    yPred[ip] = scale_y.inverse_transform(yPredTemp)

                # Evaluating statiscis: MSE
                mse[ic] = mean_squared_error(yTran[int((bl-1)/2):int(nyTran-(bl-1)/2)], yPred.flatten())


            # Optimal lead-time is decided by the minimum MSE
            xLeadOptm = comb[np.argmin(mse)]

            # Regression with optimal lead-time ----------------------------- #
            # Load predictors in the current combination
            # "ndarray" is converted to "DataFrame"
            xTemp = np.zeros([len(y), nPred])
            for ip in range(nPred):
                temp = xPred[strPred[ip]].reindex(y.index - MonthEnd(xLeadOptm[ip]))
                # Detrending
                xTemp[:,ip] = _detrend(temp) + temp.mean()
            xComb = pd.DataFrame(data=xTemp, index=y.index, columns=strPred)

            # Split training and test data sets
            xTran, xTest, yTran, yTest, yTrendTran, yTrendTest = train_test_split(xComb, y, yTrend,
                                          test_size=prct, shuffle=False)
            # Nomalize before prediction
            scale_x = StandardScaler().fit(xTran)
            scale_x.scale_ = np.std(xTran.values, axis=0, ddof=1)   # Sample STDEV
            xTran = scale_x.transform(xTran)
            xTest = scale_x.transform(xTest)
            scale_y = StandardScaler().fit(yTran[:,None])
            scale_y.scale_ = np.std(yTran, axis=0, ddof=1)          # Sample STDEV
            yTran = scale_y.transform(yTran[:,None])
            yTest = scale_y.transform(yTest[:,None])

            # Regression
            if nPred == 1:
                # (B) single climate predictor (LR)
                regr = LinearRegression()
                regr.fit(xTran, yTran)
            else:
                # (C) multiple predictors (PCR)
                # *currently, only the last PC is truncated
                regr = _pcr(xTran, yTran, xTran.shape[1]-1)
            yTranHat = regr.predict(xTran)
            yTestHat = regr.predict(xTest)


        # Post-processing
        if nPred > 0:
            # Re-scaling
            xTran = scale_x.inverse_transform(xTran)
            xTest = scale_x.inverse_transform(xTest)
            yTran = scale_y.inverse_transform(yTran).flatten()
            yTest = scale_y.inverse_transform(yTest).flatten()
            yTranHat = scale_y.inverse_transform(yTranHat).flatten()
            yTestHat = scale_y.inverse_transform(yTestHat).flatten()
            
            # Re-trending
            yTran = yTran + yTrendTran.values
            yTest = yTest + yTrendTest.values
            yTranHat = yTranHat + yTrendTran.values
            yTestHat = yTestHat + yTrendTest.values
            
            # Evaluating statiscis: gss, msess
            gss, msess = _sscore(yTran, yTest, yTestHat)

            # Update monthly box
            mbox.update({
                   'xLeadOptm': list(xLeadOptm), 'regr': regr,
                   'xTran': xTran, 'xTest': xTest,
                   'yTran': yTran, 'yTest': yTest,
                   'yTranHat': yTranHat, 'yTestHat': yTestHat,
                   'gss':gss, 'msess':msess
                   })

        # Update box
        ybox.update({'m%02d'%(i+1): mbox})

    etim = time.time() - stim
    print('point_no: %d took %.2fs' % (point_no, etim))


    #%%
    return ybox
















