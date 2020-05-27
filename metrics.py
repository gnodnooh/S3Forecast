"""
The metrics module contains functions to evaluate perfomances of forecast.
Currently, the following scores are available:

    Binary (2x2) contingency table
    - Probability of Detection (pod)
    - Frequency of Hits (foh)
    - False Alarm Ratio (far)
    - Probability of False Detection (pofd)
    - Frequency of Misses (fom)
    - Detection Failure Ratio (dfr)
    - Probability of Null (pon)
    - Frequency of Correct Null (focn)
    - Frequency Bias (bias)
    - Finley's measure, fraction correct, accuracy (accuracy)
    - Gilbert's Score or Threat Score or Critical Success Index (csi)
    - Equitable Threat Score, Gilbert Skill Score (ets)
    - Doolittle (Heidke) Skill Score (hss)
    - Peirce (Hansen-Kuipers, True) Skill Score (pss)
    - Clayton Skill Score (css)
    
    Multicategorical (e.g. 3x3) contingency table
    - Heidke Skill Score (hss)
    - Peirce Skill Score (pss)
    - Gerrity Skill Score (gss)

Source: https://github.com/djgagne/hagelslag
Revised by Donghoon Lee @ Jul-9-2019
dlee298@wisc.edu
"""

from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import gmean, rankdata
from sklearn.metrics import mean_squared_error
import warnings


# References for hydrometeological Skill scores and Performace indicators
# verif:            https://github.com/WFRT/verif
# HydroErr:         https://github.com/BYU-Hydroinformatics/HydroErr
# Hydrostats:       https://github.com/BYU-Hydroinformatics/Hydrostats
# properscoring:    https://github.com/TheClimateCorporation/properscoring
# Hagelslag:        https://github.com/djgagne/hagelslag

# See 7th International Verification Methods Workshop
# https://www.cawcr.gov.au/projects/verification/


def matlab_percentile(in_data, percentiles):
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


def makeBinaryContTable(obs, sim, clm=None, thrsd=0.5):
    """
    Initializes a binary contingency table with thrsd in percentage.
    
    Parameters
    ----------
    obs: 1d ndarray
        observed data
    sim: 1d ndarray
        forecast data
    clm: 1d ndarray
        climatological data (usually observation data in training period) to
        define ranges of contingency table and evaluate performances in testing
        period
    thrsd: float
        thresholds in percentage. Default is 0.5
        
    Returns
    -------
    table: 2d ndarray
        2x2 contingency table
        
    Donghoon Lee @ Jul-11-2019
    dlee298@wisc.edu
    """

    # Define a boundary
    assert len(obs) == len(sim), 'Lengths of arrays are different.'
    if clm is not None:
        # Use clm to make terciles
        terc = matlab_percentile(clm, thrsd*100)
    else:
        # Use obs to make terciles
        terc = matlab_percentile(obs, thrsd*100)
    
    # Classify data into ANB table
    combined = np.vstack((obs, sim)).T
    idx01 = (combined <= terc)
    idx02 = (combined > terc)
    combined[idx01] = 0; combined[idx02] = 1
    table = np.zeros([2,2])
    for (i,j) in np.ndindex(*table.shape):
        table[i,j] = np.sum( (combined[:,0] == j) & (combined[:,1] == i) )
    return table


def makeMultiContTable(obs, sim, clm=None, thrsd=[1/3, 2/3]):
    """
    Initializes a multiclass contingency table. Currently, it generates 3x3
    contingency table. Default setting makes 3x3 tercile table which is 
    Above-normal, Near-normal, Below-normal (ANB) contingency table from 
    observed and forecast data.
    
    Parameters
    ----------
    obs: 1d ndarray
        observed data
    sim: 1d ndarray
        forecast data
    clm: 1d ndarray
        climatological data (usually observation data in training period) to
        define ranges of contingency table and evaluate performances in testing
        period
    thrsd: list
        thresholds in percentages. Default is terciles (e.g. [1/3, 2/3])
        
    Returns
    -------
    table: 2d ndarray
        3x3 contingency table
        
    Donghoon Lee @ Jul-11-2019
    dlee298@wisc.edu
    """

    # Define boundaries
    assert len(obs) == len(sim), 'Lengths of arrays are different.'
    if clm is not None:
        # Use clm to make terciles
        terc = matlab_percentile(clm, np.array(thrsd)*100)
    else:
        # Use obs to make terciles
        terc = matlab_percentile(obs, np.array(thrsd)*100)
    
    # Classify data into ANB table
    combined = np.vstack((obs, sim)).T
    idx01 = (combined <= terc[0])
    idx02 = (terc[0] < combined) & (combined <= terc[1])
    idx03 = (terc[1] < combined)
    combined[idx01] = 0; combined[idx02] = 1; combined[idx03] = 2
    table = np.zeros([3,3])
    for (i,j) in np.ndindex(*table.shape):
        table[i,j] = np.sum( (combined[:,0] == j) & (combined[:,1] == i) )
    return table


class ContingencyTable(object):
    """
    Initializes a binary contingency table and generates many skill scores.
    The 2x2 Contingency table is represented as:

                    Observed
                    Yes   No
    Forecast Yes    a     b
             No     c     d
    
    Currently, the following scores are available:
        - Probability of Detection (pod)
        - Frequency of Hits (foh)
        - False Alarm Ratio (far)
        - Probability of False Detection (pofd)
        - Frequency of Misses (fom)
        - Detection Failure Ratio (dfr)
        - Probability of Null (pon)
        - Frequency of Correct Null (focn)
        - Frequency Bias (bias)
        - Finley's measure, fraction correct, accuracy (accuracy)
        - Gilbert's Score or Threat Score or Critical Success Index (csi)
        - Equitable Threat Score, Gilbert Skill Score (ets)
        - Doolittle (Heidke) Skill Score (hss)
        - Peirce (Hansen-Kuipers, True) Skill Score (pss)
        - Clayton Skill Score (css)
        
    
    Args:
        table: 2d ndarray ([[a, b],[c, d]])

    Attributes:
        table (numpy.ndarray): contingency table
        N: total number of items in table
        
        
    Source: https://github.com/djgagne/hagelslag
    Revised by Donghoon Lee @ Jul-10-2019
    """
    def __init__(self, table=None, N=None):
        self.table = table
        self.N = self.table.sum()

    def update(self, a, b, c, d):
        """
        Update contingency table with new values without creating a new object.
        """
        self.table.ravel()[:] = [a, b, c, d]
        self.N = self.table.sum()

    def __add__(self, other):
        """
        Add two contingency tables together and return a combined one.

        Args:
            other: Another contingency table

        Returns:
            Sum of contingency tables
        """
        sum_ct = ContingencyTable(*(self.table + other.table).tolist())
        return sum_ct

    def pod(self):
        """
        Probability of Detection (POD) or Hit Rate.
        Formula:  a/(a+c)
        """
        return self.table[0, 0] / (self.table[0, 0] + self.table[1, 0])

    def foh(self):
        """
        Frequency of Hits (FOH) or Success Ratio.
        Formula:  a/(a+b)
        """
        return self.table[0, 0] / (self.table[0, 0] + self.table[0, 1])

    def far(self):
        """
        False Alarm Ratio (FAR).
        Formula:  b/(a+b)
        """
        return self.table[0, 1] / (self.table[0, 0] + self.table[0, 1])

    def pofd(self):
        """
        Probability of False Detection (POFD).
        b/(b+d)
        """
        return self.table[0, 1] / (self.table[0, 1] + self.table[1, 1])

    def fom(self):
        """
        Frequency of Misses (FOM).
        Formula:  c/(a+c)."""
        return self.table[1, 0] / (self.table[0, 0] + self.table[1, 0])

    def dfr(self):
        """Returns Detection Failure Ratio (DFR).
           Formula:  c/(c+d)"""
        return self.table[1, 0] / (self.table[1, 0] + self.table[1, 1])

    def pon(self):
        """Returns Probability of Null (PON).
           Formula:  d/(b+d)"""
        return self.table[1, 1] / (self.table[0, 1] + self.table[1, 1])

    def focn(self):
        """Returns Frequency of Correct Null (FOCN).
           Formula:  d/(c+d)"""
        return self.table[1, 1] / (self.table[1, 0] + self.table[1, 1])

    def bias(self):
        """
        Frequency Bias.
        Formula:  (a+b)/(a+c)"""
        return (self.table[0, 0] + self.table[0, 1]) / (self.table[0, 0] + self.table[1, 0])

    def accuracy(self):
        """Finley's measure, fraction correct, accuracy (a+d)/N"""
        return (self.table[0, 0] + self.table[1, 1]) / self.N

    def csi(self):
        """Gilbert's Score or Threat Score or Critical Success Index a/(a+b+c)"""
        return self.table[0, 0] / (self.table[0, 0] + self.table[0, 1] + self.table[1, 0])

    def ets(self):
        """Equitable Threat Score, Gilbert Skill Score, v, (a - R)/(a + b + c - R), R=(a+b)(a+c)/N"""
        r = (self.table[0, 0] + self.table[0, 1]) * (self.table[0, 0] + self.table[1, 0]) / self.N
        return (self.table[0, 0] - r) / (self.table[0, 0] + self.table[0, 1] + self.table[1, 0] - r)

    def hss(self):
        """Doolittle (Heidke) Skill Score.  2(ad-bc)/((a+b)(b+d) + (a+c)(c+d))"""
        return 2 * (self.table[0, 0] * self.table[1, 1] - self.table[0, 1] * self.table[1, 0]) / (
            (self.table[0, 0] + self.table[0, 1]) * (self.table[0, 1] + self.table[1, 1]) +
            (self.table[0, 0] + self.table[1, 0]) * (self.table[1, 0] + self.table[1, 1]))

    def pss(self):
        """Peirce (Hansen-Kuipers, True) Skill Score (ad - bc)/((a+c)(b+d))"""
        return (self.table[0, 0] * self.table[1, 1] - self.table[0, 1] * self.table[1, 0]) / \
               ((self.table[0, 0] + self.table[1, 0]) * (self.table[0, 1] + self.table[1, 1]))

    def css(self):
        """Clayton Skill Score (ad - bc)/((a+b)(c+d))"""
        return (self.table[0, 0] * self.table[1, 1] - self.table[0, 1] * self.table[1, 0]) / \
               ((self.table[0, 0] + self.table[0, 1]) * (self.table[1, 0] + self.table[1, 1]))

    def __str__(self):
        table_string = '\tEvent\n\tYes\tNo\nYes\t%d\t%d\nNo\t%d\t%d\n' % (
            self.table[0, 0], self.table[0, 1], self.table[1, 0], self.table[1, 1])
        return table_string



class MulticlassContingencyTable(object):
    """
    This class is a container for a contingency table containing more than 2 
    classes. The contingency table is stored in table as a numpy array with the 
    rows corresponding to forecast categories, and the columns corresponding to 
    observation categories. Columns represent observed categories, and rows 
    represent forecast categories, for example:
    
                    Observed
                  A    B    C
              A   50   91   71
    Forecast  B   47   2364 170
              C   54   205  3288
    
    Currently, the following scores are available:
        - Heidke Skill Score (hss)
        - Peirce Skill Score (pss)
        - Gerrity Skill Score (gss)
              
    Source: https://github.com/djgagne/hagelslag
    Revised by Donghoon Lee @ Jul-10-2019
    """

    def __init__(self, table=None, n_classes=2, class_names=("1", "0")):
        self.table = table
        self.n_classes = n_classes
        self.class_names = class_names
        if table is None:
            self.table = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def __add__(self, other):
        assert self.n_classes == other.n_classes, "Number of classes does not match"
        return MulticlassContingencyTable(self.table + other.table,
                                          n_classes=self.n_classes,
                                          class_names=self.class_names)
        
    def heidke_skill_score(self):
        """Compute Heidke Skill Score (HSS)
        
        HSS measures the fraction of correct forecasts after eliminating those
        forecasts which would be correct due purely to random chances
        
        Range -inf ≤ HSS ≤ 1, 1 means perfect score
        
        """
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (nf * no).sum() / n ** 2)


    def peirce_skill_score(self):
        """Compute Peirce Skill Score (PSS) (also Hanssen and Kuipers score, 
        True Skill Score)
        
        PSS is similar to the HSS, except that in the demoninator the fraction
        of correct forecasts due to rando mchance is for an unbiased forecast.

        Range: -1 ≤ PSS ≤ 1, 0 means no skill and 1 means perfect score

        """
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (no * no).sum() / n ** 2)

    def gerrity_skill_score(self):
        """Compute the Gerrity Skill Score (GSS)
        
        GSS uses a scoring matrix, which is a tabulation of the reward or penalty
        Every forecast/observation outcome represented by the contingency table.
        
        Range: -1 ≤ GSS ≤ 1, 0 means no skill and 1 means perfect score
        
        Parameters
        ----------
        table: 2d ndarray
            Contingency table.
            
        Returns
        -------
        float
            The Gerrity Skill Score (GSS) value.
    
        References
        ----------
        - Wilks, D., 2011, Statistical Methods in the Atmospheric Sciences
        - Gerrity, J.P., 1992, A Note on Gandin and Murphy's Equitable Skill Score., 
        Monthly Weather Review, 120, 2709-2712.
        """
        k = self.table.shape[0]
        n = float(self.table.sum())
        p_o = self.table.sum(axis=0) / n        # Marginal distribution
        # Control of marginal distribution when a category has all zero counts
        if np.sum(p_o == 0) > 0:
            nzero = np.sum(p_o == 0)
            p_o[p_o == 0] = 0.001
            p_o[np.argmax(p_o)] = p_o[np.argmax(p_o)] - 0.001*nzero
        # J-1 odds ratio
        p_sum = np.cumsum(p_o)[:-1]
        a = (1.0 - p_sum) / p_sum
        # Gerrity(1992) scoring weights
        s = np.zeros(self.table.shape, dtype=float)
        for (i, j) in np.ndindex(*s.shape):
            if i == j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:j]) + np.sum(a[j:k-1]))
            elif i < j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:i]) - (j - i) + np.sum(a[j:k-1]))
            else:
                s[i, j] = s[j, i]
        # Gandin-Murphy Skill Scores (GMSS)
        gmss = np.sum(self.table / float(self.table.sum()) * s)
        return gmss


def msess(yObsd, yFcst, yClim):
   # Mean Squared Error Skill Score (MSESS) (%)
   # MSESS = (1 - MSE_pred/MSE_clim)*100
   #    MSE_pred = 1/n*sum((y-yHat).^2)
   #    MSE_clim = 1/n*sum((y-yMean).^2)
   #
   mse_pred = mean_squared_error(yObsd, yFcst)
   mse_clim = mean_squared_error(yObsd, np.ones(yObsd.shape)*yClim.mean())
   msess = (1 - mse_pred/mse_clim)*100

   return msess

