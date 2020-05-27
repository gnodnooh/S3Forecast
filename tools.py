import os
import numpy as np
import pandas as pd
import scipy.io as spio
from itertools import combinations
from dataclasses import dataclass
from typing import Any
import urllib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

@dataclass
class MBOX:
    point_no:   np.int64 = None
    status:     int = None
    month:      int = None
    maxPred:    Any = None
    maxcorr:    Any = None
    maxsign:    Any = None
    maxlead:    Any = None
    strRemoved: Any = None
    strPred:    Any = None
    nPred:      Any = None
    xLeadCorr:  Any = None
    strAuto:    Any = None
    strLead:    Any = None
    nAuto:      Any = None
    nLead:      Any = None
    xLeadOptm:  Any = None
    regr:       Any = None
    xTran:      Any = None
    xTest:      Any = None
    yTran:      Any = None
    yTest:      Any = None
    yTranHat:   Any = None
    yTestHat:   Any = None
        
@dataclass
class OUTBOX:
    m01:MBOX = None
    m02:MBOX = None
    m03:MBOX = None
    m04:MBOX = None
    m05:MBOX = None
    m06:MBOX = None
    m07:MBOX = None
    m08:MBOX = None
    m09:MBOX = None
    m10:MBOX = None
    m11:MBOX = None
    m12:MBOX = None

def save_hdf(filn, df, set_print = True):
    df.to_hdf(filn, key='df', complib='blosc:zstd', complevel=9)
    if set_print:
        print('%s is saved.' % filn)

def multi_equal(*args):
    for pair in combinations(args, 2):
        assert np.array_equal(pair[0], pair[1])
        
        
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    Source: https://stackoverflow.com/a/8832212/10164193
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


def DownloadFromURL(fullURL, fullDIR, showLog = False):
    '''
    Downloads the inserted hyperlinks (URLs) to the inserted files in the disk
    '''
    # Make parent directories if they do not exist
    if type(fullDIR) == list:
        parentDIRS = list(np.unique([os.path.dirname(DIR) for DIR in fullDIR]))
        for parentDIR in parentDIRS:
            os.makedirs(parentDIR, exist_ok=True)
        # Download all files
        nError = 0
        nExist = 0
        nDown = 0
        for file_url, file_dir in zip(fullURL, fullDIR):
            if not os.path.exists(file_dir):
                try:
                    urllib.request.urlretrieve(file_url, file_dir)
                    nDown += 1
                    print(file_dir, 'is saved.')
                except:
                    nError += 1
                    pass
            else:
                nExist += 1
        if showLog:
            print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
            
    elif type(fullDIR) == str:
        parentDIRS = os.path.dirname(fullDIR)
        # Download all files
        nError = 0
        nExist = 0
        nDown = 0
        if not os.path.exists(fullDIR):
                try:
                    urllib.request.urlretrieve(fullURL, fullDIR)
                    nDown += 1
                    print(fullDIR, 'is saved.')
                except:
                    nError += 1
                    pass
                else:
                    nExist += 1
        if showLog:
            print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
    return


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# Colarmap and Colorbar controller
def cbarpam(bounds, color, labloc='on', boundaries=None, extension=None):
    '''Returns parameters for colormap and colorbar objects with a specified style.

        Parameters
        ----------
        bounds: list of bounds
        color: name of colormap or list of color names

        labloc: 'on' or 'in'
        boundaries: 
        extension: 'both', 'min', 'max'

        Return
        ------
        cmap: colormap
        norm: nomalization
        vmin: vmin for plotting
        vmax: vmax for plotting
        boundaries: boundaries for plotting
    '''
    
    gradient = np.linspace(0, 1, len(bounds)+1)
    # Create colorlist
    if type(color) is list:
        cmap = colors.ListedColormap(color,"")
    elif type(color) is str:
        cmap = plt.get_cmap(color, len(gradient))    
        # Extension
        colorsList = list(cmap(np.arange(len(gradient))))
        if extension is 'both':
            cmap = colors.ListedColormap(colorsList[1:-1],"")
            cmap.set_under(colorsList[0])
            cmap.set_over(colorsList[-1])
        elif extension is 'max':
            cmap = colors.ListedColormap(colorsList[:-1],"")
            cmap.set_over(colorsList[-1])
        elif extension is 'min':
            cmap = colors.ListedColormap(colorsList[1:],"")
            cmap.set_under(colorsList[0])
        elif extension is None:
            gradient = np.linspace(0, 1, len(bounds)-1)
            cmap = plt.get_cmap(color, len(gradient))
        else:
            raise ValueError('Check the extension')
    else:
        raise ValueError('Check the type of color.')
    # Normalization
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # vmin and vmax
    vmin=bounds[0]
    vmax=bounds[-1]
    # Ticks
    if labloc == 'on':
        ticks = bounds
    elif labloc == 'in':
        ticks = np.array(bounds)[0:-1] + (np.array(bounds)[1:] - np.array(bounds)[0:-1])/2
    
    return cmap, norm, vmin, vmax, ticks, boundaries


def GDFPlotOrder(gdf, boundaries, ax, column, cmap, norm, vmin, vmax, order='seq'):
    gdf = gdf.copy()
    orderList = np.arange(0,len(boundaries)-1)
    if order == 'div':
        if len(boundaries) % 2 == 0:
            # Center color exist
            mid = np.median(orderList).astype(int)
            orderList = np.array([mid, *np.vstack((np.flipud(orderList[:mid]), 
                                                   orderList[mid+1:])).flatten('F')])
        else:
            # No center color
            n = len(orderList)
            orderList = np.vstack((np.flipud(orderList[:int(n/2)]), 
                                   orderList[int(n/2):])).flatten('F')
    for i in orderList:
        gdfTemp = gdf[(boundaries[i] <= gdf[column]) & (gdf[column] < boundaries[i+1])]
        if len(gdfTemp) > 0:
            gdfTemp.plot(ax=ax, column=column, markersize=30, edgecolor='black', 
                         cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)    