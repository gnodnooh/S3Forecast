#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-
'''
    This script does attribute variability of seasonal and monthly streamflow 
    to global and local predictors and predict streamflow.
    
    Current prediction types:
        SP1: Seasonal prediction (s-1)
        SP2: Seasonal prediction (s-1)
        MP1: Monthly prediction (t-1)
        MP2: Monthly prediction (t-2)
        MP3: Monthly prediction (t-3)

    Donghoon Lee (dlee298@wisc.edu)
    Date created: 01/16/2019
'''

import os, sys, getopt
import numpy as np
import pandas as pd
import sspred
import time

def attribute(args):
    
    # REMOVE: Argument demonstration
    args = '--flow=/Users/dlee/data/attribute/chtc_in/dfFlowGrid813.hdf \
            --glob=/Users/dlee/data/attribute/chtc_in/dfPredGlob.hdf \
            --locl=/Users/dlee/data/attribute/chtc_in/dfPredGridLocl813.hdf \
            --note=test'.split()

    # Arguments and Parameters control
    isnote = False
    try: 
        opts, args = getopt.getopt(args, '', ['flow=', 'glob=', 'locl=', 'note='])
    except getopt.GetoptError:
        print('{} --flow=<filename> --mon=<filename> --pred=<filename>'.format(sys.argv[0]))
        sys.exit(2)
    for o, a in opts:
        if o == '--flow':
            filnFlow = a
        elif o == '--glob':
            filnGlob = a
        elif o == '--locl':
            filnLocl = a
        elif o == '--note':
            isnote = True
            note = a
        else:
            assert False, "unhandled option"
    
    
    # Load required data (Please see details in "attribute_load.py")
    # Flow Dataframe
    dfFlow = pd.read_hdf(filnFlow)
    pointList = dfFlow.columns
    print('"%s" is imported.' % filnFlow)
    # Predictor Dataframe
    dfGlob = pd.read_hdf(filnGlob)
    print('"%s" is imported.' % filnGlob)
    dfLocl = pd.read_hdf(filnLocl)
    print('"%s" is imported.' % filnLocl)
    # Seasonal averages
    dfFlowMo3 = dfFlow.rolling(3, min_periods=1, center=True).mean()
    dfLoclMo3 = dfLocl.rolling(3, min_periods=1, center=True).mean()
        
    # Set each matching item into a tuple
    sp1 = []         # S1: Seasonal Prediction  (Glob: 8-3, Locl: 3)
    mp1 = []         # M1: Monthly Prediction   (Glob: 8-2, Locl: 1)
    mp2 = []         # M2: Monthly Prediction   (Glob: 8-3, Locl: 2)
    mp3 = []         # M3: Monthly Prediction   (Glob: 8-4, Locl: 3)
    stime = time.time()
    for i in range(pointList.shape[0]):
        point_no = pointList[i]        
        # S1 Prediction
        dfFlow_point = dfFlowMo3[point_no]
        dfPred_point = pd.concat([dfGlob, dfLoclMo3[point_no]], axis=1)
        leadMat = np.array([[8,8,8,8,3,3], [3,3,3,3,3,3]])     # [[Max][Min]]
        iargs = (dfFlow_point, dfPred_point, leadMat, 13, point_no)
        sp1.append(sspred.predict(iargs))
        # M1 Prediction
        dfFlow_point = dfFlow[point_no]
        dfPred_point = pd.concat([dfGlob, dfLocl[point_no]], axis=1)
        leadMat = np.array([[8,8,8,8,1,1], [2,2,2,2,1,1]])     # [[Max][Min]]
        iargs = (dfFlow_point, dfPred_point, leadMat, 13, point_no)
        mp1.append(sspred.predict(iargs))
        # M2 Prediction
        dfFlow_point = dfFlow[point_no]
        dfPred_point = pd.concat([dfGlob, dfLocl[point_no]], axis=1)
        leadMat = np.array([[8,8,8,8,2,2], [3,3,3,3,2,2]])     # [[Max][Min]]
        iargs = (dfFlow_point, dfPred_point, leadMat, 13, point_no)
        mp2.append(sspred.predict(iargs))
        # M3 Prediction
        dfFlow_point = dfFlow[point_no]
        dfPred_point = pd.concat([dfGlob, dfLocl[point_no]], axis=1)
        leadMat = np.array([[8,8,8,8,3,3], [4,4,4,4,3,3]])     # [[Max][Min]]
        iargs = (dfFlow_point, dfPred_point, leadMat, 13, point_no)
        mp3.append(sspred.predict(iargs))
        
    # Printing total results
    etime = time.time() - stime
    print('%d points took %.2fs' % (len(sp1),etime))
    
    # Save prediction results
    outfiln = os.path.splitext(filnFlow)[0].split('/')[-1]
    if isnote:
        outfiln = outfiln + '_' + note
    np.savez_compressed(outfiln, sp1=sp1, mp1=mp1, mp2=mp2, mp3=mp3)
    print('{}.npz is saved.'.format(outfiln))


if __name__ == "__main__":
    attribute(sys.argv[1:])




