#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-

# Streamflow NETwoK (SNETK)


import numpy as np
import gdal
import osr

def adBoundary(fdir):
    vempty = np.zeros([1, fdir.shape[1]])
    hempty = np.zeros([fdir.shape[0]+2, 1])
    fdir = np.vstack([vempty, fdir, vempty])
    fdir = np.hstack([hempty, fdir, hempty])
    return fdir


def rmBoundary(fdir):
    return fdir[1:-1, 1:-1]


def focalSum(funt, fdir):
    '''
    Calculates focal flow sum from FDIR
    '''
    slices = []
    slices.append(funt[1:-1,:-2] * (fdir[1:-1,:-2] == 1))      # East
    slices.append(funt[:-2,:-2] * (fdir[:-2,:-2] == 2))       # SouthEast
    slices.append(funt[:-2,1:-1] * (fdir[:-2,1:-1] == 4))      # South
    slices.append(funt[:-2,2:] * (fdir[:-2,2:] == 8))        # SouthWest
    slices.append(funt[1:-1,2:] * (fdir[1:-1,2:] == 16))      # West
    slices.append(funt[2:,2:] * (fdir[2:,2:] == 32))        # NorthWest
    slices.append(funt[2:,1:-1] * (fdir[2:,1:-1] == 64))      # North
    slices.append(funt[2:,:-2] * (fdir[2:,:-2] == 128))      # NorthEast
    stack = np.dstack(slices)
    incr = np.sum(stack,2)
    return incr


def flowAccumulate(fdir, init=None):
    # https://gis.stackexchange.com/questions/63160/arcmap-10-restrict-flow-accumulation
    
# =============================================================================
#     # Revising ---------------- #
#     fdir = fdirCopy.copy()
#     init = temp1
#     # --------------------------# 
# =============================================================================
    
    fdir = adBoundary(fdir)    
    # First step with initial unit area
    if init is None:
        # Number of grids
        init = np.ones(fdir.shape)
        incr = focalSum(init, fdir)
        totl = np.ones(incr.shape) + incr
        
    else:
        # Accumulated area
        incr = focalSum(adBoundary(init), fdir)
        totl = init + incr
    
    # Repeat steps until it meets no more flows
    while np.sum(incr) != 0:
        incr = focalSum(adBoundary(incr), fdir)
        totl = totl + incr
    
    return totl



def make_raster(in_ds, fn, data, data_type, nodata=None):
    """Create a one-band GeoTiff.

    in_ds     - datasource to copy projection and geotransform from
    fn        - path to the file to create
    data      - Numpy array containing data to archive
    data_type - output data type
    nodata    - optional NoData burn_values
    """

    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(
        fn, in_ds.RasterXSize, in_ds.RasterYSize, 1, data_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    #out_band.ComputerStaitstics(False)
    print('"{}" is printed.'.format(fn))
    return out_ds


def make_raster360(fn, data, data_type, nodata=None):
    """Create a one-band GeoTiff.

    fn        - path to the file to create
    data      - Numpy array containing data to archive
    data_type - output data type
    nodata    - optional NoData burn_values
    """
    
    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(fn, data.shape[1], data.shape[0], 1, data_type)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.SetGeoTransform((-180.0, 0.5, 0.0, 90.0, 0.0, -0.5))
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    #out_band.ComputerStaitstics(False)
    print('"{}" is printed.'.format(fn))
    
    return out_ds







