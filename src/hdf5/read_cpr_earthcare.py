#!/usr/bin/env python3

#
# (C) Copyright 2020-2024 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
#
# contribution based on prototype by 2023, Isaac Moradi
#

import os
from glob import glob
import sys

import numpy as np
from datetime import datetime, timezone
import xarray as xr
import h5py
from collections import OrderedDict
from pyiodaconv.def_jedi_utils import epoch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def is_cpr_earthcare(fname):
    
    # return True if file is hdf5
    try:
        xr.open_dataset(fname, group='ScienceData')
        return True
    except OSError:
        return False


def read_cpr_earthcare(fname):

    print(fname)
    data = xr.open_dataset(fname, group='ScienceData/Data')
    geo  = xr.open_dataset(fname, group='ScienceData/Geo')
    if 'phony_dim_10' in data.dims:
        dobsdim = 'phony_dim_10'
        delevdim = 'phony_dim_11'
        reflk = 'radarReflectivityFactor'
        timek = 'profileTime'
        doplk = 'dopplerVelocity'
        heightk = 'binHeight'
    else:    
        dobsdim = 'phony_dim_3'
        delevdim = 'phony_dim_4'
        reflk = 'cloud_radar_reflectivity_1km'
        timek = 'time'
        doplk = 'cloud_terminal_velocity1_1km'
        heightk = 'height'

    if 'phony_dim_14' in geo.dims:
        gobsdim = 'phony_dim_14'
        gelevdim = 'phony_dim_15'
    else:    
        gobsdim = 'phony_dim_7'
        gelevdim = 'phony_dim_8'
        
    data = data.rename({dobsdim: 'obs_id', delevdim: 'elevation'})
    geo = geo.rename({gobsdim: 'obs_id', gelevdim: 'elevation'})
    
    time1 = geo[timek]
    epoch_np = np.datetime64(epoch)
    epoch_time = ((time1 - epoch_np) / np.timedelta64(1, "s")).astype(np.int64)
    nobs = data[reflk].shape[0]
    nlev = data[reflk].shape[1]
    nchan = 1
    obs_id = np.arange(nobs)
    channel = [1]
    elevation = np.arange(nlev) 
    # build output dictionary
    ecdata = xr.Dataset()
    ecdata['obs_id'] = xr.DataArray(obs_id, dims={'obs_id': obs_id})
    ecdata['channel'] = xr.DataArray(channel, dims={'channel': channel})
    ecdata['elevation'] = xr.DataArray(elevation, dims={'elevation': elevation})
    ecdata['centerFreq'] = xr.DataArray(np.array([94.05]), dims={"channel": 1})  # GHz
    ecdata['centerWN'] = xr.DataArray(np.array([3.1371]), dims={"channel": 1})   # 1/cm

    
    ecdata['lat'] = geo['latitude']       # "nscan,nray=nfov";
    ecdata['lon'] = geo['longitude']      # "nscan,nray=nfov";
    ecdata['height'] = geo[heightk]      # "nscan,nray=nfov,nbin=nelev";
    ecdata['epoch_time'] = geo[timek]
    ecdata['epoch_time'].values = np.squeeze(epoch_time)
    ecdata['ReflectivityAttenuated'] = 10 * np.log10(data[reflk])  # "nscan,nray=nfov,nbin=nelev,nfreq=nchan"
    ecdata['ReflectivityAttenuated'].values[ecdata['ReflectivityAttenuated'].values < -100] = np.nan
    ecdata['DopplerVelocity'] = data[doplk]
    if 'rayQualityFlag' in data:
        ecdata['PreQC_ReflectivityAttenuated'] =  data['rayQualityFlag']
        ecdata['PreQC_DopplerVelocity'] =  data['dopplerStatusFlag']
        ecdata['surfaceBinNumber'] = data['surfaceBinNumber']
        ecdata['rangeBinValidNumber'] = data['rangeBinValidNumber']
    else: 
        ecdata['quality_flag'] =  data['quality_flag_1km'] #rayQualityFlag']
    ecdata['zenith_angle'] = ecdata['obs_id'].copy() * 0.0
    ecdata['azimuth_angle'] = ecdata['obs_id'].copy() * 0.0
    ecdata['fov1'] = ecdata['obs_id'].copy() * 0.0
    ecdata['scan_line'] = ecdata['obs_id'].copy()
    ecdata['fov1'] = ecdata['obs_id'].copy() * 0.0

    # expand the dimension for obs (reflectivities)
    expand_dims = {'ReflectivityAttenuated': {'channel': 1}}
    if 'DopplerVelocity' in ecdata:
        expand_dims['DopplerVelocity'] = {'channel': 1}

    for k in expand_dims:
        ecdata[k] = ecdata[k].expand_dims(dim=expand_dims[k])

    # reverse the coordinates and add obs_id
    ecdata = ecdata.transpose('obs_id', 'channel', 'elevation')

    # convert to jd/lev/lat/lon
    lon = ecdata['lon'].values
    lon[lon < 0] = 360 + lon[lon < 0]
    ecdata['lon'].values = lon
    ecdata["sequenceNumber"] = xr.DataArray(np.arange(ecdata.obs_id.size), ecdata.obs_id.coords)
    
    return ecdata

