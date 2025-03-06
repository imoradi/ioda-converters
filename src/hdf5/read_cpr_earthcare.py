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
    data = data.rename({'phony_dim_10': 'obs_id', 'phony_dim_11': 'elevation'})
    geo = geo.rename({'phony_dim_14': 'obs_id', 'phony_dim_15': 'elevation'})
    
    time1 = geo['profileTime']
    epoch_np = np.datetime64(epoch)
    epoch_time = (time1 - epoch_np).astype(np.int64) 
    nobs = data['radarReflectivityFactor'].shape[0]
    nlev = data['radarReflectivityFactor'].shape[1]
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
    ecdata['height'] = geo['binHeight']  # "nscan,nray=nfov,nbin=nelev";
    ecdata['epoch_time'] = geo['profileTime']
    ecdata['epoch_time'].values = np.squeeze(epoch_time)
    ecdata['ReflectivityAttenuated'] = 10 * np.log10(data['radarReflectivityFactor'])  # "nscan,nray=nfov,nbin=nelev,nfreq=nchan"
    ecdata['ReflectivityAttenuated'].values[ecdata['ReflectivityAttenuated'].values < -100] = np.nan
    ecdata['DopplerVelocity'] = data['dopplerVelocity']
    ecdata['surfaceBinNumber'] = data['surfaceBinNumber']
    ecdata['rayQualityFlag'] =  data['rayQualityFlag']
    ecdata['rangeBinValidNumber'] = data['rangeBinValidNumber']
    
    ecdata['zenith_angle'] = ecdata['obs_id'].copy() * 0.0
    ecdata['azimuth_angle'] = ecdata['obs_id'].copy() * 0.0
    ecdata['fov1'] = ecdata['obs_id'].copy() * 0.0


    ecdata['scan_line'] = ecdata['obs_id'].copy()
    ecdata['fov1'] = ecdata['obs_id'].copy() * 0.0

    # expand the dimension for obs (reflectivities)
    expand_dims = {'ReflectivityAttenuated': {'channel': 1}, 'DopplerVelocity': {'channel': 1}}

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
