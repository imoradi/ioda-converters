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
    if 'surfaceBinNumber' in ecdata:
        # L1b: average about 25 km along the track (as for GPM DPR)
        ecdata = average_cpr_along_track(ecdata, fname, profile_seconds(time1))

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
    '''
    lon = ecdata['lon'].values
    lon[lon < 0] = 360 + lon[lon < 0]
    ecdata['lon'].values = lon
    '''
    ecdata["sequenceNumber"] = xr.DataArray(np.arange(ecdata.obs_id.size), ecdata.obs_id.coords)


    ecdata = ecdata.rename_vars({"elevation": "elevation1"})
    ecdata = ecdata.stack(Location=['obs_id', 'elevation']).reset_index("Location")
    ecdata = ecdata.transpose("Location", "channel")
    reff_att = ecdata.ReflectivityAttenuated
    # the -35 to 40 dBZ range applies to the profiles before averaging (average_cpr_along_track);
    # every block mean is kept
    valid_mask = (~np.isnan(reff_att)) & (~np.isinf(reff_att))
    valid_locations = ecdata.Location.values[valid_mask.all(dim="channel")]
    ecdata = ecdata.sel(Location=valid_locations)


    return ecdata



def profile_seconds(time1):
    """Profile times in seconds since 1970 (float, so that the overlap with the previous granule can be found)."""
    return ((time1.values - np.datetime64(epoch)) / np.timedelta64(1, "s")).astype(np.float64)


def previous_granule_end(fname):
    """Time of the last profile of the granule before fname (same or previous day folder), or None."""
    day = os.path.dirname(fname)
    try:
        d = datetime.strptime("/".join(day.split("/")[-3:]), "%Y/%m/%d")
        prev_day = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(day))),
                                (d - np.timedelta64(1, "D").astype(object)).strftime("%Y/%m/%d"))
    except ValueError:
        prev_day = day
    pattern = os.path.basename(fname)[:18] + "*.h5"
    files = sorted(set(glob(os.path.join(day, pattern)) + glob(os.path.join(prev_day, pattern))),
                   key=os.path.basename)
    names = [os.path.basename(f) for f in files]
    if os.path.basename(fname) not in names:
        return None
    i = names.index(os.path.basename(fname))
    if i == 0:
        return None
    geo = xr.open_dataset(files[i - 1], group='ScienceData/Geo')
    return profile_seconds(geo['profileTime'])[-1]


def average_cpr_along_track(ec, fname, tsec, nprof=51, dbz_range=(-35.0, 40.0)):
    """
    Average blocks of nprof consecutive profiles (51 profiles about 0.49 km apart, about 25 km)
    at each range bin; the vertical resolution is kept.
    - profiles repeated from the previous granule (consecutive granules overlap by about 56
      profiles) are left out
    - bins outside dbz_range count as no echo (zero in the linear-Z mean); bins at and below the
      surface bin are not used; the block mean is not cut
    - latitude, longitude and time are those of the middle profile; the bin heights are block means
    - blocks with no echo in a bin give no value there
    """
    keep = np.ones(tsec.size, bool)
    tprev = previous_granule_end(fname)
    if tprev is not None:
        keep = tsec > tprev + 1e-3
    ec = ec.isel(obs_id=np.where(keep)[0])
    nb = (ec.obs_id.size // nprof) * nprof
    if nb == 0:
        return ec.isel(obs_id=slice(0, 0))
    ec = ec.isel(obs_id=slice(0, nb))
    nblk = nb // nprof
    nlev = ec.elevation.size
    dbz = ec['ReflectivityAttenuated'].values.astype(np.float64)
    height = ec['height'].values.astype(np.float64)
    height[np.abs(height) > 1e30] = np.nan
    sbin = ec['surfaceBinNumber'].values
    above = np.arange(nlev)[None, :] < sbin[:, None]          # bins above the surface
    inrange = np.isfinite(dbz) & (dbz >= dbz_range[0]) & (dbz <= dbz_range[1])
    zlin = np.where(inrange, 10 ** (dbz / 10.0), 0.0)
    zlin[~above | ~np.isfinite(height)] = np.nan
    zmean = np.nanmean(zlin.reshape(nblk, nprof, nlev), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dbz_mean = 10 * np.log10(zmean)
    dbz_mean[~np.isfinite(dbz_mean)] = np.nan
    hmean = np.nanmean(np.where(np.isfinite(zlin), height, np.nan).reshape(nblk, nprof, nlev), axis=1)
    dop = ec['DopplerVelocity'].values.astype(np.float64)
    dop = np.where(inrange & np.isfinite(zlin), dop, np.nan)
    dop_mean = np.nanmean(dop.reshape(nblk, nprof, nlev), axis=1)
    mid = np.arange(nblk) * nprof + nprof // 2

    out = ec.isel(obs_id=mid)
    out = out.assign_coords(obs_id=np.arange(nblk))
    out['ReflectivityAttenuated'] = (('obs_id', 'elevation'), dbz_mean.astype(np.float32))
    out['height'] = (('obs_id', 'elevation'), hmean.astype(np.float32))
    out['DopplerVelocity'] = (('obs_id', 'elevation'), dop_mean.astype(np.float32))
    return out
