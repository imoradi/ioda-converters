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
import pdb
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class H5ls:
    def __init__(self):
        # Store an empty list for dataset names
        self.names = []

    def __call__(self, name, h5obj):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj, 'dtype') and name not in self.names:
            self.names += [name]

        # we have no return so that the visit function is recursive


def is_hdf5(fname):
    # return True if file is hdf5
    try:
        with h5py.File(fname, 'r'):
            return True
    except OSError:
        return False


def read_dpr_hdf_file(fname):

    print(fname)
    df = h5py.File(fname)
    h5ls = H5ls()
    # this will now visit all objects inside the hdf5 file and store datasets in h5ls.names
    df.visititems(h5ls)
    # get all the dimensions
    dims = {}
    data = {}
    for k in h5ls.names:
        if 'AlgorithmRuntimeInfo' in k:
            continue

        if 'DimensionNames' in df[k].attrs:
            data[k] = df[k][:]
            dims[k] = df[k].attrs['DimensionNames'].decode('UTF-8').split(',')
    df.close()

    unique_dims = {}
    for k in data:
        shape = data[k].shape
        for idim, dname in enumerate(dims[k]):
            # xarray does not allow method as dimension for whatever reason
            dname = dname.replace('method', 'method1')
            unique_dims[dname] = np.arange(shape[idim])

    xr_data = xr.Dataset()
    for dname in unique_dims:
        xr_data[dname] = unique_dims[dname]

    for k in data:
        xr_dims = OrderedDict()
        for dname in dims[k]:
            dname = dname.replace('method', 'method1')
            xr_dims[dname] = unique_dims[dname]
        xr_data[k] = xr.DataArray(data[k], dims=xr_dims)

    epoch_time = np.zeros(xr_data.nscan.size)
    for iscan in xr_data.nscan.values:
        time1 = datetime(xr_data['FS/ScanTime/Year'].values[iscan],
                         xr_data['FS/ScanTime/Month'].values[iscan],
                         xr_data['FS/ScanTime/DayOfMonth'].values[iscan],
                         xr_data['FS/ScanTime/Hour'].values[iscan],
                         xr_data['FS/ScanTime/Minute'].values[iscan],
                         xr_data['FS/ScanTime/Second'].values[iscan],
                         xr_data['FS/ScanTime/MilliSecond'].values[iscan])
        time1 = time1.replace(tzinfo=timezone.utc)
        epoch_time[iscan] = round((time1 - epoch).total_seconds())

    # build output dictionary
    dprdata = xr.Dataset()
    dprdata['lat'] = xr_data['FS/Latitude']       # "nscan,nray=nfov";
    dprdata['lon'] = xr_data['FS/Longitude']      # "nscan,nray=nfov";
    dprdata['height'] = xr_data['FS/PRE/height']  # "nscan,nray=nfov,nbin=nelev";
    dprdata['epoch_time'] = xr_data['FS/navigation/scLat'].copy()
    dprdata['epoch_time'].values = np.squeeze(epoch_time)
    dprdata['ReflectivityAttenuated'] = xr_data['FS/PRE/zFactorMeasured']  # "nscan,nray=nfov,nbin=nelev,nfreq=nchan"
    dprdata['obs'] = xr_data['FS/SLV/zFactorFinal']  # "nscan,nray=nfov,nbin=nelev,nfreq=nchan"
    dprdata['obs'].values[dprdata['obs'].values < -100] = np.nan
    dprdata['zenith_angle'] = xr_data['FS/PRE/localZenithAngle']  # nscan,nray=nfov,nfreq=nchan
    dprdata['azimuth_angle'] = dprdata['zenith_angle'].copy() * 0
    dprdata['centerFreq'] = xr.DataArray(np.array([13.6, 35.5]), dims={"nfreq": 2})  # GHz
    dprdata['centerWN'] = xr.DataArray(np.array([0.453, 1.183]), dims={"nfreq": 2})  # 1/cm

    # rename dimensions
    dprdata = dprdata.rename({'nbin': 'elevation', 'nfreq': 'channel', 'nray': 'fov', 'nscan': 'scanline'})
    dprdata['fov1'] = dprdata.fov.copy()

    return dprdata

def average_dpr_over_fov_scanline(ds_in,
                    var_name="ReflectivityAttenuated",
                    scan_dim="scanline",
                    fov_dim="fov",
                    dbz_threshold=0 ):

    ds = ds_in.copy()
    # --------------------------------------------------
    # Replace inf with NaN
    # --------------------------------------------------
    mask = np.isfinite(ds[var_name]) & (ds[var_name] >= dbz_threshold)
    ds = ds.where(mask)

    # Block size
    block_scan = 7
    block_fov  = 7
   
    # --------------------------------------------------
    # Trim so divisible by 7
    # --------------------------------------------------
    nscan = (ds.dims[scan_dim] // block_scan) * block_scan
    nfov  = (ds.dims[fov_dim]  // block_fov)  * block_fov

    ds = ds.isel(
        {scan_dim: slice(0, nscan),
         fov_dim:  slice(0, nfov)}
    )

    # --------------------------------------------------
    # Coarsen 4D reflectivity
    # --------------------------------------------------
    ds[var_name] = 10 ** (ds[var_name] / 10.0)
    refl_block = ds[var_name].coarsen(
        {scan_dim: block_scan,
         fov_dim:  block_fov},
        boundary="trim"
    )

    refl_mean = refl_block.mean(skipna=True)
    refl_std  = refl_block.std(skipna=True)
    dbz_mean = 10 * np.log10(refl_mean)
    dbz_std = 4.343 * (refl_std / refl_mean)
    dbz_std = dbz_std.where(dbz_mean > dbz_threshold, 0)
    cm6m3_std = refl_std * 1e-6
    # --------------------------------------------------
    # Coarsen 2D variables (lat/lon)
    # --------------------------------------------------
    lat_mean = ds["lat"].coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="trim"
    ).mean(skipna=True)

    lon_mean = ds["lon"].coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="trim"
    ).mean(skipna=True)

    # --------------------------------------------------
    # Coarsen 3D height
    # --------------------------------------------------
    height_mean = ds["height"].coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="trim"
    ).mean(skipna=True)

    # --------------------------------------------------
    # Coarsen scanline-only variable
    # --------------------------------------------------
    epoch_mean = ds["epoch_time"].coarsen(
        {scan_dim: block_scan},
        boundary="trim"
    ).mean(skipna=True)

    # --------------------------------------------------
    # Coarsen angle variables
    # --------------------------------------------------
    zenith_mean = ds["zenith_angle"].coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="trim"
    ).mean(skipna=True)

    azimuth_mean = ds["azimuth_angle"].coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="trim"
    ).mean(skipna=True)

    # --------------------------------------------------
    # Build new dataset
    # --------------------------------------------------
    ds_out = xr.Dataset()

    ds_out["ReflectivityAttenuated"] = dbz_mean
    ds_out["ReflectivityAttenuated_STD"] = cm6m3_std #dbz_std

    ds_out["lat"] = lat_mean
    ds_out["lon"] = lon_mean
    ds_out["height"] = height_mean
    ds_out["epoch_time"] = epoch_mean
    ds_out["zenith_angle"] = zenith_mean
    ds_out["azimuth_angle"] = azimuth_mean
    ds_out['fov1'] = ds_out['fov'].copy()

    # Copy static variables
    ds_out["centerFreq"] = ds["centerFreq"]
    ds_out["centerWN"]   = ds["centerWN"]

    # Preserve coordinates
    ds_out = ds_out.assign_coords({
        scan_dim: refl_mean[scan_dim],
        fov_dim:  refl_mean[fov_dim],
        "elevation": ds["elevation"],
        "channel": ds["channel"]
    })

    # Preserve attributes
    ds_out["ReflectivityAttenuated"].attrs = ds[var_name].attrs
    ds_out["ReflectivityAttenuated_STD"].attrs = ds[var_name].attrs.copy()
    ds_out["ReflectivityAttenuated_STD"].attrs["description"] = \
        "Standard deviation of 7x7 averaged reflectivity"

    return ds_out

def read_dpr_gpm(fname, seqNumber_offset=None):
    dpr_data = read_dpr_hdf_file(fname)
    dpr_data = average_dpr_over_fov_scanline(dpr_data)
    # combine geovar is used to combine DPR with precipitation retrievals from DPR
    dpr_data = dpr_data.stack(obs_id=('scanline', 'fov')).reset_index('obs_id')
    dpr_data = dpr_data.transpose('obs_id', 'channel', 'elevation')

    # convert to jd/lev/lat/lon
    lon = dpr_data['lon'].values
    lon[lon < 0] = 360 + lon[lon < 0]
    dpr_data['lon'].values = lon

    # increment channel coordinate to start at 1
    dpr_data = dpr_data.assign_coords(channel=dpr_data['channel'] + 1)

    atime = np.min(dpr_data['epoch_time'])
    atime_obj = datetime.utcfromtimestamp(atime.item())
    # this will use hour and minute to offset files in serial processing
    if not seqNumber_offset:
        seqNumber_offset = 100000*int(atime_obj.strftime('%H%M'))

    dpr_data["sequenceNumber"] = xr.DataArray(seqNumber_offset + np.arange(dpr_data.obs_id.size), dpr_data.obs_id.coords)

    return dpr_data
