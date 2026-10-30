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
    """GPM 2A.GPM.DPR granule, V07 (HDF5) or V08 (NetCDF-4, same groups and DimensionNames attributes)."""

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

    epoch_time = np.full(xr_data.nscan.size, np.nan)
    for iscan in xr_data.nscan.values:
        try:
            time1 = datetime(int(xr_data['FS/ScanTime/Year'].values[iscan]),
                             int(xr_data['FS/ScanTime/Month'].values[iscan]),
                             int(xr_data['FS/ScanTime/DayOfMonth'].values[iscan]),
                             int(xr_data['FS/ScanTime/Hour'].values[iscan]),
                             int(xr_data['FS/ScanTime/Minute'].values[iscan]),
                             int(xr_data['FS/ScanTime/Second'].values[iscan]),
                             1000 * int(xr_data['FS/ScanTime/MilliSecond'].values[iscan]))
        except ValueError:
            continue    # missing scan time (fill value)
        time1 = time1.replace(tzinfo=timezone.utc)
        epoch_time[iscan] = round((time1 - epoch).total_seconds())

    # build output dictionary
    dprdata = xr.Dataset()
    # fill values (-9999, -9999.9) become NaN; V08 (NetCDF) stores height as int16 in meters
    def valid(da):
        da = da.astype(float)
        return da.where(da > -9999)

    dprdata['lat'] = valid(xr_data['FS/Latitude'])       # "nscan,nray=nfov";
    dprdata['lon'] = valid(xr_data['FS/Longitude'])      # "nscan,nray=nfov";
    dprdata['height'] = valid(xr_data['FS/PRE/height'])  # "nscan,nray=nfov,nbin=nelev";
    dprdata['epoch_time'] = xr_data['FS/navigation/scLat'].copy()
    dprdata['epoch_time'].values = np.squeeze(epoch_time)
    
    # Copy variables, format "nscan,nray=nfov,nbin=nelev,nfreq=nchan"
    dprdata['ReflectivityAttenuated'] = xr_data['FS/PRE/zFactorMeasured'].copy()
    dprdata['obs'] = xr_data['FS/SLV/zFactorFinal'].copy()

    # Replace invalid obs values with NaN
    dprdata['obs'] = dprdata['obs'].where(dprdata['obs'] >= -100)
    # Make ReflectivityAttenuated NaN wherever obs is NaN
    dprdata['ReflectivityAttenuated'] = (
               dprdata['ReflectivityAttenuated']
               .where(~np.isnan(dprdata['obs']))
    )
    # Drop the bins below the lowest clutter-free bin: there zFactorFinal is filled down to the
    # surface, but zFactorMeasured is ground clutter (50-80 dBZ). binClutterFreeBottom is 1-based.
    clutter_free_bottom = xr_data['FS/PRE/binClutterFreeBottom']
    bin_number = xr.DataArray(np.arange(1, dprdata.sizes['nbin'] + 1), dims='nbin')
    dprdata['ReflectivityAttenuated'] = (
               dprdata['ReflectivityAttenuated']
               .where(bin_number <= clutter_free_bottom)
    )
    # bins that were measured (above the clutter-free bottom of a valid footprint): where these have
    # no echo they count as zero reflectivity in the block (area) mean
    dprdata['measured'] = (bin_number <= clutter_free_bottom) & dprdata['lat'].notnull()

    # nscan,nray=nfov,nfreq=nchan; Ku and Ka look the same way, keep Ku so the angle is one value per location
    dprdata['zenith_angle'] = valid(xr_data['FS/PRE/localZenithAngle'].isel(nfreq=0))
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
                    dbz_threshold=15 ):

    ds = ds_in.copy()
    # --------------------------------------------------
    # Replace inf and values below the threshold with NaN, only in the
    # reflectivity fields (scanline, fov, elevation, channel). Applying the
    # mask to the whole dataset would broadcast elevation and channel onto
    # lat, lon, height and epoch_time, and the IODA writer would then
    # flatten those 2-D arrays and misplace every record.
    # --------------------------------------------------
    mask = np.isfinite(ds[var_name]) & (ds[var_name] >= dbz_threshold)
    for k in ds.data_vars:
        if set(mask.dims).issubset(ds[k].dims):
            ds[k] = ds[k].where(mask)
    measured = ds['measured']
    ds = ds.drop_vars('measured')

    # Block size
    block_scan = 5
    block_fov  = 5
   
    # --------------------------------------------------
    # Trim the scans so they divide by the block size (5x5 footprints, about 25 km). Across track
    # nothing is dropped: the 49 rays give 9 blocks of 5 and a last block of the 4 edge rays
    # (coarsen pads it, and the means skip the padding)
    # --------------------------------------------------
    nscan = (ds.dims[scan_dim] // block_scan) * block_scan
    nfov  = ds.dims[fov_dim]

    ds = ds.isel(
        {scan_dim: slice(0, nscan),
         fov_dim:  slice(0, nfov)}
    )

    # --------------------------------------------------
    # Coarsen 4D reflectivity
    # --------------------------------------------------
    refl_block = ds[var_name].coarsen(
        {scan_dim: block_scan,
         fov_dim:  block_fov},
        boundary="pad"
    )
    dbz_std = refl_block.std(skipna=True)
    
    # area mean in linear units over the measured footprints of the block: a footprint without
    # echo in a bin counts as zero, so a block with a few raining footprints gets a weak mean
    # (the mean over the raining footprints only would overstate the rain area and intensity)
    measured = measured.isel({scan_dim: slice(0, nscan), fov_dim: slice(0, nfov)})
    ds[var_name] = (10 ** (ds[var_name] / 10.0)).fillna(0).where(measured)
    refl_block = ds[var_name].coarsen(
        {scan_dim: block_scan,
         fov_dim:  block_fov},
        boundary="pad"
    )
    refl_mean = refl_block.mean(skipna=True)
    with np.errstate(divide='ignore'):
        dbz_mean = 10 * np.log10(refl_mean)
    # dbz_threshold (15 dBZ, Ku and Ka) applies to the footprints only: footprints below it count
    # as no echo. Every block mean is kept, except blocks with no echo at all (mean 0, -inf dBZ)
    dbz_mean = dbz_mean.where(np.isfinite(dbz_mean))
    refl_std  = refl_block.std(skipna=True)
    refl_std = refl_std * 1e-6 # mm^6/m^3 => cm^6/m^3

    # we cannot just simply average longitudes
    lon = ds["lon"]
    lon_rad = np.deg2rad(lon)
    x = np.cos(lon_rad)
    y = np.sin(lon_rad)
    x_mean = x.coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="pad"
    ).mean(skipna=True)

    y_mean = y.coarsen(
        {scan_dim: block_scan, fov_dim: block_fov},
        boundary="pad"
    ).mean(skipna=True)
    lon_mean = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360

    # --------------------------------------------------
    # Coarsen scanline-only variable
    # --------------------------------------------------
    epoch_mean = ds["epoch_time"].coarsen(
        {scan_dim: block_scan},
        boundary="pad"
    ).mean(skipna=True)


    # --------------------------------------------------
    # Build new dataset
    # --------------------------------------------------
    ds_out = xr.Dataset()

    ds_out["ReflectivityAttenuated"] = dbz_mean
    ds_out["ReflectivityAttenuated_STD_dBZ"] = dbz_std
    ds_out["ReflectivityAttenuated_STD_cm6m3"] = refl_std
    ds_out["lon"] = lon_mean
    ds_out["epoch_time"] = epoch_mean

    for k in ds:
        if k in ['ReflectivityAttenuated', 'lon', "epoch_time"]:
            continue
        # average over the block dimensions the variable has (fov1 has only fov);
        # variables with neither (e.g. centerFreq) are copied below
        window = {dim: size for dim, size in ((scan_dim, block_scan), (fov_dim, block_fov))
                  if dim in ds[k].dims}
        if not window:
            continue
        k_mean = ds[k].coarsen(window, boundary="pad").mean(skipna=True)

        ds_out[k] = k_mean

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
    ds_out["ReflectivityAttenuated_STD_dBZ"].attrs = ds[var_name].attrs.copy()
    ds_out["ReflectivityAttenuated_STD_dBZ"].attrs["description"] = \
        "Standard deviation of grids in dBZ"
    ds_out["ReflectivityAttenuated_STD_cm6m3"].attrs = ds[var_name].attrs.copy()
    ds_out["ReflectivityAttenuated_STD_cm6m3"].attrs["description"] = \
        "Standard deviation of grids in cm^6/m^3"

    return ds_out

def read_dpr_gpm(fname, seqNumber_offset=None):
    dpr_data = read_dpr_hdf_file(fname)
    dpr_data['lon'] = dpr_data['lon'] % 360 # convert to 0-360

    dpr_data = average_dpr_over_fov_scanline(dpr_data)
    # combine geovar is used to combine DPR with precipitation retrievals from DPR
    dpr_data = dpr_data.stack(obs_id=('scanline', 'fov')).reset_index('obs_id')
    dpr_data = dpr_data.transpose('obs_id', 'channel', 'elevation')

    # increment channel coordinate to start at 1
    dpr_data = dpr_data.assign_coords(channel=dpr_data['channel'] + 1)
    if dpr_data.obs_id.size > 0:
        atime = np.min(dpr_data['epoch_time'])
        atime_obj = datetime.utcfromtimestamp(atime.item())
        # this will use hour and minute to offset files in serial processing
        if not seqNumber_offset:
            seqNumber_offset = 100000*int(atime_obj.strftime('%H%M'))

    dpr_data["sequenceNumber"] = xr.DataArray(seqNumber_offset + np.arange(dpr_data.obs_id.size), dpr_data.obs_id.coords)

    dpr_data = dpr_data.rename_vars({"elevation": "elevation1"})
    dpr_data = dpr_data.stack(Location=['obs_id', 'elevation']).reset_index("Location")
    dpr_data = dpr_data.transpose("Location", "channel")
    reff_att = dpr_data.ReflectivityAttenuated
    valid_mask = (reff_att >= -100) & (reff_att <= 100) & (~np.isnan(reff_att)) & (~np.isinf(reff_att))
    valid_locations = dpr_data.Location.values[valid_mask.all(dim="channel")]
    dpr_data = dpr_data.sel(Location=valid_locations)

    # every variable except the observations must have one value per Location
    for k in ['lat', 'lon', 'height', 'epoch_time', 'zenith_angle', 'azimuth_angle', 'sequenceNumber',
              'fov1', 'elevation1']:
        if dpr_data[k].dims != ('Location',):
            raise ValueError(f"read_dpr_gpm: {k} has dimensions {dpr_data[k].dims}, expected ('Location',)")

    return dpr_data
