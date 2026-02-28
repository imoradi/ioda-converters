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
import sys
import netCDF4 as nc
import numpy as np
import xarray as xr
from datetime import datetime

import pyiodaconv.ioda_conv_engines as iconv
from pyiodaconv.orddicts import DefaultOrderedDict
from pyiodaconv.def_jedi_utils import set_metadata_attributes, set_obspace_attributes
from pyiodaconv.def_jedi_utils import epoch, iso8601_string
from pyiodaconv.def_jedi_utils import record_time
from read_cloudsat import read_cloudsat, is_hdf4
from read_dpr_gpm import read_dpr_gpm, is_hdf5
from read_cpr_earthcare import read_cpr_earthcare, is_cpr_earthcare

float_missing_value = iconv.get_default_fill_val(np.float32)
int_missing_value = iconv.get_default_fill_val(np.int32)
long_missing_value = iconv.get_default_fill_val(np.int64)

metaDataName = iconv.MetaDataName()
obsValName = iconv.OvalName()

# globals
CLOUDSAT_WMO_sat_ID = 788
GPM_WMO_sat_ID = 288
CPR_EARTHCARE_WMO_sat_ID = 146

# parameter
obs_params = {'CloudSat': ['ReflectivityAttenuated'],
              'EarthCARE-CPR': ['ReflectivityAttenuated', 'DopplerVelocity'],
              'GPM-DPR': ['ReflectivityAttenuated']}
units = {'ReflectivityAttenuated' : 'dBZ', 'DopplerVelocity' : 'm/s'}

locationKeyList = [
    ("latitude", "float"),
    ("longitude", "float"),
    ("dateTime", "long"),
]

GlobalAttrs = {}

def main(args):

    # take a user input and call decoder assume cloudsat hdf4 and GPM DPR hdf5
    # read_cloudsat and read_dpr_gpm return an xarray
    # place into a dictionary and pass to IODA writer

    # start timer
    tic = record_time()
    output_filename = args.output

    if args.input_files is None:
        print(f'no observation files provided exiting')
        sys.exit()

    file_obs_data = []
    for file_path in args.input_files:
        if not os.path.isfile(file_path):
            print(f"'{file_path}' does not exist or is not a file.")
            sys.exit()
        file_is_cpr_cloudsat  = is_hdf4(file_path)
        file_is_dpr_gpm       = is_hdf5(file_path)
        file_is_cpr_earthcare = is_cpr_earthcare(file_path)

        if file_is_cpr_cloudsat:
            xrdata = read_cloudsat(file_path)
            if xrdata.obs_id.size > 0:
               file_obs_data.append(xrdata)
            sensor_name = 'CloudSat'
        elif file_is_cpr_earthcare:
            xrdata = read_cpr_earthcare(file_path)
            if xrdata.obs_id.size > 0:
               file_obs_data.append(xrdata)
            sensor_name = 'EarthCARE-CPR'
        elif file_is_dpr_gpm:
            xrdata = read_dpr_gpm(file_path)
            if xrdata.obs_id.size > 0:
               file_obs_data.append(xrdata)
            sensor_name = 'GPM-DPR'
        print(f"Finshed Reading {sensor_name} Obs ...")

    file_obs_data = concat_file_obs_data(file_obs_data)

    # report time
    toc = record_time(tic=tic)

    if not file_obs_data:
        print(f'no observations found exiting')
        sys.exit()

    sensor_upper = sensor_name.replace("_", " ").upper()
    GlobalAttrs["platformCommonName"] = sensor_upper
    GlobalAttrs["platformLongDescription"] = f"{sensor_upper} Attenuated Reflectivity"
    GlobalAttrs["sensorCentralFrequency"] = str(file_obs_data.centerFreq.values)
    print(file_obs_data)
    obs_data = populate_obs_data(file_obs_data, sensor_name)
    nlocs_int = np.array(len(obs_data[('latitude', metaDataName)]), dtype='int64')
    nlocs = nlocs_int.item()
    nchans = len(obs_data[('sensorChannelNumber', metaDataName)])

    # prepare global attributes we want to output in the file,
    # in addition to the ones already loaded in from the input file
    dtg = False  # pass a reference time for the observation such as to an NWP analysis
    if dtg:
        GlobalAttrs['datetimeReference'] = dtg.strftime("%Y-%m-%dT%H:%M:%SZ")
    GlobalAttrs['converter'] = os.path.basename(__file__)

    # pass parameters to the IODA writer
    VarDims = {'sensorChannelNumber': ['Channel'],
               'sensorCentralFrequency': ['Channel'],
               'sensorCentralWavenumber': ['Channel'],}

    # set key for observable       
    for k in obs_params[sensor_name]:
       VarDims[k] = ['Location', 'Channel']
               
    DimDict = {
        'Location': nlocs,
        'Channel': obs_data[('sensorChannelNumber', metaDataName)],
    }
    writer = iconv.IodaWriter(output_filename, locationKeyList, DimDict)

    VarAttrs = DefaultOrderedDict(lambda: DefaultOrderedDict(dict))
    set_obspace_attributes(VarAttrs)
    set_metadata_attributes(VarAttrs)

    for k in obs_params[sensor_name]:
       VarAttrs[(k, 'ObsValue')]['_FillValue'] = float_missing_value
       VarAttrs[(k, 'ObsError')]['_FillValue'] = float_missing_value
       VarAttrs[(k, 'PreQC')]['_FillValue'] = int_missing_value
       VarAttrs[(k, 'ObsValue')]['units'] = units[k]
       VarAttrs[(k, 'ObsError')]['units'] = units[k]

    # final write to IODA file
    writer.BuildIoda(obs_data, VarDims, VarAttrs, GlobalAttrs)

    # report time
    toc = record_time(tic=tic)

def concat_file_obs_data(file_obs_data, start_date=None, end_date=None): 

    file_obs_data = xr.concat(file_obs_data, dim='obs_id')

    keep_obs_ids = np.ones(file_obs_data.obs_id.size, dtype=bool)
    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d-%H-%M-%S")
        start_date = (np.datetime64(start_date) - np.datetime64(epoch)).astype(np.int64)
        keep_id = file_obs_data.epoch_time.values >= start_date
        keep_obs_ids = keep_obs_ids[keep_id]

    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d-%H-%M-%S")
        end_date = (np.datetime64(end_date) - np.datetime64(epoch)).astype(np.int64)
        keep_id = file_obs_data.epoch_time.values <= start_date
        keep_obs_ids = keep_obs_ids[keep_id]

    # only keep obs within the time range
    file_obs_data = file_obs_data.isel(obs_id=keep_obs_ids)

    return file_obs_data

def populate_obs_data(file_obs_data, sensor_name):

    # appears to be observation cleansing and conditioning
    file_obs_data = file_obs_data.rename_vars({"elevation": "elevation1"})
    file_obs_data = file_obs_data.stack(Location=['obs_id', 'elevation']).reset_index("Location")
    file_obs_data = file_obs_data.transpose("Location", "channel")
    if sensor_name == 'dpr_gpm':
        reff_att = file_obs_data.obs_measured.values 
    else:
        reff_att = file_obs_data.ReflectivityAttenuated.values 
    locid = ( reff_att < -100) | (reff_att > 100) | np.isnan(reff_att) | np.isinf(reff_att)
    locid = file_obs_data.Location.values[np.sum(locid, axis=1) == 0]
    file_obs_data = file_obs_data.isel(Location=locid)
    # end conditioning and cleansing block

    # this function will map the cloud radar data in the cpr xarray
    # into a dictionary to be passed to the IODA writer functions
    nobs = file_obs_data.Location.size
    nchans = file_obs_data.channel.size

    # ideally an attribute from the file read in
    # example: file_obs_data.attrs['ShortName'].decode("utf-8")
    # here we are hardcoding to the function the name of the satellite
    WMO_sat_ID = get_WMO_satellite_ID(sensor_name)

    # allocate space for output depending on which variables are to be saved
    obs_data = init_obs_loc(sensor_name)

    obs_data[('latitude', metaDataName)] = file_obs_data.lat.values.astype(np.float32)
    obs_data[('longitude', metaDataName)] = file_obs_data.lon.values.astype(np.float32)
    obs_data[('satelliteIdentifier', metaDataName)] = np.full((nobs), WMO_sat_ID, dtype='int32')
    obs_data[('sensorCentralFrequency', metaDataName)] = file_obs_data.centerFreq.values.astype(np.float32)
    obs_data[('sensorCentralWavenumber', metaDataName)] = file_obs_data.centerWN.values.astype(np.float32)
    obs_data[('sensorScanPosition', metaDataName)] = file_obs_data.fov1.values.astype(np.int32)
    obs_data[('sensorChannelNumber', metaDataName)] = file_obs_data.channel.values.astype(np.int32)
    # add satellite altitude (height in meters) and use compute_scan_angle function
    obs_data[('sensorViewAngle', metaDataName)] = file_obs_data.zenith_angle.values.astype(np.float32)
    obs_data[('sensorZenithAngle', metaDataName)] = file_obs_data.zenith_angle.values.astype(np.float32)
    obs_data[('sensorAzimuthAngle', metaDataName)] = file_obs_data.azimuth_angle.values.astype(np.float32)
    obs_data[('solarZenithAngle', metaDataName)] = np.zeros(nobs).astype(np.float32)
    obs_data[('solarAzimuthAngle', metaDataName)] = np.zeros(nobs).astype(np.float32)
    obs_data[('height', metaDataName)] = file_obs_data.height.values.astype(np.float32)
    obs_data[('Layer', metaDataName)] = file_obs_data.elevation1.values.astype(np.float32)
    obs_data[('sequenceNumber', metaDataName)] = file_obs_data.sequenceNumber.values.astype(np.int32)
    obs_data[('dateTime', metaDataName)] = file_obs_data.epoch_time.values.astype(np.int64)

    obs_data[('sequenceNumber', metaDataName)] = file_obs_data.sequenceNumber.values.astype(np.int32)

    # have to reorder the channel axis to be last then merge ( nscans x nspots = nlocs )
    for k in obs_params[sensor_name]:
       obs_data[(k, "ObsValue")] = file_obs_data[k].values.astype(np.float32)
       obs_data[(k, "ObsError")] = np.full((nobs, nchans), 5.0, dtype='float32')
       if f"PreQC_{k}" in file_obs_data:
           preqc = file_obs_data[f"PreQC_{k}"].values.astype('int32')
       else:
           preqc = np.full((nobs, nchans), 0, dtype='int32')
       obs_data[(k, "PreQC")] = preqc

    return obs_data


def init_obs_loc(sensor_name):
    obs = {
        ('satelliteIdentifier', metaDataName): [],
        ('sensorChannelNumber', metaDataName): [],
        ('latitude', metaDataName): [],
        ('longitude', metaDataName): [],
        ('dateTime', metaDataName): [],
        ('solarZenithAngle', metaDataName): [],
        ('solarAzimuthAngle', metaDataName): [],
        ('sensorZenithAngle', metaDataName): [],
        ('sensorAzimuthAngle', metaDataName): [],
        ('height', metaDataName): [],
        ('Layer', metaDataName): [],
    }

    for k in obs_params[sensor_name]:
        obs[(k, "ObsValue")] = []
        obs[(k, "ObsError")] = []
        obs[(k, "PreQC")] = []
        
    return obs


def get_WMO_satellite_ID(attrs_shortname):

    if 'CloudSat' in attrs_shortname:
        WMO_sat_ID = CLOUDSAT_WMO_sat_ID
    elif 'GPM' in attrs_shortname:
        WMO_sat_ID = GPM_WMO_sat_ID
    elif 'EarthCARE' in attrs_shortname:
        WMO_sat_ID = CPR_EARTHCARE_WMO_sat_ID        
    else:
        WMO_sat_ID = -1
        print("could not determine satellite from filename: %s" % attrs_shortname)
        sys.exit()

    return WMO_sat_ID


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=(
            'Reads the satellite data '
            ' convert into IODA formatted output files. '
            ' Multiple files are concatenated')
    )

    required = parser.add_argument_group(title='required arguments')
    required.add_argument(
        '-f', '--input_files',
        help="path of satellite observation input file(s)",
        type=str, nargs='+')
    optional = parser.add_argument_group(title='optional arguments')
    optional.add_argument(
        '-o', '--output',
        help='path to output ioda file',
        type=str, default=os.path.join(os.getcwd(), 'output.nc4'))
    optional.add_argument(
        '-s', '--start_date',
        help='state date for output data in YYYY-MM-DD-HH-MM-SS',
        type=str, default=None)
    optional.add_argument(
        '-e', '--end_date',
        help='end date for output data in YYYY-MM-DD-HH-MM-SS',
        type=str, default=None)

    args = parser.parse_args()

    main(args)
