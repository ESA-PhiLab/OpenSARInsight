# SARFI (Synthetic Aperture Radar File Indexer)

SARFI is a tool to convert latitude, longitude coordinates to SLC coordinates (line and sample).
Given a csv file, with columns latitude, longitude, start time and end time and the directory containing 
the SAR products the tool matches the timestamped latitude and longitude to the SAR products the user gave.
The tool currently supports matching of timestamped latitude longitude for Sentinel-1 products only,
expects the product to follow the .SAFE product format where the product folder contains an annotation folder
with .xml files, these files are used to match the timestamped latitude and longitude to the line and sample. 

## Install

From within the opensar docker container navigate to the ``SARFI`` directory and execute
the command ``pip install .`` to install the SARFI package.

## Running

To run the script you need to have Sentinel-1 products downloaded and provide an input
csv file to convert timestamped latitude and longitude coordinates to SLC coordinates. An example input csv is provided in ``Input CSV``. Once you have updated the config,
you can run the script like this ``cli.py -f best``.
### Input CSV
```csv
start_time,stop_time,latitude,longitude
2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000,6.2487081130,3.1850089290
2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000,6.2724962780,3.1941481660
```
### Configuration
You should update the following in ``slc_matcher_config.yaml``:
* ``safe_dir_fp``, the path to directory containing the Sentinel-1 products
* ``input_csv_fp``, the file path to input csv used to convert timestamped lat lon to SLC coords
* ``output_csv_fp``, the file path to output matching csv file to
* ``dem_cache_dir``, the path to the directory to save the DEM files to

### Output CSV

An example of the output csv produced for the given ``Input CSV`` (note filenames
have been modified for security reasons):
```csv
filename,line,sample,target_latitude,target_longitude,target_start_time,target_stop_time
/input/S1A_IW_SLC__1SDV_20200126T052956_20200126T053026_030967_038E50_4681.SAFE/annotation/s1a-iw3-slc-vh-20200126t052958-20200126t053026-030967-038e50-003.xml,6826,16349,6.248708113,3.185008929,2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000
/input/S1A_IW_SLC__1SDV_20200126T052956_20200126T053026_030967_038E50_4681.SAFE/annotation/s1a-iw3-slc-vh-20200126t052958-20200126t053026-030967-038e50-003.xml,6628,16208,6.272496278,3.194148166,2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000
/input/S1A_IW_SLC__1SDV_20200126T052956_20200126T053026_030967_038E50_4681.SAFE/annotation/s1a-iw3-slc-vv-20200126t052958-20200126t053026-030967-038e50-006.xml,6826,16349,6.248708113,3.185008929,2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000
/input/S1A_IW_SLC__1SDV_20200126T052956_20200126T053026_030967_038E50_4681.SAFE/annotation/s1a-iw3-slc-vv-20200126t052958-20200126t053026-030967-038e50-006.xml,6628,16208,6.272496278,3.194148166,2020-01-26T05:29:56.0000,2020-01-26T05:30:30.0000
```
