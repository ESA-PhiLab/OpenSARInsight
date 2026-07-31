# Geocoding Block

The function of geocoding block is to geocode SLC image patches in our dataset,
with support for geocoding vessel, rfi and flood image patches.
<br>
In order to run the code the user must provide the correct paths in the configuration file
(``configuration.yaml``):<br>

```yaml
# Configuration file for Geocoder class (see geocode.py)
# product_directory_path = path to directory containing L1 products
product_directory_path: /mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/full_raw_scenes/
# orbit_directory_path = path to directory containing orbit files
orbit_directory_path: /media/raid/opensar/POEORB/
# labels_directory_path = path to directory containing L1 label files
labels_directory_path: /media/raid/opensar/new_labels/vessel/
# dem_directory_path = path to directory to use for geocoding 
dem_directory_path: /home/dadd/OPENSAR/latest-code/backend/data/cop_glo_30_geocoding/
# lut_file_path = path to excel file containing the look up table for the dataset
lut_file_path: /home/dadd/OPENSAR/latest-code/backend/data/lut/dataset_product_lut.xlsx
# pol = polarization (vh or vv)
pol: vh
# dem_server = server to fetch the DEM from (aws or msft)
dem_server: aws
# overwrite = true means to overwrite files in dem_directory_path otherwise false
overwrite: true
```
The ``lut_file_path`` defines the look up table for the dataset, with the columns:
* ``USE_CASE`` must be one of FD, VD, RFI
* ``L0_PRODUCT_NAME`` Sentinel-1 L0 (Raw) product name
* ``L1_PRODUCT_NAME`` Sentinel-1 L1 (SLC) product name
* ``L1_PRODUCT_NAME_INTA`` Sentinel-1 L1 (SLC) product name used in label files
* ``ORBIT_FILE_NAME`` S1A orbit file name

Once the user has updated the ``configuration.yaml`` file they can run the geocoding
tests in ``test_geocode.py``. If there is an issue with the tests then it's likely the
user has not entered the correct values in the configuration file, otherwise if the current tests are passing then the user can create new tests to geocode other patches. The user should run these scripts from within the docker container (see
``docker`` folder in this repo).
<br>
The output of the script is a geocoded tiff image. You can query the latitude and longitude values of each pixel using rasterio (see geocode_patch in ``test_geocode.py`` for how to do this).

NOTE: these geocoding scripts were produced by adapting eotools to geocode patches.
