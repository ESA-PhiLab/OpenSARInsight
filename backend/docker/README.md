# Docker Image Instructions

1. Navigate to ``build_docker_img_scripts`` directory and execute ``build_docker_img_opensar_torch_complete.sh``
2. Now that you have built the image, update ``docker_dev.env`` and ``run_docker_opensar_gui_gpun_v3_2.sh``
3. Navigate to ``run_docker_container_scripts`` directory and execute ``run_docker_opensar_gui_gpun_v3_2.sh``
4. Navigate to ``build_docker_img_scripts`` and execute ``install_opensar_internal_dependencies.sh``

## Environment file

You need to update docker_dev.env with the paths to the opensar internal dependencies so that they can be installed as packages.
docker_dev.env vars explained:
```bash
# Paths for opensar internal dependencies 
SARFI_PATH=/replace/with/path/to/SARFI/directory
SLC2RAW_PATH=/replace/with/path/to/slc2raw-main/directory
L0READER_PATH=/replace/with/path/to/L0Reader-main/directory
L0PATCHER_PATH=/replace/with/path/to/L0Patcher-main/directory
L0_RC_PATH=/replace/with/path/to/l0_to_range_compressed/directory
```
