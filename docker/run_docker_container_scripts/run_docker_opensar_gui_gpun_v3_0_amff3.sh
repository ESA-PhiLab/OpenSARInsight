# should change these variables
PORTS=6064:6036 # should be a unique port
USER=amff # change to your user name
GPUN=1 # change to gpu device number
MEMORY_LIMIT=67108864 # maximum memory for process in bytes

# not recommended to change environment variables below
CONTAINER_NAME=${USER}3_opensar_nogui_gpu${GPUN}_v3_0
SHM_SIZE=1g 
USER_MOUNT=/home/${USER}/ 

docker run --rm -p $PORTS -e HOST_UID='id -u' -it --name $CONTAINER_NAME \
--shm-size=$SHM_SIZE --ulimit memlock=-1 --ulimit stack=$MEMORY_LIMIT \
-v $USER_MOUNT:$USER_MOUNT \
-v /data/:/data/ \
-v /media/raid/:/media/raid/ \
-v /mnt/appide_nas/data_lake/:/mnt/appide_nas/data_lake/ \
-v /media/ubuntu_24_04/data/:/media/ubuntu_24_04/data/ \
--gpus device=$GPUN opensar_torch_gpu_v3_0_complete \
bash -c "apt-get update && apt-get install --reinstall -y libmpich-dev hwloc-nox libmpich12 mpich && bash"