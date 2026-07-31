# should change these variables
PORTS=12770:12777 # should be a unique port
USER=rili # change to your user name
GPUN=0 # change to gpu device number
MEMORY_LIMIT=67108864 # maximum memory for process in bytes

# not recommended to change environment variables below
CONTAINER_NAME=${USER}_opensar_nogui_gpu${GPUN}_v3_2
SHM_SIZE=1g 
USER_MOUNT=/home/${USER}/  

cd ../build_docker_img_scripts/ && export $(grep -v '^#' docker_dev.env | xargs) && \
docker run --rm -p $PORTS -e HOST_UID='id -u' -it --name $CONTAINER_NAME \
--shm-size=$SHM_SIZE --ulimit memlock=-1 --ulimit stack=$MEMORY_LIMIT \
-v $USER_MOUNT:$USER_MOUNT \
-v $DATA_MNT \
-v $RAID_MNT \
-v $NAS_MNT \
-v $OPENSAR_MNT \
-v "/home/hamk/OpenSAR:/app/OpenSAR" \
-v "$PWD":/app \
--gpus device=$GPUN opensar_torch_gpu_v3_2_complete \
bash -c "bash"
