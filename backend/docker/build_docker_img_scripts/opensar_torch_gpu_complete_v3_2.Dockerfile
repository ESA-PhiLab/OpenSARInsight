FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set non-interactive mode for apt and install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH \ 
&& export CPLUS_INCLUDE_PATH=/usr/include/gdal \
&& export C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update && apt-get install -y --no-install-recommends \
software-properties-common build-essential wget curl tree git ca-certificates && \
add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
apt-get install -y --no-install-recommends python3-dev python3-pip procps libvips libopencv-dev && \
apt-get install -y --no-install-recommends python3-opencv libglib2.0-0 ffmpeg libsm6 libxext6 && \
apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends --reinstall gdal-bin python3-gdal
RUN apt-get install -y --no-install-recommends --reinstall libpq-dev libpq5 libgdal-dev 

# Install any python packages you need
COPY requirements_v3_2.txt /tmp/requirements_v3_2.txt
RUN python3 -m pip install --break-system-packages --force-reinstall --upgrade --ignore-installed -r /tmp/requirements_v3_2.txt
RUN apt-get update && apt-get install --reinstall -y libmpich-dev hwloc-nox libmpich12 mpich
# after building containiner update docker_dev.env and then run install_opensar_internal_dependencies.sh to install the internal dependencies in the container
# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]