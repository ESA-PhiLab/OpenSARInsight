export $(grep -v '^#' docker_dev.env | xargs)  && \
cd $SLC2RAW_PATH && python3 -m pip install --break-system-packages . && \
cd $L0READER_PATH && python3 -m pip install --break-system-packages  . && \
cd $L0PATCHER_PATH && python3 -m pip install --break-system-packages  . && \
cd $SARFI_PATH && python3 -m pip install --break-system-packages  . && \ 
cd $L0_RC_PATH && python3 -m pip install --break-system-packages  .
