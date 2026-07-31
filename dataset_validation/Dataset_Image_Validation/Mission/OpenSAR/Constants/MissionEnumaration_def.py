#=====================================
#    - DO NOT MODIFY THIS FILE -
#-------------------------------------
# Property of @INDRA_DEIMOS
#-------------------------------------
# Generated Enumeration from MissionEnumaration sheet
#=====================================

from enum import Enum

class LogDetailLevel(Enum):
    COMPRESSED = 0  # save the log
    DETAILED = 1  # delete the log

class UseCaseSelector(Enum):
    UV_FLOOD = 1  # use case flood detection
    UC_VESSEL = 2  # use case vessel detection
    UC_RFI = 3  # use case RFI
    UC_ALL = 4  # all use cases

