import CoreLog_lib as OBJ_CORE_LOG
import CoreEnumeration_def as OBJ_CORE_ENUMERATION

def fcn_ConsoleSelection(selectionFromArguments, UCselected, VERselected, LOGselected):
    if selectionFromArguments == False:
        options_set = ["Flood Detection", "Vessel Detection", "Radio Frequency Interference", "All use-cases"]
        UC_Selected = OBJ_CORE_LOG.fcn_ConsoleSelectOption(options_set)
        options_set = ["Low", "Medium", "High"]
        VER_Selected = OBJ_CORE_LOG.fcn_ConsoleSelectOption(options_set, "Select an option for verbosity")
        options_set = ["Save", "Delete"]
        LOG_Selected = OBJ_CORE_LOG.fcn_ConsoleSelectOption(options_set, "Select an option for log file storage")
    
        if VER_Selected == 1 or VER_Selected>4 or VER_Selected < 1:
            VER_Selected = OBJ_CORE_ENUMERATION.Verbosity.LOW
        else:
            if VER_Selected == 2:
                VER_Selected = OBJ_CORE_ENUMERATION.Verbosity.HIGH
            else:
                if VER_Selected == 3:
                    VER_Selected = OBJ_CORE_ENUMERATION.Verbosity.VERY_HIGH
        
        if LOG_Selected == 1 or LOG_Selected>2 or LOG_Selected < 1:
            LOG_Selected = OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG
        else:
            if LOG_Selected == 2:
                LOG_Selected = OBJ_CORE_ENUMERATION.LoggingValidity.DELETE_LOG 
        return  UC_Selected, VER_Selected, LOG_Selected     
    else:
        if VERselected == 1 or VERselected>4 or VERselected < 1:
            VERselected = OBJ_CORE_ENUMERATION.Verbosity.LOW
        else:
            if VERselected == 2:
                VERselected = OBJ_CORE_ENUMERATION.Verbosity.HIGH
            else:
                if VERselected == 3:
                    VERselected = OBJ_CORE_ENUMERATION.Verbosity.VERY_HIGH
        
        if LOGselected == 1 or LOGselected>2 or LOGselected < 1:
            LOG_Selected = OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG
        else:
            if LOGselected == 2:
                LOGselected = OBJ_CORE_ENUMERATION.LoggingValidity.DELETE_LOG 
        return  UCselected, VERselected, LOGselected 
    