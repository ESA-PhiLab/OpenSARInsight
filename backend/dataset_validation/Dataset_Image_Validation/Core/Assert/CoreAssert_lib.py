import inspect
import CoreEnumeration_def as OBJ_CORE_ENUM
# ========================================================================================
# =================================== LOCAL FUNCTIONS ====================================
# ========================================================================================

def fcn_GetVariableName(var):
    for name, value in globals().items():
        if value is var:
            return name
    return None  # Return None if the variable name is not found

# Compares substrings of two strings based on given positions and length.
# If no positions or length are provided, compares full strings.
def fcn_AssertString(str1, str2, verbosity: OBJ_CORE_ENUM.Verbosity, str1_startChar=None, str2_startChar=None, char_len=None):
    # print name of fcn for debugging
    if verbosity.value>OBJ_CORE_ENUM.Verbosity.MEDIUM.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUM.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUM.Colors.RESET.value}:")
    # If no start indices are given, start from the beginning
    str1_startChar = str1_startChar if str1_startChar is not None else 0
    str2_startChar = str2_startChar if str2_startChar is not None else 0
    # If no length is given, compare full remaining strings from the start positions
    char_len = char_len if char_len is not None else min(len(str1) - str1_startChar, len(str2) - str2_startChar)
    # data for debugging
    if verbosity.value>OBJ_CORE_ENUM.Verbosity.VERY_HIGH.value:
        print(f'\t\t\t - compare {OBJ_CORE_ENUM.Colors.YELLOW.value}{str1[str1_startChar:str1_startChar+char_len]}{OBJ_CORE_ENUM.Colors.RESET.value} with {OBJ_CORE_ENUM.Colors.YELLOW.value}{str2[str2_startChar:str2_startChar+char_len]}{OBJ_CORE_ENUM.Colors.RESET.value}.')   
    return str1[str1_startChar:str1_startChar+char_len] == str2[str2_startChar:str2_startChar+char_len]

# Compares two integers and returns True if they are equal, otherwise False.
def fcn_AssertInteger(int1, int2, verbosity: OBJ_CORE_ENUM.Verbosity):
    # print name of fcn for debugging
    if verbosity.value>OBJ_CORE_ENUM.Verbosity.VERY_HIGH.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUM.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUM.Colors.RESET.value}:")
        print(f"\t\t\t - compare {OBJ_CORE_ENUM.Colors.YELLOW.value}{int1}{OBJ_CORE_ENUM.Colors.RESET.value} and {OBJ_CORE_ENUM.Colors.YELLOW.value}{int2}{OBJ_CORE_ENUM.Colors.RESET.value}.")
    if abs(int1-int2)>0:
        return False
    else:
        return True

# Compares two floating point numbers
def fcn_AssertFloatingPoint(fp1, fp2, verbosity: OBJ_CORE_ENUM.Verbosity, precision=None):
    # param fp1: First floating point number
    # param fp2: Second floating point number
    # param precision: The allowed precision for comparison (the absolute difference)
     # print name of fcn for debugging
    if verbosity.value>OBJ_CORE_ENUM.Verbosity.HIGH.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUM.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUM.Colors.RESET.value}:")
    # assign double precision
    if (precision==None):
        precision=1e-14  
    # print precision
    if verbosity.value>OBJ_CORE_ENUM.Verbosity.VERY_HIGH.value:
        print(f"\t\t\t - precision for comparing {OBJ_CORE_ENUM.Colors.YELLOW.value}{fp1}{OBJ_CORE_ENUM.Colors.RESET.value} and {OBJ_CORE_ENUM.Colors.YELLOW.value}{fp2}{OBJ_CORE_ENUM.Colors.RESET.value} is {OBJ_CORE_ENUM.Colors.YELLOW.value}{precision}{OBJ_CORE_ENUM.Colors.RESET.value}.")         
    # Compare the absolute difference between the two floating point numbers
    if abs(fp1 - fp2) < precision:
        #print(f"Success: {fp1} is equal to {fp2} within the tolerance of {precision}.")
        return True
    else:
        #print(f"Error: {fp1} is NOT equal to {fp2} within the tolerance of {precision}.")
        return False
