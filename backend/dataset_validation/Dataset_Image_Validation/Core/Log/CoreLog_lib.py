import os
import datetime

# user-defined core libraries
import CoreEnumeration_def as OBJ_CORE_ENUMERATION

def fcn_PrintBanner():
    print(f"===================================================================================\n")
    print(
            " ██████╗ ██████╗ ███████╗███╗   ██╗    ███████╗ █████╗ ██████╗     ██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗\n"
            "██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██╔════╝██╔══██╗██╔══██╗    ██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝\n"
            "██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ███████╗███████║██████╔╝    ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   \n"
            "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ╚════██║██╔══██║██╔══██╗    ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   \n"
            "╚██████╔╝██║     ███████╗██║ ╚████║    ███████║██║  ██║██║  ██║    ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   \n"
            " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   \n"
        )
    print(f"===================================================================================\n")

# Displays a list of options in the console and allows the user to select one.
#   options (list): A list of options to choose from, maintaining order.
#   prompt (str): The message displayed to the user.
def fcn_ConsoleSelectOption(options, prompt="Select a dataset for validation:"):
    if not isinstance(options, list):
        raise TypeError("Options must be a list to maintain order.")
    while True:
        print("\n" + prompt)
        for i, option in enumerate(options, 1):  # Always maintains input order
            print(f"{i}. {option}")
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(options):
                return choice  # Returns the selected option number (not text)
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            return 0  # Return 0 for invalid input

# Returns True if search_term is found in text, otherwise False.
def fcn_ContainsSubstring(text, search_term):
    return search_term in text

# Define the function for generating the log file
def fcn_GenerateLogFile(output_path, folderTestPath, useCase, verbosity: OBJ_CORE_ENUMERATION.Verbosity):
    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the log filename and full path
    filename = f"{output_path}/log_{useCase}_{current_time}.txt"

    try:
        # Check if the file exists, if not, create it
        if not os.path.exists(output_path):
            os.makedirs(output_path)  # Ensure the directory exists
        
        with open(filename, "w") as log_file:
            log_file.write(f"===================================================================================\n")
            log_file.write(f"   ____                      _____ ___    ____     ____           _       __    __ \n")
            log_file.write(f"  / __ \\____  ___  ____     / ___//   |  / __ \\   /  _/___  _____(_)___ _/ /_  / /_\n")
            log_file.write(f" / / / / __ \\/ _ \\/ __ \\    \\__ \\/ /| | / /_/ /   / // __ \\/ ___/ / __ `/ __ \\/ __/\n")
            log_file.write(f"/ /_/ / /_/ /  __/ / / /   ___/ / ___ |/ _, _/  _/ // / / (__  ) / /_/ / / / / /_  \n")
            log_file.write(f"\\____/ .___/\\___/_/ /_/   /____/_/  |_/_/ |_|  /___/_/ /_/____/_/\\__, /_/ /_/\\__/  \n")
            log_file.write(f"    /_/                                                         /____/             \n")
            log_file.write(f"===================================================================================\n")
            # Write initial validation information
            log_file.write(f"\nValidation of files from {folderTestPath} started at {current_time}.\n")
            log_file.write(f"Test cases description:\n")
            log_file.write(f"\tTC-IMG-1: Verify if L0/L1 images are available.\n")
            log_file.write(f"\tTC-IMG-2: Ensure that a minimum of 1000 samples are available.\n")
            log_file.write(f"\tTC-IMG-3: Verify the resolution of the images.\n")
            log_file.write(f"\tTC-IMG-4: Verify if SLC/GEOTiff are available.\n")
            log_file.write(f"\tTC-IMG-5: Verify if Raw images have XML correspondent.\n") 
            log_file.write(f"\tTC-IMG-6: Verify if Raw images have correct format.\n")  
            log_file.write(f"\tTC-IMG-7: Verify if a minimum of 10 SAR Products are available.\n")  
            log_file.write(f"\tTC-IMG-8: Verify if the number of RAW files is 2 times larger than the number of label files.\n")  
            log_file.write(f"\tTC-IMG-9: Verify if the number of tiles is 4 times larger than the number of label files.\n")  
            log_file.write(f"\tTC-IMG-10: Verify if the number of masks is equal to the number of label files.\n")     
                  
            # Verbose output for log creation
            if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:
                print(f"Log file created: {filename}")
        
            # Flush the content to ensure it's written to disk
            log_file.flush()  # Explicitly flush the buffer to disk
        
        return filename  # Return the log file name

    except Exception as e:
        # Error handling if something goes wrong while creating the log file
        if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
            print(f"Error creating log file: {e}")
        return None  # Return None if there was an error


# Appends the given string_info to the log file
def fcn_AppendLog(log_filename, string_info, verbosity: OBJ_CORE_ENUMERATION.Verbosity):
    if string_info:  # Check if string_info is not None and not an empty string
        try:
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(string_info + "\n")
            return True
        except Exception as e:
            print(f"Error writing to log file: {e}")
            return False
    else:
        if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
            print("Provided string_info is empty or None.")
        return False

# Close log file
def fcn_CloseLog(log_file, verbosity: OBJ_CORE_ENUMERATION.Verbosity):
    # Closes the log file safely by ensuring it exists before attempting to close.
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, "a") as file:  # Open in append mode to ensure it exists
                pass  # No need to write anything, just ensure it's accessible
            
            if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
                print(f"Log file '{log_file}' is safely closed.")   
            return True
        except Exception as e:
            if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
                print(f"Error closing log file '{log_file}': {e}")
            return False
    else:
        if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
            print("Invalid log file or file does not exist.")
        return False

# delete log file (useful for debugging)
def fcn_DeleteFile(filename, verbosity: OBJ_CORE_ENUMERATION.Verbosity):
    # Deletes the specified file if it exists
    try:
        if os.path.exists(filename):
            os.remove(filename)
            if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
                print(f"File '{filename}' deleted successfully.")
            return True
        else:
            if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
                print(f"File '{filename}' does not exist.")
            return False
    except Exception as e:
        if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:
            print(f"Error deleting file '{filename}': {e}")
        return False
    
# Function to delete all files in the specified log folder.
def fcn_DeleteAllLogFiles(log_folder_path: str):
    try:
        # List all files in the log folder
        for filename in os.listdir(log_folder_path):
            file_path = os.path.join(log_folder_path, filename)
            # If it's a file (not a directory), delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted log file: {file_path}")
    except Exception as e:
        print(f"Error deleting log files: {e}")
    
