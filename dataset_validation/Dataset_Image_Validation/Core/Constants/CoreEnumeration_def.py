#=====================================
#    - DO NOT MODIFY THIS FILE -
#-------------------------------------
# Property of @INDRA_DEIMOS
#-------------------------------------
# Generated Enumeration from CoreEnumeration sheet
#=====================================

from enum import Enum

class Colors(Enum):
    RESET = "\033[0m"  # Reset to default color
    BLACK = "\033[30m"  # Black text
    RED = "\033[31m"  # Red text
    GREEN = "\033[92m"  # Green text
    YELLOW = "\033[33m"  # Yellow text
    BLUE = "\033[34m"  # Blue text
    MAGENTA = "\033[35m"  # Magenta text
    CYAN = "\033[36m"  # Cyan text
    WHITE = "\033[37m"  # White text
    BOLD = "\033[1m"  # Bold text
    UNDERLINE = "\033[4m"  # Underlined text
    BRIGHT_BLACK = "\033[90m"  # Bright black (gray)
    BRIGHT_RED = "\033[91m"  # Bright red
    BRIGHT_GREEN = "\033[92m"  # Bright green
    BRIGHT_YELLOW = "\033[93m"  # Bright yellow
    BRIGHT_BLUE = "\033[94m"  # Bright blue
    BRIGHT_MAGENTA = "\033[95m"  # Bright magenta
    BRIGHT_CYAN = "\033[96m"  # Bright cyan
    BRIGHT_WHITE = "\033[97m"  # Bright white

class ImageFormat(Enum):
    TIFF = "tiff"  # Tagged Image File Format, commonly used for high-quality images
    PNG = "png"  # Portable Network Graphics, supports lossless compression
    JPG = "jpg"  # Joint Photographic Experts Group, lossy compression format
    JPEG = "jpeg"  # Alternative notation for JPG, widely used for web images
    BMP = "bmp"  # Bitmap image format, uncompressed and large in size

class LoggingValidity(Enum):
    SAVE_LOG = 0  # save the log
    DELETE_LOG = 1  # delete the log

class Verbosity(Enum):
    NONE = 0  # No output
    LOW = 1  # Minimal output
    MEDIUM = 2  # Medium output 
    HIGH = 3  # Detailed output
    VERY_HIGH = 4  # Maximum level of output
    EXTREME = 5  # Experimental - Add assert details

