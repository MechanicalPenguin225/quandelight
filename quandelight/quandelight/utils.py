import numpy as np
import os
import h5py

#### WIP
def make_pngs(hdf5_filename) :
    current_filename = os.path.basename(__file__)

    file = h5py.File(f'{current_filename[:-3]}-out/{hdf5_filename}.h5', 'r')

#### NOT WIP.
def pprint(string, color):
    """prints string in color using ANSI terminal standards.
    Parameters :
    ----------
    string : printable
        thing to print (possibly a string)

    color : str
        color to print in. Available:
        red
        green
        yellow
        light purple
        purple
        cyan
        light gray
        black (srsly don't use this man)
    """

    colors = ["red", "green", "yellow", "light purple", "purple", "cyan", "light gray", "black"]

    codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "light purple": "\033[94m",
        "purple" : "\033[95m",
        "cyan": "\033[96m",
        "light gray": "\033[97m",
        "black": "\033[98m"
    }

    for available_color in colors :
        if color == available_color :
            print(f"""{codes[color]} {string}\033[00m""")
            break
    else :
        print(string)
    return None
