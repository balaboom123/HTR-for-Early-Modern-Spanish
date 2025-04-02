from glob import glob
import os

def find_files(directory, pattern):
    """Recursively find all files with a given pattern"""
    return glob(os.path.join(directory, '**', pattern), recursive=True)