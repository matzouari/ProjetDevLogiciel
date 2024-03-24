"""
This module adds context manager for temporary files generated by the tests.
"""

import shutil
import os


class TmpFileManager:
    """
    A class to track record of every temporary files created by the tests.
    """
    tmp_files = set('')
    tmp_folders = set('')

    @classmethod
    def tmp_file(cls, name=''):
        cls.tmp_files.add(name)
        return name

    @classmethod
    def tmp_folder(cls, name=''):
        cls.tmp_folders.add(name)
        return name

    @classmethod
    def cleanup(cls):
        while cls.tmp_files:
            file = cls.tmp_files.pop()
            if os.path.isfile(file):
                os.remove(file)
        while cls.tmp_folders:
            folder = cls.tmp_folders.pop()
            shutil.rmtree(folder)

def cleanup_tmp_files(test_func):
    """
    A decorator to help test codes remove temporary files after the tests.
    """
    def wrapper_function():
        try:
            test_func()
        finally:
            TmpFileManager.cleanup()

    return wrapper_function
