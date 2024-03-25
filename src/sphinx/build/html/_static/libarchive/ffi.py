from __future__ import division, print_function, unicode_literals

from ctypes import (
    c_char_p, c_int, c_uint, c_long, c_longlong, c_size_t, c_void_p,
    c_wchar_p, CFUNCTYPE, POINTER,
)

try:
    from ctypes import c_ssize_t
except ImportError:
    from ctypes import c_longlong as c_ssize_t

import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sys

from .exception import ArchiveError


logger = logging.getLogger('libarchive')

page_size = mmap.PAGESIZE

# libarchive path order:
# * path specified by the LIBARCHIVE environment variable
# * platform dependent location used in conda packages
# * location determined by find_library
libarchive_path = os.environ.get('LIBARCHIVE')
backup_path = os.environ.get('PATH')
if libarchive_path is None:
    if sys.platform == 'win32':
        library_bin = os.path.join(sys.prefix, 'Library', 'bin')
        test_path = os.path.join(library_bin, 'archive.dll')
        if backup_path:
            os.environ['PATH'] = "%s;%s" % (library_bin, backup_path)
    elif sys.platform == 'darwin':
        test_path = os.path.join(sys.prefix, 'lib', 'libarchive.dylib')
    elif sys.platform.startswith('linux'):
        test_path = os.path.join(sys.prefix, 'lib', 'libarchive.so')
    else:
        test_path = find_library('archive')
    try:
        ctypes.cdll.LoadLibrary(test_path)
        libarchive_path = test_path
    except:
        libarchive_path = None
if libarchive_path is None:
    libarchive_path = find_library('archive')
if not libarchive_path:
    raise ImportError("Failed to find libarchive library")
try:
    libarchive = ctypes.cdll.LoadLibrary(libarchive_path)
except:
    raise ImportError("Failed to load libarchive library from %s - are any dependencies missing?"
                      "Is your environment activated?")
if backup_path:
    os.environ['PATH'] = backup_path


# Constants

ARCHIVE_EOF = 1       # Found end of archive.
ARCHIVE_OK = 0        # Operation was successful.
ARCHIVE_RETRY = -10   # Retry might succeed.
ARCHIVE_WARN = -20    # Partial success.
ARCHIVE_FAILED = -25  # Current operation cannot complete.
ARCHIVE_FATAL = -30   # No more operations are possible.
REGULAR_FILE = 0o100000
DEFAULT_UNIX_PERMISSION = 0o664


# Callback types

WRITE_CALLBACK = CFUNCTYPE(
    c_ssize_t, c_void_p, c_void_p, POINTER(c_void_p), c_size_t
)
READ_CALLBACK = CFUNCTYPE(
    c_ssize_t, c_void_p, c_void_p, POINTER(c_void_p)
)
OPEN_CALLBACK = CFUNCTYPE(c_int, c_void_p, c_void_p)
CLOSE_CALLBACK = CFUNCTYPE(c_int, c_void_p, c_void_p)
VOID_CB = lambda *_: ARCHIVE_OK


# Type aliases, for readability

c_archive_p = c_void_p
c_archive_entry_p = c_void_p


# Helper functions

def _error_string(archive_p):
    msg = error_string(archive_p)
    if msg is None:
        return
    try:
        return msg.decode('ascii')
    except UnicodeDecodeError:
        return msg


def archive_error(archive_p, retcode):
    msg = _error_string(archive_p)
    raise ArchiveError(msg, errno(archive_p), retcode, archive_p)


def check_null(ret, func, args):
    if ret is None:
        raise ArchiveError(func.__name__+' returned NULL')
    return ret


def check_int(retcode, func, args):
    if retcode >= 0:
        return retcode
    elif retcode == ARCHIVE_WARN:
        logger.warning(_error_string(args[0]))
        return retcode
    else:
        raise archive_error(args[0], retcode)


def ffi(name, argtypes, restype, errcheck=None):
    f = getattr(libarchive, 'archive_'+name)
    f.argtypes = argtypes
    f.restype = restype
    if errcheck:
        f.errcheck = errcheck
    globals()[name] = f
    return f


# FFI declarations

# archive_util

errno = ffi('errno', [c_archive_p], c_int)
error_string = ffi('error_string', [c_archive_p], c_char_p)

# archive_entry

ffi('entry_new', [], c_archive_entry_p, check_null)

ffi('entry_filetype', [c_archive_entry_p], c_int)
ffi('entry_atime', [c_archive_entry_p], c_int)
ffi('entry_birthtime', [c_archive_entry_p], c_int)
ffi('entry_mtime', [c_archive_entry_p], c_int)
ffi('entry_ctime', [c_archive_entry_p], c_int)
ffi('entry_atime_nsec', [c_archive_entry_p], c_long)
ffi('entry_birthtime_nsec', [c_archive_entry_p], c_long)
ffi('entry_mtime_nsec', [c_archive_entry_p], c_long)
ffi('entry_ctime_nsec', [c_archive_entry_p], c_long)
ffi('entry_pathname', [c_archive_entry_p], c_char_p)
ffi('entry_pathname_w', [c_archive_entry_p], c_wchar_p)
ffi('entry_sourcepath', [c_archive_entry_p], c_char_p)
ffi('entry_size', [c_archive_entry_p], c_longlong)
ffi('entry_size_is_set', [c_archive_entry_p], c_int)
ffi('entry_mode', [c_archive_entry_p], c_int)
ffi('entry_strmode', [c_archive_entry_p], c_char_p)
ffi('entry_hardlink', [c_archive_entry_p], c_char_p)
ffi('entry_hardlink_w', [c_archive_entry_p], c_wchar_p)
ffi('entry_symlink', [c_archive_entry_p], c_char_p)
ffi('entry_symlink_w', [c_archive_entry_p], c_wchar_p)
ffi('entry_rdevmajor', [c_archive_entry_p], c_uint)
ffi('entry_rdevminor', [c_archive_entry_p], c_uint)
ffi('entry_uid', [c_archive_entry_p], c_longlong)
ffi('entry_gid', [c_archive_entry_p], c_longlong)

ffi('entry_set_size', [c_archive_entry_p, c_longlong], None)
ffi('entry_set_filetype', [c_archive_entry_p, c_uint], None)
ffi('entry_set_perm', [c_archive_entry_p, c_int], None)
ffi('entry_set_atime', [c_archive_entry_p, c_int, c_long], None)
ffi('entry_set_mtime', [c_archive_entry_p, c_int, c_long], None)
ffi('entry_set_ctime', [c_archive_entry_p, c_int, c_long], None)
ffi('entry_set_birthtime', [c_archive_entry_p, c_int, c_long], None)

ffi('entry_update_pathname_utf8', [c_archive_entry_p, c_char_p], None)

ffi('entry_clear', [c_archive_entry_p], c_archive_entry_p)
ffi('entry_free', [c_archive_entry_p], None)

# archive_read

ffi('read_new', [], c_archive_p, check_null)

READ_FORMATS = set((
    '7zip', 'all', 'ar', 'cab', 'cpio', 'empty', 'iso9660', 'lha', 'mtree',
    'rar', 'raw', 'tar', 'xar', 'zip', 'warc'
))
for f_name in list(READ_FORMATS):
    try:
        ffi('read_support_format_'+f_name, [c_archive_p], c_int, check_int)
    except AttributeError:  # pragma: no cover
        logger.info('read format "%s" is not supported' % f_name)
        READ_FORMATS.remove(f_name)

READ_FILTERS = set((
    'all', 'bzip2', 'compress', 'grzip', 'gzip', 'lrzip', 'lzip', 'lzma',
    'lzop', 'none', 'rpm', 'uu', 'xz', 'lz4', 'zstd'
))
for f_name in list(READ_FILTERS):
    try:
        ffi('read_support_filter_'+f_name, [c_archive_p], c_int, check_int)
    except AttributeError:  # pragma: no cover
        logger.info('read filter "%s" is not supported' % f_name)
        READ_FILTERS.remove(f_name)

ffi('read_open',
    [c_archive_p, c_void_p, OPEN_CALLBACK, READ_CALLBACK, CLOSE_CALLBACK],
    c_int, check_int)
ffi('read_open_fd', [c_archive_p, c_int, c_size_t], c_int, check_int)
ffi('read_open_filename_w', [c_archive_p, c_wchar_p, c_size_t],
    c_int, check_int)
ffi('read_open_memory', [c_archive_p, c_void_p, c_size_t], c_int, check_int)

ffi('read_next_header', [c_archive_p, POINTER(c_void_p)], c_int, check_int)
ffi('read_next_header2', [c_archive_p, c_void_p], c_int, check_int)

ffi('read_close', [c_archive_p], c_int, check_int)
ffi('read_free', [c_archive_p], c_int, check_int)

# archive_read_disk

ffi('read_disk_new', [], c_archive_p, check_null)
ffi('read_disk_set_behavior', [c_archive_p, c_int], c_int, check_int)
ffi('read_disk_set_standard_lookup', [c_archive_p], c_int, check_int)
ffi('read_disk_open', [c_archive_p, c_char_p], c_int, check_int)
ffi('read_disk_open_w', [c_archive_p, c_wchar_p], c_int, check_int)
ffi('read_disk_descend', [c_archive_p], c_int, check_int)

# archive_read_data

ffi('read_data_block',
    [c_archive_p, POINTER(c_void_p), POINTER(c_size_t), POINTER(c_longlong)],
    c_int, check_int)
ffi('read_data', [c_archive_p, c_void_p, c_size_t], c_ssize_t, check_int)
ffi('read_data_skip', [c_archive_p], c_int, check_int)

# archive_write

ffi('write_new', [], c_archive_p, check_null)
ffi('write_set_options', [c_archive_p, c_char_p], c_int, check_int)

ffi('write_disk_new', [], c_archive_p, check_null)
ffi('write_disk_set_options', [c_archive_p, c_int], c_int, check_int)

WRITE_FORMATS = set((
    '7zip', 'ar_bsd', 'ar_svr4', 'cpio', 'cpio_newc', 'gnutar', 'iso9660',
    'mtree', 'mtree_classic', 'pax', 'pax_restricted', 'shar', 'shar_dump',
    'ustar', 'v7tar', 'xar', 'zip', 'warc'
))
for f_name in list(WRITE_FORMATS):
    try:
        ffi('write_set_format_'+f_name, [c_archive_p], c_int, check_int)
    except AttributeError:  # pragma: no cover
        logger.info('write format "%s" is not supported' % f_name)
        WRITE_FORMATS.remove(f_name)

WRITE_FILTERS = set((
    'b64encode', 'bzip2', 'compress', 'grzip', 'gzip', 'lrzip', 'lzip', 'lzma',
    'lzop', 'uuencode', 'xz', 'lz4', 'zstd'
))
for f_name in list(WRITE_FILTERS):
    try:
        ffi('write_add_filter_'+f_name, [c_archive_p], c_int, check_int)
    except AttributeError:  # pragma: no cover
        logger.info('write filter "%s" is not supported' % f_name)
        WRITE_FILTERS.remove(f_name)

ffi('write_open',
    [c_archive_p, c_void_p, OPEN_CALLBACK, WRITE_CALLBACK, CLOSE_CALLBACK],
    c_int, check_int)
ffi('write_open_fd', [c_archive_p, c_int], c_int, check_int)
ffi('write_open_filename', [c_archive_p, c_char_p], c_int, check_int)
ffi('write_open_filename_w', [c_archive_p, c_wchar_p], c_int, check_int)
ffi('write_open_memory',
    [c_archive_p, c_void_p, c_size_t, POINTER(c_size_t)],
    c_int, check_int)

ffi('write_get_bytes_in_last_block', [c_archive_p], c_int, check_int)
ffi('write_get_bytes_per_block', [c_archive_p], c_int, check_int)
ffi('write_set_bytes_in_last_block', [c_archive_p, c_int], c_int, check_int)
ffi('write_set_bytes_per_block', [c_archive_p, c_int], c_int, check_int)

ffi('write_header', [c_archive_p, c_void_p], c_int, check_int)
ffi('write_data', [c_archive_p, c_void_p, c_size_t], c_ssize_t, check_int)
ffi('write_data_block', [c_archive_p, c_void_p, c_size_t, c_longlong],
    c_int, check_int)
ffi('write_finish_entry', [c_archive_p], c_int, check_int)

ffi('write_fail', [c_archive_p], c_int, check_int)

ffi('write_close', [c_archive_p], c_int, check_int)
ffi('write_free', [c_archive_p], c_int, check_int)

# library version
ffi('version_number', [], c_int, check_int)