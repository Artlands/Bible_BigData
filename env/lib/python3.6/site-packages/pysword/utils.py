# -*- coding: utf-8 -*-
###############################################################################
# PySword - A native Python reader of the SWORD Project Bible Modules         #
# --------------------------------------------------------------------------- #
# Copyright (c) 2008-2018 Various developers:                                 #
# Kenneth Arnold, Joshua Gross, Tomas Groth, Ryan Hiebert, Philip Ridout,     #
# Matthew Wardrop                                                             #
# --------------------------------------------------------------------------- #
# This program is free software; you can redistribute it and/or modify it     #
# under the terms of the GNU General Public License as published by the Free  #
# Software Foundation; version 2 of the License.                              #
#                                                                             #
# This program is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    #
# more details.                                                               #
#                                                                             #
# You should have received a copy of the GNU General Public License along     #
# with this program; if not, write to the Free Software Foundation, Inc., 51  #
# Franklin St, Fifth Floor, Boston, MA 02110-1301 USA                         #
###############################################################################
import os
import sys
try:
    import pathlib
    pathlib_available = True
except ImportError:
    pathlib_available = False

PY3 = sys.version_info > (3,)


def path_like_to_str(path):
    """
    Take an object and convert it to a string representation of a file path.

    :param path: The object to convert, should be an `os.PathLike` object, a `pathlib.Path` object or a str object
    :return: The string representation of the path
    """
    if PY3:
        if isinstance(path, str):
            return path
        if hasattr(path, '__fspath__'):
            # py 3.6 and above implemented os.PathLike objects, which make use if the __fspath__ method.
            return os.fspath(path)
        if pathlib_available and isinstance(path, pathlib.Path):
            # py 3.4 and 3.5 have the pathlib.Path object, but it doesn't have the fspath interface.
            return str(path)
    else:
        if isinstance(path, unicode):
            return path
    raise TypeError
