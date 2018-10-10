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
import io
import os
import zipfile
import tempfile
import shutil
import sys
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pysword.bible import SwordBible
from pysword.utils import path_like_to_str

PY3 = sys.version_info > (3,)


class SwordModules(object):
    """
    Class used to parse module conf-files.
    """
    def __init__(self, path=None, encoding=None):
        """
        Initialize the SwordModules object.

        :param path: Path the SWORD datapath or to a zip-file containing a module. Defaults to the platforms expected
                     SWORD datapath.
        :param encoding: The encoding to use when reading the text.
        """
        self._encoding = encoding
        if path is None:
            # Based on the platform find the SWORD data path
            if sys.platform.startswith(u'win32'):
                self._sword_path = os.path.join(os.getenv(u'APPDATA'), u'Sword')
            elif sys.platform.startswith(u'darwin'):
                self._sword_path = os.path.join(os.getenv(u'HOME'), u'Library', u'Application Support', u'Sword')
            else:  # Linux etc.
                self._sword_path = os.path.join(os.getenv(u'HOME'), u'.sword')
        else:
            try:
                self._sword_path = path_like_to_str(path)
            except TypeError:
                raise TypeError(u'`path` should be a str, PathLike object or an instance of pathlib.Path')

        self._modules = {}
        self._temp_folder = None

    def __del__(self):
        """
        Clean up. If we decompressed a zip-file remove the files again.
        """
        if self._temp_folder:
            # TODO: This sometimes fails because it is getting GC'ed before the files in the bible module are closed!
            shutil.rmtree(self._temp_folder, ignore_errors=True)

    def parse_modules(self):
        """
        Based on the datapath given to the constructor parse modules conf-files and return the result

        :return: A dict containing the data read from the conf files.
        """
        # If path is a zipfile, we extract it to a temp-folder
        if self._sword_path.endswith(u'.zip'):
            self._temp_folder = tempfile.mkdtemp()
            zipped_module = zipfile.ZipFile(self._sword_path)
            zipped_module.extractall(self._temp_folder)
            conf_folder = os.path.join(self._temp_folder, u'mods.d')
        else:
            conf_folder = os.path.join(self._sword_path, u'mods.d')
        # Loop over config files and save data in a dict
        for f in os.listdir(conf_folder):
            if f.endswith(u'.conf'):
                conf_filename = os.path.join(conf_folder, f)
                if PY3:
                    config = configparser.ConfigParser(strict=False)
                    config_file_reader = config.read_file
                else:
                    config = configparser.SafeConfigParser()
                    config_file_reader = config.readfp
                # Config files are either utf-8 or iso-8859-1 (aka Latin-1) encoded
                try:
                    if self._encoding is None:
                        # No encoding specified try uft-8
                        try:
                            with io.open(conf_filename, mode=u'rt', encoding=u'utf-8', errors=u'strict') as conf_file:
                                config_file_reader(conf_file)
                        except UnicodeDecodeError:
                            # Fallback to iso=8859-1 (latin-1)
                            self._encoding = u'iso-8859-1'
                    if self._encoding:
                        with io.open(conf_filename, mode=u'rt', encoding=self._encoding, errors=u'strict') as conf_file:
                            config_file_reader(conf_file)
                except Exception as e:
                    print(u'Exception while parsing %s\n%s' % (f, e))
                    continue
                module_name = config.sections()[0]
                self._modules[module_name] = dict(config._sections[module_name])
        # Return the modules metadata
        return self._modules

    def get_bible_from_module(self, module_key):
        """
        Return a SwordBible object for the key given.

        :param module_key: The key to use for finding the module.
        :return: a SwordBible object for the key given.
        """
        # TODO could replace the try excepts with a simple.get
        bible_module = self._modules[module_key]
        if self._temp_folder:
            module_path = os.path.join(self._temp_folder, bible_module[u'datapath'])
        else:
            module_path = os.path.join(self._sword_path, bible_module[u'datapath'])
        module_type = bible_module[u'moddrv'].lower()
        try:
            module_versification = bible_module[u'versification'].lower()
        except KeyError:
            module_versification = u'kjv'
        try:
            module_encoding = bible_module[u'encoding'].lower()
        except KeyError:
            module_encoding = None
        try:
            source_type = bible_module[u'sourcetype']
        except KeyError:
            source_type = None
        return SwordBible(module_path, module_type, module_versification, module_encoding, source_type)
