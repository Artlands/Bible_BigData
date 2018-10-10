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

# Util to convert SWORD canon header files into pysword format.
# Place the script in the same folder as the canon header files and run like this:
# python3 canon-parser.py > canons.py


def parse_canon_header(canon_name, canon_filename, otbook_struct_filename, ntbook_struct_filename):
    # Open file
    canon_header_file = open(canon_filename, 'rt')
    fulltext = canon_header_file.read()
    canon_header_file.close()
    if otbook_struct_filename:
        otbook_struct_file = open(otbook_struct_filename, 'rt')
        otbook_struct_fulltext = otbook_struct_file.read()
        otbook_struct_file.close()
    else:
        otbook_struct_fulltext = fulltext
    if ntbook_struct_filename:
        ntbook_struct_file = open(ntbook_struct_filename, 'rt')
        ntbook_struct_fulltext = ntbook_struct_file.read()
        ntbook_struct_file.close()
    else:
        ntbook_struct_fulltext = fulltext
    # Detect if OT books are listed
    otbooks_pos = otbook_struct_fulltext.find('struct sbook otbooks')
    if otbooks_pos > 0:
        # Find the declaration of the OT struct, first instance of "struct sbook otbooks[] = {",
        # but we just search for the last part
        ot_struct_start = otbook_struct_fulltext.find('= {', otbooks_pos) + 4
        # Find end of OT struct
        ot_struct_end = otbook_struct_fulltext.find('};', ot_struct_start)
        # Extract OT struct
        ot_struct = otbook_struct_fulltext[ot_struct_start:ot_struct_end]
    else:
        ot_struct = ''
    # Detect if NT books are listed
    ntbooks_pos = ntbook_struct_fulltext.find('struct sbook ntbooks')
    if ntbooks_pos > 0:
        # Find start NT struct
        nt_struct_start = ntbook_struct_fulltext.find('= {', ntbooks_pos) + 4
        # Find end of NT struct
        nt_struct_end = ntbook_struct_fulltext.find('};', nt_struct_start)
        # Extract NT struct
        nt_struct = ntbook_struct_fulltext[nt_struct_start:nt_struct_end]
    else:
        nt_struct = ''
    verse_struct_loc = fulltext.find('int vm')
    # Find start verse number struct
    verse_struct_start = fulltext.find('= {', verse_struct_loc) + 4
    # Find end verse number struct
    verse_struct_end = fulltext.find('};', verse_struct_start)
    # Extract verse struct
    verse_struct = fulltext[verse_struct_start:verse_struct_end]
    # Convert/evaluate the ot and nt structs into python
    ot = eval('[' + ot_struct.replace('{', '[').replace('}', ']') + ']')
    nt = eval('[' + nt_struct.replace('{', '[').replace('}', ']') + ']')
    # Convert/evaluate the verse struct into python
    verses_per_chapter = eval('[' + verse_struct.replace('//', '#') + ']')
    # Print the structure in the format pysword uses
    idx = 0
    print('u%r : {' % canon_name)
    for testament, contents in (('ot', ot), ('nt', nt)):
        print('u%r: [' % testament)
        for num, (name, osis, pref_abbr, num_chapters) in enumerate(contents):
            new_idx = idx + num_chapters
            if name:
                print('(u%r, u%r, u%r, %r),' % (name, osis, pref_abbr, verses_per_chapter[idx:new_idx]))
            idx = new_idx
        print('],')
    print('},')


if __name__ == '__main__':
    print(u'canons = {')
    # The canons and where to get ot, nt booklists. None means the file itself has it.
    canons = [
        (u'kjv', u'canon.h', None, None),
        (u'calvin', u'canon_calvin.h', u'canon.h', u'canon.h'),
        (u'catolic', u'canon_catholic.h', None, u'canon.h'),
        (u'catolic2', u'canon_catholic2.h', None, u'canon.h'),
        (u'darbyfr', u'canon_darbyfr.h', u'canon.h', u'canon.h'),
        (u'german', u'canon_german.h', None, u'canon.h'),
        (u'kjva', u'canon_kjva.h', None, u'canon.h'),
        (u'leningrad', u'canon_leningrad.h', None, None),
        (u'luther', u'canon_luther.h', None, None),
        (u'lxx', u'canon_lxx.h', None, u'canon.h'),
        (u'mt', u'canon_mt.h', None, None),
        (u'nrsv', u'canon_nrsv.h', u'canon.h', u'canon.h'),
        (u'nrsva', u'canon_nrsva.h', None, u'canon.h'),
        (u'orthodox', u'canon_orthodox.h', None, u'canon.h'),
        (u'segond', u'canon_segond.h', u'canon.h', u'canon.h'),
        (u'synodal', u'canon_synodal.h', None, None),
        (u'synodalprot', u'canon_synodalprot.h', None, u'canon_synodal.h'),
        (u'vulg', u'canon_vulg.h', None, None)
    ]
    for canon in canons:
        parse_canon_header(*canon)
    print(u'}')
