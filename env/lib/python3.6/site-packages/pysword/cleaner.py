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

import re


class Cleaner(object):
    def __init(self):
        pass

    def clean(self, text):
        pass


class OSISCleaner(Cleaner):
    """
    Class to clean text of OSIS tags. OSIS spec can be found here: http://www.bibletechnologies.net/
    """
    def __init__(self):
        self.__setup()

    def __setup(self):
        """
        Compile regular expressions that will be used to remove OSIS tags.
        Not all OSIS tags are "mentioned" here since we should only have to deal with those that
        can be found in the biblical texts.
        """
        remove_content_tagnames = [r'note', r'milestone', r'title', r'abbr', r'catchWord', r'index', r'rdg',
                                   r'rdgGroup', r'figure']
        self.__remove_content_regexes = []
        for tag_name in remove_content_tagnames:
            tag_regex = r'<' + tag_name + r'.*?' + tag_name + r'>'
            self.__remove_content_regexes.append(re.compile(tag_regex, re.IGNORECASE))
            single_tag_regex = r'<' + tag_name + r'[^<]*/>'
            self.__remove_content_regexes.append(re.compile(single_tag_regex, re.IGNORECASE))

        keep_content_tagnames = [r'p', r'l', r'lg', r'q', r'a', r'w', r'divineName', r'foreign', r'hi', r'inscription',
                                 r'mentioned', r'name', r'reference', r'seg', r'transChange', r'salute', r'signed',
                                 r'closer', r'speech', r'speaker', r'list', r'item', r'table', r'head', r'row', r'cell',
                                 r'caption', r'chapter', r'div']
        self.__keep_content_regexes = []
        for tag_name in keep_content_tagnames:
            tag_regex = r'<' + tag_name + r'.*?>(.*?)</' + tag_name + r'>'
            self.__keep_content_regexes.append(re.compile(tag_regex, re.IGNORECASE))
            # Just remove if tag appear in single form
            single_tag_regex = r'<' + tag_name + r'[^<]*/>'
            self.__remove_content_regexes.append(re.compile(single_tag_regex, re.IGNORECASE))

    def clean(self, text):
        """
        Clean text for OSIS tags.

        :param text: The text to be cleaned
        :return: The cleaned text is returned
        """
        text = re.sub(r'(<[^\>]+type="x-br"[^\>]+\>)', r'\1 ', text)
        for regex in self.__remove_content_regexes:
            text = regex.sub(u'', text)
        for regex in self.__keep_content_regexes:
            text = regex.sub(r'\1', text)
        return text


class GBFCleaner(Cleaner):
    """
    Class to clean text of GBF tags. GBF spec can be found here: http://ebible.org/bible/gbf.htm
    """
    def __init__(self):
        self.__setup()

    def __setup(self):
        """
        Compile regular expressions that will be used to remove GBF 'tags'.
        Not all GBF tags are "mentioned" here since we should only have to deal with those that
        can be found in the biblical texts.
        """

        remove_content_tags = [r'<TB>.*?<Tb>', r'<TC>.*?<Tc>', r'<TH>.*?<Th>', r'<TS>.*?<Ts>', r'<TT>.*?<Tt>',
                               r'<TN>.*?<Tn>', r'<TA>.*?<Ta>', r'<TP>.*?<Tp>',
                               r'<FB>.*?<fb>', r'<FC>.*?<fc>', r'<FI>.*?<fi>', r'<FN.*?>.*?<fn>', r'<FO>.*?<fo>',
                               r'<FR>.*?<fr>', r'<FS>.*?<fs>', r'<FU>.*?<fu>', r'<FV>.*?<fv>',
                               r'<RF>.*?<Rf>', r'<RB>', r'<RP.*?>', r'<Rp.*?>', r'<RX.*?>', r'<Rx.*?>',
                               r'<H.*?>',
                               r'<B.*?>',
                               r'<ZZ>',
                               r'<D.*?>', r'<J.*?>', r'<P.>',
                               r'<W.*?>',
                               r'<S.*?>',
                               r'<N.*?>',
                               r'<C.>']
        self.__remove_content_regexes = []
        for tag in remove_content_tags:
            self.__remove_content_regexes.append(re.compile(tag))

    def clean(self, text):
        """
        Clean text for GBF tags.

        :param text: The text to be cleaned
        :return: The cleaned text is returned
        """
        for regex in self.__remove_content_regexes:
            text = regex.sub(u'', text)
        # TODO: Support special char tags <CAxx> and <CUxxxx>
        return text


class ThMLCleaner(Cleaner):
    """
    Class to clean text of ThML tags. ThML spec can be found here: https://www.ccel.org/ThML/
    """
    def __init__(self):
        self.__setup()

    def __setup(self):
        """
        Compile regular expressions that will be used to remove ThML tags.
        Not all ThML tags are "mentioned" here since we should only have to deal with those that
        can be found in the biblical texts.
        """

        remove_content_tags = [r'<scripRef.*?>.*?</scripRef>', r'<scripCom.*?>.*?</scripCom>', r'<.*?>']
        self.__remove_content_regexes = []
        for tag in remove_content_tags:
            self.__remove_content_regexes.append(re.compile(tag))

    def clean(self, text):
        """
        Clean text for ThML tags.

        :param text: The text to be cleaned
        :return: The cleaned text is returned
        """
        for regex in self.__remove_content_regexes:
            text = regex.sub(u'', text)
        return text
