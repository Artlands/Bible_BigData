#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time

class CreateData:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        creator = CreateData()
        creator.create()

    def create(self):

        print (f"Creating data for analysis")
        start = time.time()

        out_file = self.get_new_file()
        result = {}
        books = []
        verses = []
        for line in self.in_file:
            data = json.loads(line)
            for book in data['books']:
                text=''
                books.append(book['name'])
                for chapter in book['chapters']:
                    for verse in chapter['verses']:
                        text = text + verse['text']
                verses.append(text)

        result['books'] = books
        result['verses'] = verses

        print(f"Books number is {len(books)}")

        out_file.write('%s\n' % json.dumps(result))
        out_file.close()

        end = time.time()
        print (f"Created data for analysis in one json file. it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data{self.file_ext}"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file {new_file_path}")
        return open(new_file_path, "w")

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name = argv[1]
            self.in_file = open(self.file_name, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Seperate file into several documents

        Usage:

            $ python create_data.py <file_name>

        """

if __name__ == "__main__":
    CreateData.run()
