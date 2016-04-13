# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:03:51 2016

@author: belinkov
"""

# convert raw text (in Buckwalter) to extracted format, ready for preparation to currennt

import sys


def convert_text(text_filename, output_filename):
    """ Convert text to extracted format
        
    text_filename: file with raw text, once sentence per line, no ids, all in Buckwalter
    output_filename: file to write output, one word per line, sentences separated by blank line
    """

    first_line = True    
    with open(output_filename, 'w') as g:
        with open(text_filename) as f:
            for line in f:
                if first_line:
                    first_line = False
                else:
                    g.write('\n')
                for word in line.strip().split():
                    g.write(word + '\n')
    print 'written extracted data to:', output_filename


if __name__ == '__main__':
    if len(sys.argv) == 3:
        convert_text(sys.argv[1], sys.argv[2])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <input file> <output file>'
        print 'input file should be one sentence per line, in Buckwalter'
        print 'output file will be one word per line, sentences separated by blank line'
                
