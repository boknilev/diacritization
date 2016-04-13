# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:41:07 2016

@author: belinkov
"""

# convert predictions to raw text

import sys, codecs
from data_utils import DIACS, MADA_LATIN_TAG
from utils import bw2utf8


def convert(pred_filename, output_filename, orig_filename_with_lat_mark=None):
    """
    Convert predictions to raw text
    
    pred_filename: file containing diacritization predictions, each line is one sentence
                    each line starts with an id, then list of word:word_diac pairs
    output_filename: file to write diacritized text, one sentence per line
    orig_filename_with_lat_mark: optionally, file containing the original text, 
                    with Latin words marked with @@LAT@@
                    if given, will use original Latin words in output
    Note: all files are expected to be in Buckwalter format
    """
    
    pred_lines = open(pred_filename).readlines()
    orig_lines_with_lat = None
    if orig_filename_with_lat_mark:
        orig_lines_with_lat = open(orig_filename_with_lat_mark).readlines()
        assert len(pred_lines) == len(orig_lines_with_lat), 'incompatible pred and orig (with lat) files'
    with open(output_filename, 'w') as g:
        with codecs.open(output_filename + '.utf8', 'w', encoding='utf8') as g_utf8:
            for i in xrange(len(pred_lines)):
                pred_splt = pred_lines[i].strip().split()
                pairs = pred_splt[1:]
                pred_words_diac = []
                for pair in pairs:
                    pair_splt = pair.split(':')
                    # handle words containing ':'
                    if len(pair_splt) != 2:
                        if pair == ':::':
                            word, pred_word_diac = ':', ':'
                        else:
                            assert False, 'bad pair ' + pair + ' in line ' + str(i)
                    else:
                        word, pred_word_diac = pair_splt
                    pred_words_diac.append(pred_word_diac)
                utf8_str = bw2utf8(' '.join(pred_words_diac))
                pred_words_diac_utf8 = utf8_str.split()
                #print pred_words_diac_utf8 
                # handle Latin marking
                if orig_lines_with_lat:
                    orig_words_with_lat = orig_lines_with_lat[i].strip().split()
                    print 'line:',  i
                    if len(pred_words_diac) != len(orig_words_with_lat):
                        print 'incompatible pred and orig in line ' + str(i)
                        print 'pred_words_diac:', pred_words_diac
                        print 'orig_words_with_lat:', orig_words_with_lat
                        print 'trying to recover by restoring latin words'
                        for j in xrange(len(orig_words_with_lat)):
                            # a latin word that is only once char, which is a diacritic, 
                            # might have been omitted 
                            if orig_words_with_lat[j].startswith(MADA_LATIN_TAG) \
                                    and orig_words_with_lat[j][7:] in DIACS:
                                pred_words_diac.insert(j, orig_words_with_lat[j])
                                pred_words_diac_utf8.insert(j, orig_words_with_lat[j])
                    for j in xrange(len(pred_words_diac)):
                        if orig_words_with_lat[j].startswith(MADA_LATIN_TAG):
                            pred_words_diac[j] = orig_words_with_lat[j][7:]
                            pred_words_diac_utf8[j] = orig_words_with_lat[j][7:]
                g.write(' '.join(pred_words_diac) + '\n')
                g_utf8.write(' '.join(pred_words_diac_utf8) + '\n')
    print 'written predicted diacritized words to:', output_filename


if __name__ == '__main__':
    if len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <pred file> <output file> [<orig file with latin marking>]'
        
  
    
    
    
    