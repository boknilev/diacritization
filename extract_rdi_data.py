__author__ = 'belinkov'

import sys
from data_utils import DIACS, REGEX_DIACS, MADA_LATIN_TAG


def extract_data(rdi_bw_filename, output_word_filename, output_word_diac_filename):
    """
    Extract data from an RDI file

    :param rdi_bw_filename: file containing raw Arabic text, preprocessed by MADA preprocessor (keeping diacritics)
    :param output_word_filename: file to write words without diacritics
    :param output_word_diac_filename: file to wrote words with diacritics
    :return:
    """

    print 'extracting data from:', rdi_bw_filename
    g_word = open(output_word_filename, 'w')
    g_word_diac = open(output_word_diac_filename, 'w')
    with open(rdi_bw_filename) as f:
        for line in f:
            for token in line.strip().split():
                if token.startswith(MADA_LATIN_TAG):
                    sys.stderr.write('Warning: found Latin word: ' + token + '. skipping word.\n')
                    continue
                word_str = REGEX_DIACS.sub('', token)
                word_diac_str = token
                if word_str == '' or word_diac_str == '':
                    sys.stderr.write('Warning: empty word_str ' + word_str + ' or word_diac_str ' + word_diac_str + \
                                     '. skipping word.\n')
                    continue
                g_word.write(word_str + '\n')
                g_word_diac.write(word_diac_str + '\n')
            g_word.write('\n')
            g_word_diac.write('\n')
    g_word.close()
    g_word_diac.close()

    print 'written words to file:', output_word_filename
    print 'written words diac to file:', output_word_diac_filename


if __name__ == '__main__':

    if len(sys.argv) == 4:
        extract_data(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <rdi bw file> <word output file> <word diac output file>'

