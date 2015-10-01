__author__ = 'belinkov'

import sys
import os
import codecs
import re
from data_utils import DIACS, REGEX_DIACS


REGEX_SOLUTION_DIAC = re.compile(r'\((.+?)\)')  # for gold diacritized word


class WordAnalysis(object):
    """
    A simplified pos analysis from treebank pos/before-treebank files.

    Attributes:
        input_string (str): INPUT STRING from LDC file
        lookup_word (str): LOOK-UP WORD from LDC file (if exists)
        comment (str): Comment from LDC file
        index (str): INDEX from LDC file
        gold_solution (str): the gold * SOLUTION from LDC file
        word (str): for Arabic words, same as lookup_word with diacritics removed;
                    for non-Arabic words, same as input_string
        word_diac (str): for Arabic words, the diacritized lookup_word from gold_solution;
                         for non-Arabic words, same as input_string
    """

    def __init__(self, input_string, comment, index, gold_solution=None, lookup_word=None):

        self.input_string = input_string
        self.comment = comment
        self.index = index
        self.gold_solution = gold_solution
        self.lookup_word = lookup_word
        # if this is an Arabic script word
        if lookup_word:
            self.word = REGEX_DIACS.sub('', lookup_word)
            if gold_solution:
                match = REGEX_SOLUTION_DIAC.match(gold_solution)
                if not match:
                    sys.stderr.write('Warning: could not find diacritized solution in: ' + gold_solution + '. ' + \
                                     'Writing lookup word as is: ' + lookup_word + '\n')
                    self.word_diac = lookup_word
                else:
                    self.word_diac = match.groups()[0]
                    self.check_match()

            # there may be no solution if the word is unknown, so just write the lookup word
            else:
                self.word_diac = lookup_word

        # this is a non-Arabic script word
        else:
            # TODO consider marking as Lating words (and exclude later)
            self.word = input_string
            self.word_diac = input_string

    def check_match(self):
        """
        Check match between word and word_diac
        """

        if REGEX_DIACS.sub('', self.word_diac) != self.word:
            sys.stderr.write('Warning: word ' + self.word + ' != word_diac ' + self.word_diac + \
                             ' after removing diacritics. Attempting to correct\n')
            self.unnormalize()
        if REGEX_DIACS.sub('', self.word_diac) != self.word:
            sys.stderr.write('Warning: could not correct, word ' + self.word + ' != word_diac ' + \
                             self.word_diac + '. Using undiacritized word_diac as word.\n')
            self.word = REGEX_DIACS.sub('', self.word_diac)
        if REGEX_DIACS.sub('', self.word_diac) != self.word:
            sys.stderr.write('Warning: still word ' + self.word + ' != word_diac ' + self.word_diac + '\n')

    def unnormalize(self):
        """
        Try to reverse Buckwalter normalizations on diacritized word
        """

        # first, remove "_" (elongation character)
        self.word = self.word.replace('_', '')
        self.word_diac = self.word_diac.replace('_', '')

        # next, check for normalization mismatches
        word_ind = 0
        word_diac_ind = 0
        new_word_diac = ''
        while word_ind < len(self.word) and word_diac_ind < len(self.word_diac):
            word_char = self.word[word_ind]
            word_diac_char = self.word_diac[word_diac_ind]
            if word_char == word_diac_char:
                new_word_diac += word_diac_char
                word_ind += 1
                word_diac_ind += 1
            elif word_diac_char in DIACS:
                new_word_diac += word_diac_char
                word_diac_ind += 1
            else:
                # this is probably a normalization
                # print 'word_char:', word_char, 'word_diac_char:', word_diac_char
                new_word_diac += word_char
                word_ind += 1
                word_diac_ind += 1
        if word_ind == len(self.word) and word_diac_ind == len(self.word_diac) - 1:
            # if we have one more char in word_diac
            word_diac_char = self.word_diac[word_diac_ind]
            if word_diac_char in DIACS:
                new_word_diac += word_diac_char

        self.word_diac = new_word_diac


def process_treebank_file(treebank_filename, output_file, output_file_diac):
    """
    Extract data from a treebank file

    :param treebank_filename: pos/before-treebank file
    :param output_file: file to write words without diacritics
    :param output_file_diac: file to write words with diacritics
    :return:
    """

    print 'extracting data from file:', treebank_filename
    f = codecs.open(treebank_filename, encoding='utf8')

    input_string, comment, index, gold_solution, lookup_word = ['']*5
    prev_index = ''  # keep track of previous index
    for line in f:
        if line.strip() == '':
            if input_string == '':
                continue
            word_analysis = WordAnalysis(input_string, comment, index, gold_solution, lookup_word)
            # check for a new paragraph
            if prev_index.startswith('P') and index.startswith('P') and not prev_index.startswith(index.split('W')[0]):
                output_file.write('\n')
                output_file_diac.write('\n')
            output_file.write(word_analysis.word + '\n')
            output_file_diac.write(word_analysis.word_diac + '\n')
            prev_index = index
            input_string, comment, index, gold_solution, lookup_word = ['']*5
        else:
            splt = line.strip().split(':', 1)
            if len(splt) != 2:
                sys.stderr.write('Warning: could not split line on :, in: ' + line + '\n')
                continue
            field_name, field_val = splt[0].strip(), splt[1].strip()
            if field_name == 'INPUT STRING':
                input_string = field_val
            elif field_name == 'LOOK-UP WORD':
                lookup_word = field_val
            elif field_name == 'Comment':
                comment = field_val
            elif field_name == 'INDEX':
                index = field_val
            elif field_name.startswith('* SOLUTION'):
                gold_solution = field_val
            elif field_name.startswith('SOLUTION') or field_name == '(GLOSS)':
                continue
            else:
                sys.stderr.write('Warning: unkown field: ' + field_name + '\n')

    f.close()


def process_dir(treebank_dir, output_filename, output_filename_diac):
    """
    Extract data from a treebank dir

    :param treebank_dir: pos/before-treebank directory
    :param output_file: file to write words without diacritics
    :param output_file_diac: file to write words with diacritics
    :return:
    """

    print 'processing treebank dir:', treebank_dir
    g = codecs.open(output_filename, 'w', encoding='utf8')
    g_diac = codecs.open(output_filename_diac, 'w', encoding='utf8')

    for f in os.listdir(treebank_dir):
        process_treebank_file(treebank_dir + '/' + f, g, g_diac)

    g.close()
    g_diac.close()
    print 'written words to:', output_filename
    print 'written diacritized words to:', output_filename_diac


if __name__ == '__main__':

    if len(sys.argv) == 4:
        process_dir(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <treebank dir> <output word file> <output diacritized word file>'
