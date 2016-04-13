__author__ = 'belinkov'

import re
import sys
import os
import numpy as np

REGEX_DIACS = re.compile(r'[iauo~FNK`]+')
REGEX_DIACS_NOSHADDA = re.compile(r'[iauoFNK`]+')
DIACS = {'i', 'a', 'u', 'o', '~', 'F', 'N', 'K', '`'}
DIACS_NOSHADDA = {'i', 'a', 'u', 'o', 'F', 'N', 'K', '`'}
PUNCS_STOP = {'!', '.', ':', ';', '?', '-'}  # punctuation for heuristic sentence stop
MADA_LATIN_TAG = '@@LAT@@'
SHADDA = '~'


class Word(object):
    """
    A class for a single word with its diacritics

    word (str): a word without diacritics extracted from the Treebank
    word_diac (str): a word with diacritics extracted from the Treebank
    letters (list): a list of characters representing the word
    diacs (list): a list of 0-2 diacritics for each letter
    shadda (str): strategy for dealing with shadda
    """

    WORD_BOUNDARY = '_###_'
    SHADDA_WITH_NEXT = 'with_next'
    SHADDA_IGNORE = 'ignore'
    SHADDA_ONLY = 'only'

    def __init__(self, word, word_diac, shadda=SHADDA_WITH_NEXT):

        self.diacs = []
        self.letters = []
        self.word = word
        self.word_diac = word_diac
        self.make_labels(shadda)

    def __str__(self):

        return 'word: ' + self.word + ' word_diac: ' + self.word_diac

    def make_labels(self, shadda=SHADDA_WITH_NEXT):
        """
        Make labels for the word
        shadda: 'with_next' will create a new label for each pair of seen shadda+vowel
                       'ignore' will ignore all shadda occurrences
                       'only' will create only shadda labels and ignore all other diacritics
        return:
        """

        if self.word != REGEX_DIACS.sub('', self.word_diac):
            # TODO consider ignoring Latin words
            sys.stderr.write('Warning: word ' + self.word + ' != word_diac ' + self.word_diac + ' after removing diacritics. Will use all chars as letters\n')
            self.letters = list(self.word_diac)
            self.diacs = ['']*len(self.letters)
        else:
            if shadda == Word.SHADDA_WITH_NEXT:
                # check all letters except for the last
                for i in xrange(len(self.word_diac)-1):
                    if self.word_diac[i] in DIACS:
                        continue
                    self.letters.append(self.word_diac[i])
                    # there may be another diacritic after shadda, so add both under the 'with_next' shadda strategy
                    if self.word_diac[i+1] == SHADDA and i < len(self.word_diac)-2 and self.word_diac[i+2] in DIACS:
                        self.diacs.append(self.word_diac[i+1] + self.word_diac[i+2])
                    # normally just choose the diacritic following the letter
                    elif self.word_diac[i+1] in DIACS:
                        self.diacs.append(self.word_diac[i+1])
                    # if there's no diacritic, choose an empty string
                    else:
                        self.diacs.append('')
                if self.word_diac[-1] not in DIACS:
                # if the last letter is not a diacritic, add it as well
                    self.letters.append(self.word_diac[-1])
                    self.diacs.append('')
            elif shadda == Word.SHADDA_IGNORE:
                for i in xrange(len(self.word_diac)-1):
                    if self.word_diac[i] in DIACS:
                        continue
                    self.letters.append(self.word_diac[i])
                    # there may be another diacritic after shadda, so add only that diacritic under the 'ignore' shadda strategy
                    if self.word_diac[i+1] == SHADDA and i < len(self.word_diac)-2 and \
                                    self.word_diac[i+2] in DIACS and self.word_diac[i+2] != SHADDA:
                        self.diacs.append(self.word_diac[i+2])
                    # add non-shadda diacritic following the letter
                    elif self.word_diac[i+1] in DIACS and self.word_diac[i+1] != SHADDA:
                        self.diacs.append(self.word_diac[i+1])
                    # if there's no non-shadda diacritic, choose an empty string
                    else:
                        self.diacs.append('')
                if self.word_diac[-1] not in DIACS:
                # if the last letter is not a diacritic, add it as well
                    self.letters.append(self.word_diac[-1])
                    self.diacs.append('')
            elif shadda == Word.SHADDA_ONLY:
                for i in xrange(len(self.word_diac)-1):
                    if self.word_diac[i] in DIACS:
                        continue
                    self.letters.append(self.word_diac[i])
                    # under the 'only' shadda strategy, add only shadda and ignore all other diacritics
                    if self.word_diac[i+1] == SHADDA:
                        self.diacs.append(self.word_diac[i+1])
                    else:
                        self.diacs.append('')
                if self.word_diac[-1] not in DIACS:
                 # if the last letter is not a diacritic, add it as well
                    self.letters.append(self.word_diac[-1])
                    self.diacs.append('')
            else:
                sys.stderr.write('Error: unknown shadda strategy \"' + shadda + '\" in make_labels()\n')
                return

        if len(self.letters) != len(self.diacs):
            sys.stderr.write('Error: incompatible lengths of letters and diacs in word: [ ' + str(self) + ' ]\n')

    @property
    def num_letters(self):

        return len(self.letters)


class KaldiWord(Word):
    """
    A class for a word in Kaldi data
    """

    def __init__(self, word, word_diac, shadda=Word.SHADDA_WITH_NEXT):

        self.diacs = []
        self.letters = []
        self.word = word
        self.word_diac = word_diac
        if REGEX_DIACS.sub('', self.word_diac) != self.word:
            sys.stderr.write('Warning: word ' + self.word + ' != word_diac ' + self.word_diac + \
                             ' after removing diacritics. Attempting to correct\n')
            self.unnormalize()
        if REGEX_DIACS.sub('', self.word_diac) != self.word:
            sys.stderr.write('Warning: could not correct, word ' + self.word + ' != word_diac ' + \
                             self.word_diac + '. Using undiacritized word_diac as word.\n')
            self.word = REGEX_DIACS.sub('', self.word_diac)
        self.make_labels(shadda)

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
        # print 'done normalizing. word:', self.word, 'word_diac:', self.word_diac


class Sequence(object):
    """
    A class for a sequence of words

    words_str (list): a list of word strings
    words_diac_str (list): a list of diacritized word strings
    seq_id (str): an ID for the sequence
    shadda (str): strategy for dealing with shadda
    """

    def __init__(self, words_str, words_diac_str, seq_id, shadda=Word.SHADDA_WITH_NEXT, word_type=type(Word)):

        if len(words_str) != len(words_diac_str):
            sys.stderr.write('Error: incompatible words_str ' + str(words_str) + ' and words_diac_str ' + str(words_diac_str) + '\n')
            return

        self.words = []
        for i in xrange(len(words_str)):
            word_str, word_diac_str = words_str[i], words_diac_str[i]
            if word_type == type(KaldiWord):
                word = KaldiWord(word_str, word_diac_str, shadda)
            else:
                word = Word(word_str, word_diac_str, shadda)
            self.words.append(word)

        self.seq_id = seq_id

    def __len__(self):

        return len(self.words)

    def __str__(self):

        res = ''
        for word in self.words:
            res += str(word) + '\n'
        return res

    def num_letters(self, count_word_boundary=False):

        num = 0
        for word in self.words:
            num += word.num_letters
        if count_word_boundary:
            num += len(self) + 1  # include boundaries for beginning and end of sequence
        return num

    def get_sequence_letters(self, include_word_boundary=False):

        letters = []
        if include_word_boundary:
            letters.append(Word.WORD_BOUNDARY)
        for word in self.words:
            letters += word.letters
            if include_word_boundary:
                letters.append(Word.WORD_BOUNDARY)
        return letters

    def get_sequence_letter2word(self, include_word_boundary=False):
        """
        Get a list of words corresponding to the letter sequence

        word i in the list is the word containing the i-th letter in the sequence
        """

        letter2word = []
        if include_word_boundary:
            letter2word.append(Word.WORD_BOUNDARY)
        for word in self.words:
            for letter in word.letters:
                letter2word.append(word)
            if include_word_boundary:
                letter2word.append(Word.WORD_BOUNDARY)
        return letter2word

    @staticmethod
    def is_sequence_stop(word_str, word_diac_str, stop_on_punc=False):
        """
        Check if there is a sequence stop at this word
        """

        if word_str == word_diac_str:
            if word_str == '':
                return True
            elif stop_on_punc and word_str in PUNCS_STOP:
                return True
        return False


def load_extracted_data(word_filename, word_diac_filename, stop_on_punc=False, shadda=Word.SHADDA_WITH_NEXT):
    """
    Load data extracted from the Treebank

    word_filename (str): A text file with one word per line, a blank line between sentences.
    word_diac_filename (str): A text file with one diacritized word per line, a blank line between sentences.
                               Corresponds to word_filename.
    stop_on_punc (bool): If True, stop sequence on punctuation
    shadda (str): Strategy for dealing with shadda
    return: sequences (list): A list of Sequence objects containing sentences for the data set
    """

    print 'loading extracted data from:', word_filename, word_diac_filename
    if stop_on_punc:
        print 'stopping sequences on punctuations'

    word_lines = open(word_filename).readlines()
    word_diac_lines = open(word_diac_filename).readlines()
    if len(word_lines) != len(word_diac_lines):
        sys.stderr.write('Error: incompatible word file ' + word_filename + \
                         ' and word_diac file ' + word_diac_filename + '\n')
        return

    sequences = []
    words_str = []
    words_diac_str = []
    sequence_lengths = []
    for word_line, word_diac_line in zip(word_lines, word_diac_lines):
        word_str = word_line.strip()
        word_diac_str = word_diac_line.strip()
        if (word_str == '' and word_diac_str != '') or (word_str != '' and word_diac_str == ''):
            sys.stderr.write('Warning: word_str ' + word_str + ' xor word_diac_str ' + word_diac_str + \
                             ' is empty, ignoring word')
            continue
        if Sequence.is_sequence_stop(word_str, word_diac_str, stop_on_punc):
            # if stopped on non-empty word, include it in the sequence
            if word_str != '':
                words_str.append(word_str)
                words_diac_str.append(word_diac_str)
            seq_id = os.path.basename(word_filename) + ':' + str(len(sequences) + 1)
            sequence = Sequence(words_str, words_diac_str, seq_id, shadda)
            sequences.append(sequence)
            sequence_lengths.append(sequence.num_letters())
            words_str = []
            words_diac_str = []
        else:
            words_str.append(word_str)
            words_diac_str.append(word_diac_str)
    if words_str:
        seq_id = os.path.basename(word_filename) + ':' + str(len(sequences) + 1)
        sequence = Sequence(words_str, words_diac_str, seq_id, shadda)
        sequences.append(sequence)

    print 'found', len(sequences), 'sequences'
    print 'average sequence length:', np.mean(sequence_lengths), 'std dev:', np.std(sequence_lengths), 'max:', max(sequence_lengths)
    return sequences


def load_kaldi_data(bw_mada_filename, shadda=Word.SHADDA_WITH_NEXT):
    """
    Load data used in Kaldi experiments

    bw_mada_filename: each line contains an id followed by bw:mada strings
    shadda:
    return:
    """

    print 'loading kaldi data from:', bw_mada_filename

    sequences = []
    words_str = []
    words_diac_str = []
    sequence_lengths = []
    f = open(bw_mada_filename)
    for line in f:
        splt = line.strip().split()
        seq_id = splt[0]
        for pair in splt[1:]:
            word_str, word_diac_str = pair.split(':')
            if (word_str == '' and word_diac_str != '') or (word_str != '' and word_diac_str == ''):
                sys.stderr.write('Warning: word_str ' + word_str + ' xor word_diac_str ' + word_diac_str + \
                                 ' is empty, ignoring word')
                continue
            words_str.append(word_str)
            words_diac_str.append(word_diac_str)
        sequence = Sequence(words_str, words_diac_str, seq_id, shadda, word_type=type(KaldiWord))
        sequences.append(sequence)
        sequence_lengths.append(sequence.num_letters())
        words_str = []
        words_diac_str = []

    f.close()
    print 'found', len(sequences), 'sequences'
    print 'average sequence length:', np.mean(sequence_lengths), 'std dev:', np.std(sequence_lengths), 'max:', max(sequence_lengths)
    return sequences


def load_label_indices(label_indices_filename):
    """
    Load label indices used in training
    
    label_indices_filename: one label (diacritic) per line, in order used in Current
    returns class2label, label2class (dicts): maps from index to label and from label to index
    """
    
    labels = open(label_indices_filename).readlines()
    labels = [label.strip() for label in labels]
    class2label, label2class = dict(enumerate(labels)), dict(zip(labels, range(len(labels))))
    return class2label, label2class


