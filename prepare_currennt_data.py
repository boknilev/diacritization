__author__ = 'belinkov'

import operator
import time
import sys
from netCDF4 import Dataset, stringtoarr
from data_utils import Word, load_extracted_data, load_kaldi_data, load_label_indices
from utils import *
import numpy as np
import argparse
from gensim.models import Word2Vec

UNK_LETTER = '__UNK__'
UNK_WORD = 'UNK'  # for word vectors


class CurrenntDataset(object):
    """
    Wraps a data set that will be used in Currennt
    """

    MAX_SEQ_TAG_LENGTH = 800
    MAX_TARGET_STRING_LENGTH = 10000
    INCLUDE_WORD_BOUNDARY = True
    DEFAULT_WINDOW_SIZE = 5

    def __init__(self, nc_filename, sequences, letter_features_size, map_letter2features,
                 window_size=DEFAULT_WINDOW_SIZE, map_label2class=None, word_vectors=None):
        """

        nc_filename (str): A file to write the dataset in netCDF format
        sequences (list): A list of Sequence objects containing the data
        map_letter2features (dict): a map from letters to feature vectors
        map_label2class (dict): a map from label to class
        word_vectors (dict): a map from word to vector
        """

        print 'preparing Currennt dataset'
        self.nc_filename = nc_filename
        self.sequences = sequences
        self.letter_features_size = letter_features_size
        self.input_pattern_size = letter_features_size * (2 * window_size + 1)
        if word_vectors:
            self.input_pattern_size += get_word_vectors_size(word_vectors)
        self.map_letter2features = map_letter2features
        self.window_size = window_size

        nc_file = Dataset(nc_filename, 'w')

        # collect label information
        # if given a map (say, from training set), use it
        if map_label2class:
            self.map_label2class = map_label2class
            max_label_length = 0
            for label in self.map_label2class:
                max_label_length = max(max_label_length, len(label))
        # otherwise create a new map
        else:
            labels = set()
            max_label_length = 0
            for sequence in sequences:
                for word in sequence.words:
                    for diac in word.diacs:
                        labels.add(diac)
                        max_label_length = max(max_label_length, len(diac))
            labels.add(Word.WORD_BOUNDARY)  # word boundary label (same as word boundary symbol)
            max_label_length = max(max_label_length, len(Word.WORD_BOUNDARY))
            # create map from label (diacritic) to class (integer)
            map_label2class = dict()
            for label in labels:
                map_label2class[label] = len(map_label2class)  # TODO: make sure classes are 0-indexed
            self.map_label2class = map_label2class
        print 'label2class map:', self.map_label2class

        # create dimensions
        dim_num_seqs = nc_file.createDimension('numSeqs', len(sequences))
        num_timesteps = 0
        for sequence in sequences:
            num_timesteps += sequence.num_letters(count_word_boundary=self.INCLUDE_WORD_BOUNDARY)
        dim_num_timesteps = nc_file.createDimension('numTimesteps', num_timesteps)
        dim_input_pattern_size = nc_file.createDimension('inputPattSize', self.input_pattern_size)
        dim_max_seq_tag_length = nc_file.createDimension('maxSeqTagLength', self.MAX_SEQ_TAG_LENGTH)
        # optional dimensions
        dim_num_labels = nc_file.createDimension('numLabels', len(map_label2class))
        dim_max_label_length = nc_file.createDimension('maxLabelLength', max_label_length)
        dim_max_target_string_length = nc_file.createDimension('maxTargStringLength', self.MAX_TARGET_STRING_LENGTH)

        # create variables
        var_seq_tags = nc_file.createVariable('seqTags', 'S1', ('numSeqs', 'maxSeqTagLength'))
        var_seq_tags.longname = 'sequence tags'
        var_seq_lengths = nc_file.createVariable('seqLengths', 'i4', ('numSeqs'))
        var_seq_lengths.longname = 'sequence lengths'
        var_inputs = nc_file.createVariable('inputs', 'f4', ('numTimesteps', 'inputPattSize'))
        var_inputs.longname = 'inputs'
        var_target_classes = nc_file.createVariable('targetClasses', 'i4', ('numTimesteps'))
        var_target_classes.longname = 'target classes'
        # optional variables
        var_num_target_classes = nc_file.createVariable('numTargetClasses', 'i4')
        var_num_target_classes.longname = 'number of target classes'
        var_labels = nc_file.createVariable('labels', 'S1', ('numLabels', 'maxLabelLength'))
        var_labels.longname = 'target labels'
        var_target_strings = nc_file.createVariable('targetStrings', 'S1', ('numSeqs', 'maxTargStringLength'))
        var_target_strings.longname = 'target strings'

        # write data to variables
        print 'writing sequence tags'
        seq_tags = []
        for sequence in sequences:
            seq_tags.append(stringtoarr(sequence.seq_id, self.MAX_SEQ_TAG_LENGTH))
        var_seq_tags[:] = seq_tags
        print 'writing sequence lengths'
        seq_lengths = []
        for sequence in sequences:
            seq_lengths.append(sequence.num_letters(count_word_boundary=self.INCLUDE_WORD_BOUNDARY))
        var_seq_lengths[:] = seq_lengths
        print 'writing inputs'
        # create empty array for the inputs
        inputs = np.empty((0, self.input_pattern_size))
        for sequence in sequences:
            sequence_features = self.generate_sequence_features(sequence)
            inputs = np.concatenate((inputs, sequence_features))
        var_inputs[:,:] = inputs
        print 'writing target classes'
        target_classes = []
        for sequence in sequences:
            if self.INCLUDE_WORD_BOUNDARY:
                target_classes.append(map_label2class[Word.WORD_BOUNDARY])
            for word in sequence.words:
                for diac in word.diacs:
                    assert(diac in map_label2class)
                    target_classes.append(map_label2class[diac])
                if self.INCLUDE_WORD_BOUNDARY:
                    target_classes.append(map_label2class[Word.WORD_BOUNDARY])
        var_target_classes[:] = target_classes
        # write data for optional variables
        var_num_target_classes[:] = len(map_label2class)
        labels_arr = np.empty((0, max_label_length))
        labels_ordered = [i[0] for i in sorted(self.map_label2class.items(), key=operator.itemgetter(1))]
        for label in labels_ordered:
            labels_arr = np.concatenate((labels_arr, [stringtoarr(label, max_label_length)]))
        var_labels[:,:] = labels_arr
        print 'writing target strings'
        target_strings = np.empty((0, self.MAX_TARGET_STRING_LENGTH))
        for sequence in sequences:
            sequence_letters = sequence.get_sequence_letters(include_word_boundary=self.INCLUDE_WORD_BOUNDARY)
            if len(sequence_letters) > self.MAX_TARGET_STRING_LENGTH:
                sys.stderr.write('Warning: length of sequence letters in sequence: ' + sequence.seq_id + \
                                 ' > MAX_TARGET_STRING_LENGTH\n')
            target_strings = np.concatenate((target_strings, \
                                            [stringtoarr(''.join(sequence_letters), self.MAX_TARGET_STRING_LENGTH)]))
        var_target_strings[:,:] = target_strings

        nc_file.close()
        print 'Currennt dataset written to:', nc_filename
        # print nc_file.dimensions
        # print nc_file

    def generate_sequence_features(self, sequence, word_vectors=None):
        """
        Generate a feature vector for a sequence

        :param sequence: a Sequence object
        :param word_vectors (dict): a map from word to vector
        :return: feature_vector (numpy.ndarray): a 2d array of (sequence length, input_pattern_size * (2*window_size+1))
                                                 representing the sequence features
        """

        # print 'generating features for sequence:\n', sequence
        letters = sequence.get_sequence_letters(include_word_boundary=self.INCLUDE_WORD_BOUNDARY)
        if word_vectors:
            letter2word = sequence.get_sequence_letter2word(include_word_boundary=self.INCLUDE_WORD_BOUNDARY)
        feature_vector_shape = (len(letters), self.input_pattern_size)
        feature_vector = np.zeros(feature_vector_shape)
        for i in xrange(len(letters)):
            for j in xrange(2 * self.window_size + 1):
                pos = i - self.window_size + j  # position in the sequence
                # if we're in the sequence
                if 0 <= pos < len(letters):
                    letter = letters[pos]
                    if letter in self.map_letter2features:
                        letter_features = self.map_letter2features[letter]
                    else:
                        sys.stderr.write('Warning: found unknown letter ' + letter + ' in sequence: ' + \
                                         sequence.seq_id + ', using unknown letter features\n')
                        letter_features = self.map_letter2features[UNK_LETTER]
                    feature_vector[i][self.letter_features_size * j: self.letter_features_size * (j + 1)] = letter_features
            if word_vectors:
                vec_size = get_word_vectors_size(word_vectors)
                # append a word vector for the word containing the current letter
                word = letter2word[i]
                if word in word_vectors:
                    vec = word_vectors[word]
                elif self.INCLUDE_WORD_BOUNDARY and word == Word.WORD_BOUNDARY:
                    vec = np.zeros(vec_size)
                elif UNK_WORD in word_vectors:
                    vec = word_vectors[UNK_WORD]
                else:
                    vec = np.zeros(vec_size)
                feature_vector[i][self.letter_features_size * (2 * self.window_size + 1): \
                    self.letter_features_size * (2 * self.window_size + 1) + vec_size] = vec

        return feature_vector

    @staticmethod
    def seq_target_strings2string(seq_target_strings):
        """
        Get a a concatenation of the sequence target strings
        """

        return ''.join(seq_target_strings.data)

    @staticmethod
    def get_vocab_from_nc_file(nc_file):

        print 'getting vocabulary from file:', nc_file
        vocab = set()
        if CurrenntDataset.INCLUDE_WORD_BOUNDARY:
            vocab.add(Word.WORD_BOUNDARY)
        for seq in nc_file.variables['targetStrings']:
            seq_target_string = CurrenntDataset.seq_target_strings2string(seq)
            for word in seq_target_string.split(Word.WORD_BOUNDARY):
                vocab.add(word)

        print 'vocab size:', len(vocab)
        return vocab


class FeatureInitializer(object):
    """
    Class for initializing letter feature vectors
    """

    DIST_GAUSSIAN = 'Gaussian'
    STRAT_RAND = 'Strategy_random'  # initialize by random distributions
    STRAT_WORD2VEC = 'Strategy_word2vec'  # initialize by running word2vec on letter sequences
    STRAT_FILE = 'Strategy_file' # initialize from file

    def __init__(self, sequences, min_letter_count=50, letter_features_size=10, \
                    strategy=STRAT_RAND, letter_features_filename=None):

        print 'initializing features'
        self.min_letter_count = min_letter_count
        self.letter_features_size = letter_features_size
        map_letter2count = dict()
        for sequence in sequences:
            letters = sequence.get_sequence_letters(include_word_boundary=CurrenntDataset.INCLUDE_WORD_BOUNDARY)
            for letter in letters:
                increment_dict(map_letter2count, letter)
        self.map_letter2count = map_letter2count
        print 'letter to count:'
        print sorted(map_letter2count.items(), key=operator.itemgetter(1))
        self.map_letter2features = None
        if strategy == FeatureInitializer.STRAT_RAND:
            self.init_random()
        elif strategy == FeatureInitializer.STRAT_WORD2VEC:
            self.init_word2vec(sequences)
        elif strategy == FeatureInitializer.STRAT_FILE and letter_features_filename:
            self.init_from_file(letter_features_filename)
        else:
            sys.stderr.write('Warning: unkown strategy ' + strategy + ' in __init__(), resorting to Random\n')
            self.init_random()

    def init_word2vec(self, sequences, workers=4):

        letter_sequences = []
        for sequence in sequences:
            letters = sequence.get_sequence_letters(include_word_boundary=CurrenntDataset.INCLUDE_WORD_BOUNDARY)
            letters_or_unk = [letter if self.map_letter2count[letter] >= self.min_letter_count else UNK_LETTER \
                              for letter in letters]
            letter_sequences.append(letters_or_unk)

        word2vec_model = Word2Vec(size=self.letter_features_size, workers=workers)
        word2vec_model.build_vocab(letter_sequences)
        word2vec_model.train(letter_sequences)
        map_letter2features = dict()
        for letter in word2vec_model.vocab:
            map_letter2features[letter] = word2vec_model[letter]
        self.map_letter2features = map_letter2features

    def init_random(self, dist=DIST_GAUSSIAN, params=(0, 1), scale=0.1):

        map_letter2features = dict()
        map_letter2features[UNK_LETTER] = self.init_letter_features_random(self.letter_features_size, dist, params, scale)
        for letter in self.map_letter2count:
            # only take letters that appear above a threshold (others will be treated as unknown)
            if self.map_letter2count[letter] >= self.min_letter_count:
                map_letter2features[letter] = self.init_letter_features_random(self.letter_features_size, dist, params, scale)
        self.map_letter2features = map_letter2features

    def init_letter_features_random(self, letter_features_size, dist=DIST_GAUSSIAN, params=(0, 1), scale=0.1):
        """
        Create a feature vector for a single letter

        :param letter_features_size (int): size of the letter feature vector
        :param dist (str): the distribution from which to draw the feature vector
        :param params (tuple): parameters for the distribution
        :param scale (float): scale for the features
        :return (numpy.ndarray): the letter feature vector
        """

        if dist == FeatureInitializer.DIST_GAUSSIAN:
            letter_features = self.init_letter_features_gaussian(letter_features_size, params[0], params[1], scale)
        else:
            sys.stderr.write('Warning: unknown distribution ' + dist + \
                             ' in init_letter_features_random(), resorting to Gaussian\n')
            letter_features = self.init_letter_features_gaussian(letter_features_size, params[0], params[1], scale)
        return letter_features

    @staticmethod
    def init_letter_features_gaussian(letter_features_size, param_mean=0, param_stddev=1, scale=1):

        return scale * np.random.normal(param_mean, param_stddev, letter_features_size)
    
    def init_from_file(self, letter_features_filename):
        
        print 'initializing letter features from file:', letter_features_filename
        map_letter2features = dict()
        with open(letter_features_filename) as f:
            for line in f:
                splt = line.strip().split()
                letter = splt[0]
                vec = [float(v) for v in splt[1:]]
                if letter in map_letter2features:
                    assert map_letter2features[letter] == vec, 'bad vector comparison with duplicate letter'
                else:
                    map_letter2features[letter] = vec
        if UNK_LETTER not in map_letter2features:
            map_letter2features[UNK_LETTER] = self.init_letter_features_random(self.letter_features_size)
        self.map_letter2features = map_letter2features


def create_currennt_dataset(train_word_filename, train_word_diac_filename, train_nc_filename, \
                            test_word_filename=None, test_word_diac_filename=None, test_nc_filename=None, \
                            dev_word_filename=None, dev_word_diac_filename=None, dev_nc_filename=None, \
                            stop_on_punc=False, window_size=5, init_method=FeatureInitializer.STRAT_RAND, \
                            letter_features_size=10, shadda=Word.SHADDA_WITH_NEXT, word_vectors=None, \
                            letter_vectors_filename=None, label2class_filename=None):

    print 'loading training set'
    start_time = time.time()
    train_sequences = load_extracted_data(train_word_filename, train_word_diac_filename, stop_on_punc, shadda)                    
    feature_initializer = FeatureInitializer(train_sequences, strategy=init_method, \
                                             letter_features_size=letter_features_size, \
                                             letter_features_filename=letter_vectors_filename)
    if label2class_filename:
        _, map_label2class = load_label_indices(label2class_filename)
        train_dataset = CurrenntDataset(train_nc_filename, train_sequences, \
                                    feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                    window_size=window_size, map_label2class=map_label2class, word_vectors=word_vectors)        
    else:
        train_dataset = CurrenntDataset(train_nc_filename, train_sequences, \
                                    feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                    window_size=window_size, word_vectors=word_vectors)
    print 'elapsed time:', time.time() - start_time, 'seconds'
    if test_word_filename and test_word_diac_filename and test_nc_filename:
        print 'loading test set'
        start_time = time.time()
        test_sequences = load_extracted_data(test_word_filename, test_word_diac_filename, stop_on_punc, shadda)
        test_dataset = CurrenntDataset(test_nc_filename, test_sequences, \
                                       feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                       window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                       word_vectors=word_vectors)
        print 'elapsed time:', time.time() - start_time, 'seconds'
    if dev_word_filename and dev_word_diac_filename and dev_nc_filename:
        print 'loading dev set'
        start_time = time.time()
        dev_sequences = load_extracted_data(dev_word_filename, dev_word_diac_filename, stop_on_punc, shadda)
        dev_dataset = CurrenntDataset(dev_nc_filename, dev_sequences, \
                                       feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                       window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                       word_vectors=word_vectors)
        print 'elapsed time:', time.time() - start_time, 'seconds'


def create_currennt_dataset_from_kaldi(train_filename, train_nc_filename, test_filename, test_nc_filename, \
                            dev_filename=None, dev_nc_filename=None, \
                            window_size=5, init_method=FeatureInitializer.STRAT_RAND, \
                            letter_features_size=10, shadda=Word.SHADDA_WITH_NEXT, word_vectors=None):

    print 'loading training set'
    start_time = time.time()
    train_sequences = load_kaldi_data(train_filename, shadda)
    feature_initializer = FeatureInitializer(train_sequences, strategy=init_method, \
                                             letter_features_size=letter_features_size)
    train_dataset = CurrenntDataset(train_nc_filename, train_sequences, \
                                    feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                    window_size=window_size, word_vectors=word_vectors)
    print 'elapsed time:', time.time() - start_time, 'seconds'
    print 'loading test set'
    start_time = time.time()
    test_sequences = load_kaldi_data(test_filename, shadda)
    test_dataset = CurrenntDataset(test_nc_filename, test_sequences, \
                                   feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                   window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                   word_vectors=word_vectors)
    print 'elapsed time:', time.time() - start_time, 'seconds'
    if dev_filename and dev_nc_filename:
        print 'loading dev set'
        start_time = time.time()
        dev_sequences = load_kaldi_data(dev_filename, shadda)
        dev_dataset = CurrenntDataset(dev_nc_filename, dev_sequences, \
                                       feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                       window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                       word_vectors=word_vectors)
        print 'elapsed time:', time.time() - start_time, 'seconds'


def create_currennt_dataset_from_atb_kaldi(train_word_filename, train_word_diac_filename, train_nc_filename, \
                            test_filename, test_nc_filename, \
                            dev_word_filename=None, dev_word_diac_filename=None, dev_nc_filename=None, \
                            stop_on_punc=False, window_size=5, init_method=FeatureInitializer.STRAT_RAND, \
                            letter_features_size=10, shadda=Word.SHADDA_WITH_NEXT, word_vectors=None):

    print 'loading training set'
    start_time = time.time()
    train_sequences = load_extracted_data(train_word_filename, train_word_diac_filename, stop_on_punc, shadda)
    feature_initializer = FeatureInitializer(train_sequences, strategy=init_method, \
                                             letter_features_size=letter_features_size)
    train_dataset = CurrenntDataset(train_nc_filename, train_sequences, \
                                    feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                    window_size=window_size, word_vectors=word_vectors)
    print 'elapsed time:', time.time() - start_time, 'seconds'
    print 'loading test set'
    start_time = time.time()
    test_sequences = load_kaldi_data(test_filename, shadda)
    test_dataset = CurrenntDataset(test_nc_filename, test_sequences, \
                                   feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                   window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                   word_vectors=word_vectors)
    print 'elapsed time:', time.time() - start_time, 'seconds'
    if dev_word_filename and dev_word_diac_filename and dev_nc_filename:
        print 'loading dev set'
        start_time = time.time()
        dev_sequences = load_extracted_data(dev_word_filename, dev_word_diac_filename, stop_on_punc, shadda)
        dev_dataset = CurrenntDataset(dev_nc_filename, dev_sequences, \
                                       feature_initializer.letter_features_size, feature_initializer.map_letter2features, \
                                       window_size=window_size, map_label2class=train_dataset.map_label2class, \
                                       word_vectors=word_vectors)
        print 'elapsed time:', time.time() - start_time, 'seconds'


def main():

    parser = argparse.ArgumentParser(description='Prepare Currennt data')
    parser.add_argument('-twf', '--train_word_file', help='training word file', required=True)
    parser.add_argument('-twdf', '--train_word_diac_file', help='training word diacritized file', required=True)
    parser.add_argument('-tncf', '--train_nc_file', help='training Currennt nc file', required=True)
    parser.add_argument('-swf', '--test_word_file', help='testing word file')
    parser.add_argument('-swdf', '--test_word_diac_file', help='testing word diacritized file')
    parser.add_argument('-sncf', '--test_nc_file', help='testing Currennt nc file')
    parser.add_argument('-dwf', '--dev_word_file', help='development word file')
    parser.add_argument('-dwdf', '--dev_word_diac_file', help='development word diacritized file')
    parser.add_argument('-dncf', '--dev_nc_file', help='development Currennt nc file')
    parser.add_argument('-punc', '--stop_on_punc', help='stop on punctuation (default: False)', action='store_true')
    parser.add_argument('-win', '--window_size', help='context window size (default: 5)', type=int, default=5)
    parser.add_argument('-init', '--init_method', help='input initialization method (default: Gaussian)', \
                        default=FeatureInitializer.STRAT_RAND, \
                        choices=[FeatureInitializer.STRAT_RAND, FeatureInitializer.STRAT_WORD2VEC, \
                                 FeatureInitializer.STRAT_FILE])
    parser.add_argument('-size', '--letter_features_size', help='input letter features size', type=int, default=10)
    parser.add_argument('-shadda', '--shadda', help='shadda strategy', default=Word.SHADDA_WITH_NEXT, \
                        choices=[Word.SHADDA_WITH_NEXT, Word.SHADDA_IGNORE, Word.SHADDA_ONLY])
    parser.add_argument('-lvf', '--letter_vectors_file', help='letter vectors file (for initialization)')
    parser.add_argument('-wvf', '--word_vectors_file', help='word vectors file')
    parser.add_argument('-l2cf', '--label2class_filename', help='with with labels in order corresponding to indices')
    args = parser.parse_args()

    word_vectors = None
    if args.word_vectors_file:
        print 'loading word vectors from:', args.word_vectors_file
        word_vectors = load_word_vectors(args.word_vectors_file)
    create_currennt_dataset(args.train_word_file, args.train_word_diac_file, args.train_nc_file, \
                            args.test_word_file, args.test_word_diac_file, args.test_nc_file, \
                            args.dev_word_file, args.dev_word_diac_file, args.dev_nc_file, \
                            stop_on_punc=args.stop_on_punc, window_size=args.window_size, \
                            init_method=args.init_method, letter_features_size=args.letter_features_size, \
                            shadda=args.shadda, word_vectors=word_vectors, \
                            letter_vectors_filename=args.letter_vectors_file, \
                            label2class_filename=args.label2class_filename)


def main_kaldi():
    """
    For kaldi data

    :return:
    """

    parser = argparse.ArgumentParser(description='Prepare Currennt data')
    parser.add_argument('-tf', '--train_file', help='training bw-mada file', required=True)
    parser.add_argument('-tncf', '--train_nc_file', help='training Currennt nc file', required=True)
    parser.add_argument('-sf', '--test_file', help='testing bw-mada file', required=True)
    parser.add_argument('-sncf', '--test_nc_file', help='testing Currennt nc file', required=True)
    parser.add_argument('-df', '--dev_file', help='development bw-mada file')
    parser.add_argument('-dncf', '--dev_nc_file', help='development Currennt nc file')
    parser.add_argument('-win', '--window_size', help='context window size (default: 5)', type=int, default=5)
    parser.add_argument('-init', '--init_method', help='input initialization method (default: Gaussian)', \
                        default=FeatureInitializer.STRAT_RAND, \
                        choices=[FeatureInitializer.STRAT_RAND, FeatureInitializer.STRAT_WORD2VEC])
    parser.add_argument('-size', '--letter_features_size', help='input letter features size', type=int, default=10)
    parser.add_argument('-shadda', '--shadda', help='shadda strategy', default=Word.SHADDA_WITH_NEXT, \
                        choices=[Word.SHADDA_WITH_NEXT, Word.SHADDA_IGNORE, Word.SHADDA_ONLY])
    parser.add_argument('-wvf', '--word_vectors_file', help='word vectors file')
    args = parser.parse_args()

    word_vectors = None
    if args.word_vectors_file:
        print 'loading word vectors from:', args.word_vectors_file
        word_vectors = load_word_vectors(args.word_vectors_file)
    create_currennt_dataset_from_kaldi(args.train_file, args.train_nc_file, args.test_file, args.test_nc_file, \
                            args.dev_file, args.dev_nc_file, window_size=args.window_size, \
                            init_method=args.init_method, letter_features_size=args.letter_features_size, \
                            shadda=args.shadda, word_vectors=word_vectors)


def main_atb_kaldi():
    """
    For mixing ATB and Kaldi data

    ATB data used for train/dev, Kaldi (train) data used for test

    :return:
    """

    parser = argparse.ArgumentParser(description='Prepare Currennt data')
    parser.add_argument('-twf', '--train_word_file', help='training word file', required=True)
    parser.add_argument('-twdf', '--train_word_diac_file', help='training word diacritized file', required=True)
    parser.add_argument('-tncf', '--train_nc_file', help='training Currennt nc file', required=True)
    parser.add_argument('-sf', '--test_file', help='testing bw-mada file', required=True)
    parser.add_argument('-sncf', '--test_nc_file', help='testing Currennt nc file', required=True)
    parser.add_argument('-dwf', '--dev_word_file', help='development word file')
    parser.add_argument('-dwdf', '--dev_word_diac_file', help='development word diacritized file')
    parser.add_argument('-dncf', '--dev_nc_file', help='development Currennt nc file')
    parser.add_argument('-punc', '--stop_on_punc', help='stop on punctuation (default: False)', action='store_true')
    parser.add_argument('-win', '--window_size', help='context window size (default: 5)', type=int, default=5)
    parser.add_argument('-init', '--init_method', help='input initialization method (default: Gaussian)', \
                        default=FeatureInitializer.STRAT_RAND, \
                        choices=[FeatureInitializer.STRAT_RAND, FeatureInitializer.STRAT_WORD2VEC])
    parser.add_argument('-size', '--letter_features_size', help='input letter features size', type=int, default=10)
    parser.add_argument('-shadda', '--shadda', help='shadda strategy', default=Word.SHADDA_WITH_NEXT, \
                        choices=[Word.SHADDA_WITH_NEXT, Word.SHADDA_IGNORE, Word.SHADDA_ONLY])
    parser.add_argument('-wvf', '--word_vectors_file', help='word vectors file')
    args = parser.parse_args()

    word_vectors = None
    if args.word_vectors_file:
        print 'loading word vectors from:', args.word_vectors_file
        word_vectors = load_word_vectors(args.word_vectors_file)
    create_currennt_dataset_from_atb_kaldi(args.train_word_file, args.train_word_diac_file, args.train_nc_file, \
                            args.test_file, args.test_nc_file, \
                            args.dev_word_file, args.dev_word_diac_file, args.dev_nc_file, \
                            stop_on_punc=args.stop_on_punc, window_size=args.window_size, \
                            init_method=args.init_method, letter_features_size=args.letter_features_size, \
                            shadda=args.shadda, word_vectors=word_vectors)


if __name__ == '__main__':
    main()
