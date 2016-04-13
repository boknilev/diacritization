__author__ = 'belinkov'

from itertools import izip_longest
from numpy import cumsum
import subprocess


def grouper(iterable, n, fillvalue=None):

    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def increment_dict(dic, k):

    if k in dic:
        dic[k] += 1
    else:
        dic[k] = 1


def ravel_list(data, lengths):
    """
    Ravel a list into a list of lists based on lengths
    """

    start_indices = cumsum(lengths) - lengths
    ar = [data[start_indices[i]:start_indices[i]+lengths[i]] for i in xrange(len(start_indices))]
    return ar


def argmax_two(vals):
    """
    Find indexes of max two values

    This only works when the max value is unique
    """

    best = -float("inf")
    arg_best = -1
    second_best = -float("inf")
    arg_second_best = -1
    for i in xrange(len(vals)):
        if vals[i] > best:
            best = vals[i]
            arg_best = i
    for i in xrange(len(vals)):
        if vals[i] < best and vals[i] > second_best:
            second_best = vals[i]
            arg_second_best = i
    return arg_best, arg_second_best


def load_word_vectors(word_vectors_filename):

    word_vectors = dict()
    with open(word_vectors_filename) as f:
        for line in f:
            splt = line.strip().split()
            if len(splt) <= 2:
                continue
            word = splt[0]
            vec = [float(d) for d in splt[1:]]
            word_vectors[word] = vec
    return word_vectors


def get_word_vectors_size(word_vectors):

    for word in word_vectors:
        return len(word_vectors[word])
        
        
def bw2utf8(bw):
    
    pipe = subprocess.Popen(['perl', 'bw2utf8.pl', bw], stdout=subprocess.PIPE)
    utf8 = pipe.stdout.read().decode('utf-8')
    return utf8