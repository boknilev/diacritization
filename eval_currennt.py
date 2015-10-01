from data_utils import Word

__author__ = 'belinkov'

# Evaluate predictions by Currennt

from netCDF4 import Dataset
from utils import *
from data_utils import SHADDA
from prepare_currennt_data import CurrenntDataset
import numpy as np
import sys
import operator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import izip


def print_confusion_matrix(gold_classes, pred_classes, labels):
    cm = confusion_matrix(gold_classes, pred_classes, range(len(labels)))
    print '\nConfusion Matrix'
    print cm
    correct = np.trace(cm)
    total = len(gold_classes)
    wrong = total - correct
    print 'total:', total, 'correct (trace):', correct, 'Accuracy:', "%.2f" %(100.0*correct/total), \
            'error rate:', "%.2f" %(100.0*wrong/total)
    print 'break down of error rate per label:'
    for i in xrange(len(labels)):
        label_wrong = sum(cm[i]) - cm[i][i]
        print '\"' + labels[i] + '\"', "%.2f" %(100.0*label_wrong/total),
 
    new_labels = []
    for l in labels:
        if l == Word.WORD_BOUNDARY:
            new_labels.append('#')
        else:
            new_labels.append(l)
    # new_labels = labels

    # TODO change this
    # """
    plt.figure('1')
    plt.matshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(range(len(new_labels)), new_labels)
    plt.yticks(range(len(new_labels)), new_labels)
    plt.savefig('confusion-matrix.png', bbox_inches='tight')
    # """
    # error matrix
    cm_err = cm
    for i in xrange(len(cm_err)):
        cm_err[i][i] = 0
    print '\nConfusion Matrix (errors only)'
    print cm_err
    # """"
    plt.figure(2)
    plt.matshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(range(len(new_labels)), new_labels)
    plt.yticks(range(len(new_labels)), new_labels)
    plt.savefig('confusion-matrix-errors.png', bbox_inches='tight')
    # """


def indexize_sequence_target_string(seq_target_string, word_boundary=Word.WORD_BOUNDARY):
    """
    Convert a string representing a sequence into a list of words for each letter

    :param seq_target_string: a target string concatenating all characters in the sequence
    :return: a list of words in the size of the seq length, at each index is the word containing the letter
    """

    splt = seq_target_string.split(word_boundary)
    ind_words = []
    for i in range(1, len(splt)-1):
        ind_words.append(word_boundary)
        for j in splt[i]:
            ind_words.append(splt[i])
    ind_words.append(word_boundary)
    return ind_words


def exclude_classes(gold_classes, pred_classes, excluded_classes):

    gold_classes_new, pred_classes_new = [], []
    for i in xrange(len(gold_classes)):
        if gold_classes[i] not in excluded_classes:
            gold_classes_new.append(gold_classes[i])
            pred_classes_new.append(pred_classes[i])
    return gold_classes_new, pred_classes_new


def collect_predictions(num_labels, pred_csv_filename):

    print 'collecting predictions'
    pred_classes = []
    pred2_classes = []
    with open(pred_csv_filename) as f:
        count = 0
        for line in f:
            count += 1
            # if count % 1000 == 0:
            # print 'sequence:', count
            splt = line.strip().split(';')
            seq_id = splt[0]
            probs = [float(p) for p in splt[1:]]
            for letter_probs in grouper(probs, num_labels, 0):
                arg_best, arg_second_best = argmax_two(letter_probs)
                pred_classes.append(arg_best)
                pred2_classes.append(arg_second_best)
    return pred2_classes, pred_classes


def merge_classes(noshadda_classes, onlyshadda_classes, noshadda_class2label, onlyshadda_class2label, all_label2class):
    """
    Merge classes from data with only shadda and data without shadda
    """

    if len(noshadda_classes) != len(onlyshadda_classes):
        sys.stderr.write('Error: incompatible classes in merge_classes()\n')
        return

    print 'merging classes'
    merged_classes = []
    for noshadda_class, onlyshadda_class in izip(noshadda_classes, onlyshadda_classes):
        noshadda_label = noshadda_class2label[noshadda_class]
        onlyshadda_label = onlyshadda_class2label[onlyshadda_class]
        if onlyshadda_label == SHADDA:
            merged_label = SHADDA + noshadda_label
            if merged_label in all_label2class:
                merged_classes.append(all_label2class[merged_label])
            else:
                sys.stderr.write('Warning: merged label ' + merged_label + ' not found in label2class map, appending noshadda class\n')
                merged_classes.append(noshadda_class)
        else:
            if noshadda_label in all_label2class:
                merged_classes.append(all_label2class[noshadda_label])
            else:
                sys.stderr.write('Warning: noshadda label ' + noshadda_label + ' not found in label2class map, appending empty string\n')
                merged_classes.append(all_label2class[''])
    if len(merged_classes) != len(noshadda_classes):
        sys.stderr.write('Error: merged classes length != noshadda classes length in merge_classes()\n')
        return

    return merged_classes


def merge_shadda_nc_files(all_nc_filename, noshadda_nc_filename, onlyshadda_nc_filename):
    """
    Merge files with only shadda and data without shadda

    :param all_nc_filename:
    :param noshadda_nc_filename:
    :param onlyshadda_nc_filename:
    :return:
    """

    # all labels
    all_nc_file = Dataset(all_nc_filename)
    all_labels = [''.join(l.data) for l in all_nc_file.variables['labels']]
    all_label2class = dict(zip(all_labels, range(len(all_labels))))
    # labels without shadda
    noshadda_nc_file = Dataset(noshadda_nc_filename)
    noshadda_classes = noshadda_nc_file.variables['targetClasses']
    noshadda_labels = [''.join(l.data) for l in noshadda_nc_file.variables['labels']]
    noshadda_class2label = dict(zip(range(len(noshadda_labels)), noshadda_labels))
    # only shadda label (and default ones like empty and word boundary)
    onlyshadda_nc_file = Dataset(onlyshadda_nc_filename)
    onlyshadda_classes = onlyshadda_nc_file.variables['targetClasses']
    onlyshadda_labels = [''.join(l.data) for l in onlyshadda_nc_file.variables['labels']]
    onlyshadda_class2label = dict(zip(range(len(onlyshadda_labels)), onlyshadda_labels))

    # merge
    merged_classes = merge_classes(noshadda_classes, onlyshadda_classes, noshadda_class2label, onlyshadda_class2label, \
                                   all_label2class)
    return merged_classes


def merge_shadda_and_evaluate_file(gold_all_nc_filename, gold_noshadda_nc_filename, gold_onlyshadda_nc_filename, \
                                   pred_noshadda_csv_filename, pred_onlyshadda_csv_filename, train_all_nc_filename, \
                                   shift_gold=0):
    """
    Merge noshadda and onlyshadda files and evaluate predictions

    :param gold_all_nc_filename:
    :param gold_noshadda_nc_filename:
    :param gold_onlyshadda_nc_filename:
    :param pred_noshadda_csv_filename:
    :param pred_onlyshadda_csv_filename:
    :param train_all_nc_filename:
    :param shift_gold:
    :return:
    """

    print 'merging shadda and evaluating file'
    # merge gold noshadda and onlyshadda classes
    gold_merged_classes = merge_shadda_nc_files(gold_all_nc_filename, gold_noshadda_nc_filename, \
                                                gold_onlyshadda_nc_filename)

    # build some maps
    gold_all_nc_file = Dataset(gold_all_nc_filename)
    gold_all_seq_lengths = gold_all_nc_file.variables['seqLengths']
    gold_all_target_strings = gold_all_nc_file.variables['targetStrings']
    gold_noshadda_nc_file = Dataset(gold_noshadda_nc_filename)
    num_labels_noshadda = len(gold_noshadda_nc_file.dimensions['numLabels'])
    gold_onlyshadda_nc_file = Dataset(gold_onlyshadda_nc_filename)
    num_labels_onlyshadda = len(gold_onlyshadda_nc_file.dimensions['numLabels'])
    all_labels = [''.join(l.data) for l in gold_all_nc_file.variables['labels']]
    all_label2class = dict(zip(all_labels, range(len(all_labels))))
    noshadda_labels = [''.join(l.data) for l in gold_noshadda_nc_file.variables['labels']]
    noshadda_class2label = dict(zip(range(len(noshadda_labels)), noshadda_labels))
    onlyshadda_labels = [''.join(l.data) for l in gold_onlyshadda_nc_file.variables['labels']]
    onlyshadda_class2label = dict(zip(range(len(onlyshadda_labels)), onlyshadda_labels))

    # collect predictions
    pred2_noshadda_classes, pred_noshadda_classes = collect_predictions(num_labels_noshadda, pred_noshadda_csv_filename)
    pred2_onlyshadda_classes, pred_onlyshadda_classes = collect_predictions(num_labels_onlyshadda, pred_onlyshadda_csv_filename)

    # merge pred noshadda and only shadda classes
    pred_merged_classes = merge_classes(pred_noshadda_classes, pred_onlyshadda_classes, \
                                        noshadda_class2label, onlyshadda_class2label, all_label2class)
    pred2_merged_classes = merge_classes(pred2_noshadda_classes, pred2_onlyshadda_classes, \
                                        noshadda_class2label, onlyshadda_class2label, all_label2class)
    if len(gold_merged_classes) != len(pred_merged_classes) or len(gold_merged_classes) != len(pred2_merged_classes):
        sys.stderr.write('Error: number of gold classes != number of pred classes')
        return

    # collect train vocab
    train_all_nc_file = Dataset(train_all_nc_filename)
    train_vocab = CurrenntDataset.get_vocab_from_nc_file(train_all_nc_file)

    # evaluate
    evaluate(gold_merged_classes, pred_merged_classes, pred2_merged_classes, \
             gold_all_seq_lengths, gold_all_target_strings, train_vocab, all_labels)


def evaluate_file(gold_nc_filename, pred_csv_filename, train_nc_filename, shift_gold=0):
    """
    Evaluate Currennt predictions

    :type shift_gold (int): number of time steps to shift outputs (produced by Currennt's --output_time_lag option)
    :param gold_nc_filename (str): file in Currennt format with gold classes
    :param pred_csv_filename (str): file in csv format with predictions
    :param train_nc_filename (str): training file in Currennt format
    :return:
    """

    gold_nc_file = Dataset(gold_nc_filename)
    gold_classes = gold_nc_file.variables['targetClasses']
    target_strings = gold_nc_file.variables['targetStrings']
    num_labels = len(gold_nc_file.dimensions['numLabels'])
    seq_lengths = gold_nc_file.variables['seqLengths']
    nc_labels = [''.join(l.data) for l in gold_nc_file.variables['labels']]
    label2class = dict(zip(nc_labels, range(len(nc_labels))))
    class2label = dict(zip(range(len(nc_labels)), nc_labels))
    # TODO change this! shouldn't be hard coded
    LABEL2CLASS = {'a': 0, '': 1, '_###_': 5, 'F': 3, 'i': 4, 'K': 6, '~u': 7, 'o': 8, 'N': 9, '`': 2, '~K': 11, 'u': 12, '~i': 10, '~N': 13, '~`': 14, '~a': 15, '~': 16}
    LABELS = [i[0] for i in sorted(LABEL2CLASS.items(), key=operator.itemgetter(1))]
    if nc_labels != LABELS:
        sys.stderr.write('Warning: unexpected label set, will use labels in nc file\n')

    # collect train vocab
    train_nc_file = Dataset(train_nc_filename)
    train_vocab = CurrenntDataset.get_vocab_from_nc_file(train_nc_file)

    # collect predictions
    pred2_classes, pred_classes = collect_predictions(num_labels, pred_csv_filename)
    if len(gold_classes) != len(pred_classes):
        sys.stderr.write('Error: number of gold classes != number of pred classes')
        return

    # evaluate
    evaluate(gold_classes, pred_classes, pred2_classes, seq_lengths, target_strings, train_vocab, nc_labels, shift_gold)


def evaluate(gold_classes, pred_classes, pred2_classes, seq_lengths, target_strings, train_vocab, labels, shift_gold=0):

    print 'evaluating'
    label2class = dict(zip(labels, range(len(labels))))
    ar_gold_classes = ravel_list(gold_classes, seq_lengths)
    ar_pred_classes = ravel_list(pred_classes, seq_lengths)
    ar_pred2_classes = ravel_list(pred2_classes, seq_lengths)
    correct, correct2, total = 0, 0, 0
    correct_at_end, correct2_at_end, total_at_end = 0, 0, 0
    gold2count, pred2count, pred22count = dict(), dict(), dict()
    gold_classes, pred_classes, pred2_classes = [], [], []  # collect again, this time accounting for output shifts
    wrong_unseen, total_unseen = 0, 0
    for i in xrange(len(ar_gold_classes)):
        print 'sequence:', i
        seq_target_string = CurrenntDataset.seq_target_strings2string(target_strings[i])
        seq_ind_words = indexize_sequence_target_string(seq_target_string)
        # print target_strings[i], len(target_strings[i])
        # print seq_target_string, len(seq_target_string)
        # print seq_ind_words, len(seq_ind_words)
        # print ar_gold_classes[i], len(ar_gold_classes[i])
        if len(seq_ind_words) != len(ar_gold_classes[i]):
            sys.stderr.write('Warning: bad length of seq_ind_words\n')
            # print target_strings[i][:10], len(target_strings[i])
            # print seq_target_string, len(seq_target_string)
            # print seq_ind_words, len(seq_ind_words)
            # print ar_gold_classes[i], len(ar_gold_classes[i])
            continue

        for j in xrange(len(ar_gold_classes[i])):
            gold_j = j-shift_gold
            if gold_j < 0:
                continue
            total += 1
            gold = ar_gold_classes[i][gold_j]
            pred = ar_pred_classes[i][j]
            pred2 = ar_pred2_classes[i][j]
            increment_dict(gold2count, gold)
            increment_dict(pred2count, pred)
            increment_dict(pred22count, pred2)
            gold_classes.append(gold)
            pred_classes.append(pred)
            pred2_classes.append(pred2)
            gold_word = seq_ind_words[gold_j]
            if gold_word not in train_vocab:
                total_unseen += 1
            if gold == pred:
                correct += 1
                correct2 += 1
            else:
                if gold == pred2:
                    correct2 += 1
                if gold_word not in train_vocab:
                    wrong_unseen += 1
            # eval at end of word
            if gold_j > 0 and gold == label2class[Word.WORD_BOUNDARY]:
                total_at_end += 1
                gold_at_end = ar_gold_classes[i][gold_j-1]
                pred_at_end = ar_pred_classes[i][j-1]
                pred2_at_end = ar_pred2_classes[i][j-1]
                if gold_at_end == pred_at_end:
                    correct_at_end += 1
                    correct2_at_end += 1
                elif gold_at_end == pred2_at_end:
                    correct2_at_end += 1

    print 'Error Rate @1:', '%.2f' %(100 - 100.0*correct/total), 'Error Rate @2:', '%.2f' %(100 - 100.0*correct2/total)
    print 'At word ending: Error Rate @1:', '%.2f' %(100 - 100.0*correct_at_end/total_at_end), \
        'Error Rate @2:', '%.2f' %(100 - 100.0*correct2_at_end/total_at_end)
    print 'Error OOV rate (wrong unseen / wrong); fraction of errors in letters in unseen words out of all errors:', '%.2f' %(100.0*wrong_unseen/(total-correct))
    print 'OOV rate (unseen / total); fraction of letters in unseen words out of all letters:', '%.2f' %(100.0*total_unseen/total)
    print 'OOV error rate (wrong unseen / unseen); fraction of errors in letters in unseen words our of all letters in unseen words:', \
        '%.2f' %(100.0*wrong_unseen/total_unseen)

    print '\ntotal:', total, 'unique gold classes:', np.unique(gold_classes), \
        'unique pred classes:', np.unique(pred_classes), 'unique pred2 classes:', np.unique(pred2_classes)
    print 'gold classes counts:', gold2count
    print 'pred classes counts:', pred2count

    print_confusion_matrix(gold_classes, pred_classes, labels)

    print '\nExcluding Word Boundary (_###_)'
    print '================================='
    ex_gold_classes, ex_pred_classes = exclude_classes(gold_classes, pred_classes, [label2class[Word.WORD_BOUNDARY]])
    print_confusion_matrix(ex_gold_classes, ex_pred_classes, labels)
    #
    # print '\nExcluding Word Boundary (_###_) and no diacritic (empty string)'
    # print '================================================================='
    # ex_gold_classes, ex_pred_classes = exclude_classes(gold_classes, pred_classes, \
    #                                                    [label2class[Word.WORD_BOUNDARY], label2class['']])
    # print_confusion_matrix(ex_gold_classes, ex_pred_classes, labels)


if __name__ == '__main__':

    if len(sys.argv) == 4:
        evaluate_file(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 7:
        merge_shadda_and_evaluate_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <gold nc file> <pred csv file> <train nc file>'
        print 'Or, if you want to first merge noshadda and only shadda files, use as follows:'
        print 'USAGE: python ' + sys.argv[0] + '<gold all nc file> <gold noshadda nc file> <gold onlyshadda nc file> \
                                   <pred noshadda csv file> <pred onlyshadda csv file> <train all nc file>'
