#!/usr/bin/env python3
"""
CS 1571 Project 4: Naive Bayes - Spam Detection
Zac Yu (zhy46@)
"""

from argparse import ArgumentParser, FileType
from itertools import chain
from numpy import array, count_nonzero, genfromtxt, invert, mean, savetxt


def main():
    """Main function"""
    parser = ArgumentParser(description='Perform spam prediction.')
    parser.add_argument('dataset', type=FileType('r'),
                        help='the spam dataset in the UCI spambase format')

    args = parser.parse_args()
    sb_raw = genfromtxt(args.dataset.name, delimiter=',')
    feature_count = len(sb_raw[0][:-1])
    print('<!-- This output is best viewed with a Markdown renderer -->')
    print('# Data Splitting')
    sb_groups = [array(sb_raw[i::5]) for i in range(5)]
    sb_groups_pos_count = [count_nonzero(
        sb_groups[i], axis=0)[-1] for i in range(5)]
    sb_total_pos = sum(sb_groups_pos_count)
    print('## Table')
    print('Iteration | Pos in Train | Neg in Train | Pos in Dev | Neg in Dev')
    print('---' + ' | ----' * 4)
    for i in range(5):
        j = i
        # j = 4 - i
        dev_pos = sb_groups_pos_count[j]
        dev_neg = len(sb_groups[j]) - dev_pos
        train_pos = sb_total_pos - dev_pos
        train_neg = len(sb_raw) - sb_total_pos - dev_neg
        print('%-3d | %4d | %4d | %4d | %4d' %
              (i + 1, train_pos, train_neg, dev_pos, dev_neg))
    print('## Files')

    for i in range(5):
        j = i
        # j = 4 - i
        savetxt('%d_Train.txt' % (i + 1),
                [r for rows in sb_groups[:j] + sb_groups[j + 1:]
                 for r in rows],
                delimiter=',', fmt='%g')
        savetxt('%d_Dev.txt' % (i + 1), sb_groups[j], delimiter=',', fmt='%g')
        print('* Saved %d\\_Train.txt and %d\\_Dev.txt' % (i + 1, i + 1))
    print()
    print('# Probability Calculation')
    col_means = mean(sb_raw, axis=0)
    prob_all = []
    print('## Table')
    print('Iteration | ' + ' | '.join(['Pr(F%d <= mu%d : spam) | Pr(F%d > mu%d : spam) | Pr(F%d <= mu%d : ~spam) | Pr(F%d > mu%d : ~spam)' % (
        i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1) for i in range(feature_count)]))
    print('---' + ' | ------' * (feature_count * 4))
    for i in range(5):
        pos_count = 0
        neg_count = 0
        small_pos_feature = [0] * feature_count
        small_neg_feature = [0] * feature_count
        for j in chain(range(0, i), range(i + 1, 5)):
            # for j in chain(range(0, 4 - i), range(5 - i, 5)):
            small = sb_groups[j] <= col_means
            for k in range(feature_count):
                small_pos_feature[k] += count_nonzero(
                    small[:, k] & invert(small[:, -1]))
                small_neg_feature[k] += count_nonzero(
                    small[:, k] & small[:, -1])
            pos_count += sb_groups_pos_count[j]
            neg_count += len(sb_groups[j]) - sb_groups_pos_count[j]
        prob_features = [0] * (feature_count * 4)
        for k in range(feature_count):
            prob_features[4 * k] = small_pos_feature[k] / pos_count
            prob_features[4 * k + 1] = 1 - prob_features[4 * k]
            prob_features[4 * k + 2] = small_neg_feature[k] / neg_count
            prob_features[4 * k + 3] = 1 - prob_features[4 * k + 2]
        print('%-3d | ' % (i + 1) +
              ' | '.join(['%.6f' % p for p in prob_features]))
        prob_all.append(prob_features)
    print('## File')
    savetxt('prob.csv', prob_all, delimiter=',', fmt='%g')
    print('* Saved prob.csv')
    print()
    print('# Prediction Error Rates')
    print('## Table')
    print('Fold | False Pos | False Neg | Overall')
    print('---' + ' | --------' * 3)
    pred_error_rate_all = []
    for i in range(5):
        small = sb_groups[i] <= col_means
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for entry in small:
            prob_pos = col_means[-1]
            prob_neg = 1 - col_means[-1]
            for k in range(feature_count):
                if entry[k]:
                    pos_mult = prob_all[i][4 * k] or 0.0014
                    neg_mult = prob_all[i][4 * k + 2] or 0.0014
                else:
                    pos_mult = prob_all[i][4 * k + 1] or 0.0014
                    neg_mult = prob_all[i][4 * k + 3] or 0.0014
                prob_pos *= pos_mult
                prob_neg *= neg_mult
            pred = (prob_pos / (prob_pos + prob_neg)) > 0.5
            true_pos += pred and (not entry[-1])
            true_neg += (not pred) and entry[-1]
            false_pos += pred and entry[-1]
            false_neg += (not pred) and (not entry[-1])
        error_rate = array([false_pos / (false_pos + true_neg),
                            false_neg / (false_neg + true_pos),
                            (false_pos + false_neg) / len(sb_groups[i])])
        pred_error_rate_all.append(error_rate)
        print('%-3d | ' % (i + 1) +
              ' | '.join(['%.6f' % r for r in error_rate]))
    pred_error_rate_mean = mean(array(pred_error_rate_all), axis=0)
    print('Avg | ' + ' | '.join(['%.6f' % r for r in pred_error_rate_mean]))
    print('## Raw')
    for i, rates in enumerate(pred_error_rate_all):
        print('    Fold_%d, ' % (i + 1) + ', '.join(['%g' % r for r in rates]))
    print('    Avg, ' + ', '.join(['%g' % r for r in pred_error_rate_mean]))
    print('## File')
    savetxt('error_rates.csv', pred_error_rate_all, delimiter=',', fmt='%g')
    print('* Saved error_rates.csv')


if __name__ == '__main__':
    main()
