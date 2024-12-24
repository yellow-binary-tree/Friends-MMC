# use convex optimization tools for getting the largest reward
# 2565 qagnn environment
# https://www.cvxpy.org/examples/basic/quadratic_program.html

import os
import sys
import json
import time
import pickle
import random
from tqdm import tqdm
import argparse
from tqdm import trange
from itertools import product
from collections import Counter
from dataclasses import dataclass
import multiprocessing

import cvxpy as cp
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def blockprint(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return results
    return wrapper


@blockprint
def solve(cnn_scores, deberta_scores):
    '''
    cnn_scores: matrix of shape [n, m]
    deberta_scores: matrix of shape [n, n]
    where m = num_speakers, n = num_sents
    '''

    n, m = cnn_scores.shape
    x = cp.Variable(np.prod(cnn_scores.shape), boolean=True)
    constraints = []
    for i in range(n):
        constraints.append(cp.sum(x[i*m: i*m+m]) == 1)

    cnn_objective = cnn_scores.reshape(-1).T @ x.T
    new_deberta_scores = np.zeros((n*m, n*m))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                new_deberta_scores[i*m+k, j*m+k] += deberta_scores[i, j]

    deberta_objecive = cp.quad_form(x, new_deberta_scores) * (1/2)
    objective = cnn_objective + deberta_objecive

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver='GUROBI')
    return x.value.reshape(n, m), problem.status


def convert_cnn_preds_to_matrix(frame_names_list, cnn_scores, label='pred'):
    matrix_list, mappings_list = [], []
    for frame_names in frame_names_list:
        scores = [cnn_scores.get(frame_name, {}) for frame_name in frame_names]
        speakers = set(sum([list(i.keys()) for i in scores], list()))
        speaker_id_mappings = {speaker: i for i, speaker in enumerate(speakers)}
        matrix = np.zeros((len(frame_names), len(speakers)))
        for i, score_dict in enumerate(scores):
            for speaker, score in score_dict.items():
                matrix[i][speaker_id_mappings[speaker]] = score[label]
        matrix_list.append(matrix)
        mappings_list.append(speaker_id_mappings)
    return matrix_list, mappings_list


def solve_func(params):
    i, cnn_scores, deberta_scores, mappings = params
    res, details = solve(cnn_scores, deberta_scores)
    ans = [mappings[np.argmax(line)] for line in res]
    return (i, ans, details)


def inference(cnn_scores_list, deberta_scores_list, mappings_list, weight, n_proc):
    '''
    return string speaker test result
    '''
    params_list = list()
    for i, (cnn_scores, deberta_scores, mappings) in enumerate(zip(cnn_scores_list, deberta_scores_list, mappings_list)):
        mappings = {i: name for name, i in mappings.items()}
        cnn_scores = cnn_scores * weight
        # 为了保证deberta_scores是对称且负定的矩阵
        deberta_scores = (deberta_scores + deberta_scores.T) / 2
        deberta_scores[np.diag_indices_from(deberta_scores)] = -1000
        deberta_scores = deberta_scores * (1 - weight)
        params_list.append((i, cnn_scores, deberta_scores, mappings))

    pool = multiprocessing.Pool(n_proc)
    results = pool.map(solve_func, params_list)
    pool.close()
    pool.join()

    results.sort()
    # print('solve details:')
    # print([i[2] for i in results])
    return [i[1] for i in results]


def load_deberta_scores(fname, label='logits', drop_last=False):
    data = pickle.load(open(fname, 'rb'))

    if label == 'logits':
        # return data['logits']
        matrix_list = list()
        for matrix in data['logits']:
            matrix = matrix.numpy()
            diagonal_mask = np.eye(*matrix.shape, dtype=bool)
            if drop_last:
                matrix[:-1, -1] = 0
                matrix[-1, :-1] = 0
            mean_value = np.mean(matrix[~diagonal_mask])
            matrix_list.append(matrix - mean_value)     # 如果不减去这个均值，所有reward都是正的话，算法会倾向于给所有turn预测同样的结果以拿到所有reward。
        return matrix_list

    elif label == 'labels':     # load gold sent spekaer sim label
        matrix_list = list()
        for logits, labels in zip(data['logits'], data['labels']):
            matrix = np.zeros((logits.size(0), logits.size(1)))
            for src, dst in labels:
                matrix[src, dst] = 1
            diagonal_mask = np.eye(*matrix.shape, dtype=bool)
            if drop_last:
                matrix[:-1, -1] = 0
                matrix[-1, :-1] = 0
            mean_value = np.mean(matrix[~diagonal_mask])
            matrix_list.append(matrix - mean_value)     # 如果不减去这个均值，所有reward都是正的话，算法会倾向于给所有turn预测同样的结果以拿到所有reward。
        return matrix_list
    else:
        raise NotImplementedError()


def load_talknet_scores(asd_folder, track_folder, track_hard_folder, frame_keys_to_clip_keys):
    asd_results = dict()
    for fname in tqdm(os.listdir(asd_folder), desc='loading asd results'):
        if not fname.startswith('s03'): continue
        episode_name = fname[:6]
        episode_data = pickle.load(open(os.path.join(asd_folder, fname), 'rb'))
        for key, val in episode_data.items():
            val = [0 if not isinstance(scores, list) or len(scores) == 0 else np.sum(np.array(scores) > 0) for scores in val]
            asd_results['%s-%s' % (episode_name, key)] = val

    id_name_mapping_all = dict()
    for fname in tqdm(os.listdir(track_folder), desc="loading face track and names"):
        if not fname.startswith('s03'): continue
        episode_name = fname[:6]
        episode_data = pickle.load(open(os.path.join(track_folder, fname), 'rb'))
        for clip_name, track_info in episode_data.items():
            id_name_mapping_all[clip_name] = dict()
            for i, track in enumerate(track_info):
                id_name_mapping_all[clip_name][track['face_track_id']] = track['name']

    ret = dict()
    for frame_key, clip_key in frame_keys_to_clip_keys.items():
        scores = dict()
        for face_i, pred_score in enumerate(asd_results[clip_key]):
            if face_i in id_name_mapping_all[clip_key]:
                scores[id_name_mapping_all[clip_key][face_i]] = pred_score
        if not len(scores):
            ret[frame_key] = dict()
        else:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
            ret[frame_key] = {k: {'pred': v} for k, v in scores.items()}
    print('example pred data:', random.sample(ret.items(), 10))
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--test_cnn_pred_fname', type=str)
    parser.add_argument('--test_talknet_pred_folder', type=str)
    parser.add_argument('--test_talknet_track_folder', type=str)
    parser.add_argument('--test_talknet_track_hard_folder', type=str)
    parser.add_argument('--test_deberta_pred_fname', type=str)
    parser.add_argument('--test_metadata_fname', type=str)
    parser.add_argument('--output_fname', type=str)

    parser.add_argument('--cnn_label', type=str, default='pred', choices=["pred", "label"],
                        help='set to "label" if you want to use ground truth label from visual information')
    parser.add_argument('--deberta_label', type=str, default='logits', choices=["logits", "labels"],
                        help='set to "label" if you want to use ground truth label from text information')
    parser.add_argument('--deberta_only', action='store_true')
    parser.add_argument('--drop_last_sent_sim', action='store_true', help='set the last row/column of deberta prediction to 0 to avoid leaking the last sentence.')

    parser.add_argument('--n_proc', type=int, default=8)
    args = parser.parse_args()
    print(args)

    test_metadata = json.load(open(args.test_metadata_fname))
    test_frame_names_list = [[i['frame'] for i in example] for example in test_metadata]
    test_gold_speakers_list = [[i['speaker'] for i in example] for example in test_metadata]

    if args.test_cnn_pred_fname is not None:
        test_cnn_scores = json.load(open(args.test_cnn_pred_fname))
    elif args.test_talknet_pred_folder is not None and args.test_talknet_track_folder is not None:
        frame_keys_to_clip_keys = dict()
        for example in test_metadata:
            for turn in example:
                frame_keys_to_clip_keys[turn['frame']] = turn['video']          # '%s-%06d-%06d' % (turn['frame'][:6], turn['start'], turn['end'])
        test_cnn_scores = load_talknet_scores(args.test_talknet_pred_folder, args.test_talknet_track_folder, args.test_talknet_track_hard_folder, frame_keys_to_clip_keys)

    test_deberta_scores = load_deberta_scores(args.test_deberta_pred_fname, args.deberta_label, args.drop_last_sent_sim)

    test_cnn_matrix_list, test_mappings_list = convert_cnn_preds_to_matrix(test_frame_names_list, test_cnn_scores, args.cnn_label)
    print('test data lengths:', len(test_cnn_matrix_list), len(test_deberta_scores), len(test_mappings_list))

    pred_speakers_list = inference(test_cnn_matrix_list, test_deberta_scores, test_mappings_list, args.alpha, args.n_proc)
    if args.output_fname is not None:
        json.dump(pred_speakers_list, open(args.output_fname, 'w'))

    acc = accuracy_score(sum(pred_speakers_list, list()), sum(test_gold_speakers_list, list()))
    print('alpha: %.4f, test acc: %.4f' % (args.alpha, acc))