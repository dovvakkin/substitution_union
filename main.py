import torch
import torch.optim as optim

from pathlib import Path
import pandas as pd
import numpy as np
from time import time
import os
import regex as re
from tqdm import tqdm as tqdm
import math
import random
import torch.nn.functional as F
import pymorphy2

def load_substs(substs_fname, limit=None, drop_duplicates=True,
                data_name=None):
    if substs_fname.endswith('+'):
        split = substs_fname.strip('+').split('+')
        p1 = '+'.join(split[:-1])
        s = float(split[-1])
        p2 = re.sub(r'((<mask>)+)(.*?)T', r'T\3\1', p1)
        if p2 == p1:
            p2 = re.sub(r'T(.*?)((<mask>)+)', r'\2\1T', p1)
        print(f'Combining {p1} and {p2}')
        if p1 == p2:
            raise Exception('Cannot conver fname to symmetric one:', p1)
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name)
                          for p in (p1, p2))
        dfinp = dfinp1.merge(dfinp2, on=['context', 'positions'], how='inner',
                             suffixes=('', '_y'))
        res = bcomb3(dfinp, nmasks=len(substs_fname.split('<mask>')) - 1, s=s)
        return res
    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name)


def load_substs_(substs_fname, limit=None, drop_duplicates=True,
                 data_name=None):
    st = time()
    p = Path(substs_fname)
    npz_filename_to_save = None
    print(time() - st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        ss, pp = arr_dict['substs'], arr_dict['probs']
        print(ss.shape, ss.dtype, pp.shape, pp.dtype)
        ss, pp = [list(s) for s in ss], [list(p) for p in pp]
        substs_probs = pd.DataFrame({'substs': ss, 'probs': pp})
        substs_probs = substs_probs.apply(
            lambda r: [(p, s) for s, p in zip(r.substs, r.probs)], axis=1)
        print(substs_probs.head(3))
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']

        print(time() - st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time() - st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace=True, drop=True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(
            list)
        print(type(res_probs))

        npz_filename_to_save = p.parent / (p.name.replace('.bz2', '.npz'))
        if not os.path.isfile(npz_filename_to_save):
            print('saving npz to %s' % npz_filename_to_save)
            np.savez_compressed(p.parent / (p.name.replace('.bz2', '.npz')),
                                probs=res_probs, substs=res_substs)

        # pd.DataFrame({'probs':res_probs, 'substs':res_substs}).to_csv(p.parent/(p.name.replace('.bz2', '.npz')),sep='\t')

    p_ex = p.parent / (p.name + '.input')
    if os.path.isfile(p_ex):
        print(time() - st, 'Loading examples from ', p_ex)
        dfinp = pd.read_csv(p_ex, nrows=limit)
        dfinp['positions'] = dfinp['positions'].apply(pd.eval)
        dfinp['word_at'] = dfinp.apply(
            lambda r: r.context[slice(*r.positions)], axis=1)
    else:
        assert data_name is not None, "no input file %s and no data name provided" % p_ex
        dfinp, _ = load_data(data_name)
        if npz_filename_to_save is not None:
            input_filename = npz_filename_to_save.parent / (
                    npz_filename_to_save.name + '.input')
        else:
            input_filename = p_ex
        print('saving input to %s' % input_filename)
        dfinp.to_csv(input_filename, index=False)

    dfinp['substs_probs'] = substs_probs
    if drop_duplicates:
        dfinp = dfinp.drop_duplicates('context')
    dfinp.reset_index(inplace=True)
    print(dfinp.head())
    return dfinp


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.naive_bayes import BernoulliNB
from joblib import Memory
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster.hierarchical
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.metrics import silhouette_score
from pymorphy2 import MorphAnalyzer
import spacy
import string
import re
from collections import Counter

class Substs_loader:
    def __init__(self, data_name, lemmatizing_method, max_examples=None, delete_word_parts=False,
                 drop_duplicates=True, count_lemmas_weights = False, limit=None):
        self.data_name = data_name
        self.lemmatizing_method = lemmatizing_method
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates
        self.count_lemmas_weights = count_lemmas_weights
        self.translation = str.maketrans('', '', string.punctuation)

        self.dfs = dict()
        self.nf_cnts = dict()
        self.cache = dict()
        self.pattern = re.compile(r'\b\w+\b')

        if lemmatizing_method is not None and lemmatizing_method!='none':
            if 'ru' in data_name:
                self.analyzer = MorphAnalyzer()
            elif 'german' in data_name:
                self.analyzer = spacy.load("de_core_news_sm", disable=['ner', 'parser'])
            elif 'english' in data_name:
                self.analyzer = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
            else:
                assert "unknown data name %s" % data_name

    def get_nf_cnt(self, substs_probs):
        nf_cnt = Counter(nf for l in substs_probs for p, s in l for nf in self.analyze_russian_word(s))
        return nf_cnt

    def analyze_russian_word(self, word, nf_cnt=None):
        word = word.strip()
        if word not in self.cache:
            self.cache[word] = {i.normal_form for i in self.analyzer.parse(word)}

        if nf_cnt is not None and len(self.cache[word]) > 1:  # select most common normal form
            h_weights = [nf_cnt[h] for h in self.cache[word]]
            max_weight = max(h_weights)
            res = {h for i, h in enumerate(self.cache[word]) if h_weights[i] == max_weight}
        else:
            res = self.cache[word]

        return sorted(list(res))

    def analyze(self, word):
        if not word:
            return ['']

        if not word in self.cache:
            spacyed = self.analyzer(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            self.cache[word] = [lemma]
        return self.cache[word]

    def get_lemmas(self, word, nf_cnt=None):
        if 'ru' in self.data_name:
            return self.analyze_russian_word(word, nf_cnt)
        else:
            return self.analyze(word)

    def get_single_lemma(self, word, nf_cnt):
        return self.get_lemmas(word, nf_cnt)[0]

    def preprocess_substitutes(self, substs_probs, target_word, nf_cnt, topk, exclude_lemmas=set(),
                               delete_word_parts=False):
        """
        1) leaves only topk substitutes without spaces inside
        2) applies lemmatization
        3) excludes unwanted lemmas (if any)
        4) returns string of space separated substitutes
        """
        exclude = exclude_lemmas.union({target_word})

        if delete_word_parts:
            res = [word.strip() for prob, word in substs_probs[:topk] if
                   word.strip() and self.pattern.fullmatch(word.strip()) and word[0] == ' ']
        else:
            res = [word.strip() for prob, word in substs_probs[:topk] if
                   word.strip() and self.pattern.fullmatch(word.strip())]

        # TODO: optimise!
        if exclude:
            if self.lemmatizing_method != 'none':
                res = [s for s in res if not set(self.get_lemmas(s)).intersection(exclude)]
            else:
                res = [s for s in res if not s in exclude]

        # if self.lemmatizing_method == 'all':
        #     res = [nf for s in res for nf in self.get_lemmas(s, nf_cnt)]
        # elif self.lemmatizing_method == 'single':
        #     res = [nf for s in res for nf in self.get_single_lemma(s, nf_cnt)]

        if self.lemmatizing_method == 'single':
            res = [self.get_single_lemma(word.strip(), nf_cnt) for word in res]
        elif self.lemmatizing_method == 'all':
            res = [' '.join(self.get_lemmas(word.strip(), nf_cnt)) for word in res]
        # if self.lemmatizing_method == 'single':
        #     res = list(filter(lambda t: not set(t).intersection(exclude), map(self.get_single_lemma, res)))
        #
        #     # res1 = [self.get_single_lemma(word.strip()) for word in res]
        # elif self.lemmatizing_method == 'all':
        #     # res1 = [' '.join(self.get_lemmas(word.strip())) for word in res]
        #     res = list(filter(lambda t: not set(t).intersection(exclude),
        #                         map(lambda x: ' '.join(self.get_lemmas(x)), res)))
        else:
            assert self.lemmatizing_method == 'none', "unrecognized lemmatization method %s" % self.lemmatizing_method

        # if exclude_lemmas:
        #     words = [s for s in words if s not in exclude_lemmas]
        return ' '.join(res)

    def get_substitutes(self, path, topk, data_name=None):

        if data_name is None:
            data_name = self.data_name

        if data_name in self.dfs:
            assert data_name in self.nf_cnts
            subst = self.dfs[data_name]
            nf_cnt = self.nf_cnts[data_name]

        else:
            subst = load_substs(path, data_name=data_name, drop_duplicates=self.drop_duplicates,
                                limit=self.max_examples)

            if self.lemmatizing_method != 'none' and self.count_lemmas_weights and 'ru' in self.data_name:
                nf_cnt = self.get_nf_cnt(subst['substs_probs'])
            else:
                nf_cnt = None

            self.dfs[data_name] = subst
            self.nf_cnts[data_name] = nf_cnt

        subst['substs'] = subst.apply(lambda x: self.preprocess_substitutes(x.substs_probs, x.word, nf_cnt,
                                                  topk,delete_word_parts=self.delete_word_parts), axis=1)
        subst['word'] = subst['word'].apply(lambda x: x.replace('ё', 'е'))
        # if self.max_examples is not None:
        #     subst = subst.sample(frac=1).groupby('word').head(self.max_examples)

        return subst

    def get_substs_pair(self, path1, path2, topk):
        """
        loads subs from path1, path2 and applies preprocessing
        """
        return self.get_substitutes(path1, topk=topk, data_name=self.data_name + '_1'), \
               self.get_substitutes(path2, topk=topk, data_name=self.data_name + '_2' )


def get_vocab(df):
    ws = set()
    for context in df['context']:
        df1 = df[df['context'] == context]
        substs = df1['substs_probs']
        substs = list([v for k, v in substs.tolist()[0]])
        ws.update(substs)

        return list(ws)

def get_single_vector(line, vocab):
    index = {k : v for v, k in line['substs_probs'].tolist()[0]}
    return list([index[word] if word in index else 0 for word in vocab])

def get_vectors_by_vocab(df, vocab):
    vectors = np.zeros((df.shape[0], len(vocab)))
    for n, context in enumerate(df['context_id']):
        vectors[n] = get_single_vector(df[df['context_id'] == context], vocab)
    return vectors

def get_raw_vectors(df):
    vocab = get_vocab(df)
    return get_vectors_by_vocab(df, vocab)

def get_pos_neg_from_gold(golds):
    pos = dict()
    neg = dict()
    
    for n_i, i in enumerate(golds):
        for n_j, j in enumerate(golds):
            if n_i == n_j:
                continue

            if i == j:
                if n_i not in pos:
                    pos[n_i] = [n_j]
                else:
                    pos[n_i].append(n_j)

            if i != j:
                if n_i not in neg:
                    neg[n_i] = [n_j]
                else:
                    neg[n_i].append(n_j)

    return pos, neg

# get_tensor_from_list(l):
#     return torch.

# def get_triplet_loss_(probs, cur_ind, pos_inds, neg_inds, params):
#     all_cur = probs[:, [cur_ind], :]
#     all_pos = probs[:, pos_inds, :]
#     all_neg = probs[:, neg_inds, :]

#     # print(all_cur.dtype)
#     # print(all_pos.dtype)
#     # print(all_neg.dtype)
#     # print('pos_inds\t{}'.format(pos_inds))
#     # print('all_cur\t{}'.format(all_cur.shape))
#     # print('all_pos\t{}'.format(all_pos.shape))
#     # print('params\t{}'.format(params.shape))
#     # print('(all_cur - all_pos).shape\t{}'.format((all_cur - all_pos).shape))
#     # print('(params @ (all_cur - all_neg)).shape\t{}'.format((params @ (all_cur - all_neg)).shape))

#     left = torch.mean((((all_cur - all_pos)* params).sum(0)) ** 2)
#     right = torch.mean((((all_cur - all_neg)* params).sum(0)) ** 2)

#     # print(((((all_cur - all_pos)* params).sum(0)) ** 2).shape)
#     # print(right.item())
    
#     return left.item(),\
#       right.item(),\
#       left - right #+ 0.05 * torch.sum(params ** 2)

def get_triplet_loss(probs, cur_ind, pos_inds, neg_inds, params_mul, params_power):
    all_cur = probs[:, [cur_ind], :]
    all_pos = probs[:, pos_inds, :]
    all_neg = probs[:, neg_inds, :]

    # print(all_cur.dtype)
    # print(all_pos.dtype)
    # print(all_neg.dtype)
    # print('pos_inds\t{}'.format(pos_inds))
    # print('all_cur\t{}'.format(all_cur.shape))
    # print('all_pos\t{}'.format(all_pos.shape))
    # print('params\t{}'.format(params.shape))
    # print('(all_cur - all_pos).shape\t{}'.format((all_cur - all_pos).shape))
    # print('(params @ (all_cur - all_neg)).shape\t{}'.format((params @ (all_cur - all_neg)).shape))

    all_cur_params = (all_cur.pow(params_power) * params_mul).sum(0)
    all_pos_params = (all_pos.pow(params_power) * params_mul).sum(0)
    all_neg_params = (all_neg.pow(params_power) * params_mul).sum(0)
    left = torch.mean(1 - F.cosine_similarity(all_cur_params, all_pos_params))
    right = torch.mean(1 - F.cosine_similarity(all_cur_params, all_neg_params))

    # print(((((all_cur - all_pos)* params).sum(0)) ** 2).shape)
    # print(right.item())
    
    dif = left - right + 0.7
    dif[dif < 0] = 0

    return left.item(),\
      right.item(),\
      dif
       #+ 0.05 * torch.sum(params ** 2)


# def get_triplet_loss(probs, cur_ind, pos_inds, neg_inds, params):
#     all_cur = probs[:, [cur_ind], :]
#     all_pos = probs[:, pos_inds, :]
#     all_neg = probs[:, neg_inds, :]

#     # print(all_cur.dtype)
#     # print(all_pos.dtype)
#     # print(all_neg.dtype)
#     # print('pos_inds\t{}'.format(pos_inds))
#     # print('all_cur\t{}'.format(all_cur.shape))
#     # print('all_pos\t{}'.format(all_pos.shape))
#     # print('params\t{}'.format(params.shape))
#     # print('(all_cur - all_pos).shape\t{}'.format((all_cur - all_pos).shape))
#     # print('(params @ (all_cur - all_neg)).shape\t{}'.format((params @ (all_cur - all_neg)).shape))

#     all_cur_params = (all_cur * params).sum(0) #(all_cur * params).prod(0)
#     all_pos_params = (all_pos * params).sum(0)
#     all_neg_params = (all_neg * params).sum(0)
#     left = torch.mean(1 - F.cosine_similarity(all_cur_params, all_pos_params))
#     right = torch.mean(1 - F.cosine_similarity(all_cur_params, all_neg_params))

#     # print(((((all_cur - all_pos)* params).sum(0)) ** 2).shape)
#     # print(right.item())
    
#     dif = left - right + 0.9
#     dif[dif < 0] = 0

#     return left.item(),\
#       right.item(),\
#       dif
#        #+ 0.05 * torch.sum(params ** 2)


def get_hard_cases(probs, cur_ind, positives, negatives, params_mul, params_power):
    all_cur = probs[:, [cur_ind], :].cpu()
    all_pos = probs[:, positives, :].cpu()
    all_neg = probs[:, negatives, :].cpu()
    params_mul = params_mul.clone().detach().cpu()
    params_power = params_power.clone().detach().cpu()

    all_cur_params = (all_cur.pow(params_power) * params_mul).sum(0)
    all_pos_params = (all_pos.pow(params_power) * params_mul).sum(0)
    all_neg_params = (all_neg.pow(params_power) * params_mul).sum(0)
    left = 1 - F.cosine_similarity(all_cur_params, all_pos_params)
    right = 1 - F.cosine_similarity(all_cur_params, all_neg_params)

    # print(left.shape)
    # print(right.shape)
    left_ind = torch.argsort(left)#.item()
    right_ind = torch.argsort(right, descending=True)#.item()

    return left_ind, right_ind


# def get_hard_cases(probs, cur_ind, positives, negatives, params):
#     all_cur = probs[:, [cur_ind], :].cpu()
#     all_pos = probs[:, positives, :].cpu()
#     all_neg = probs[:, negatives, :].cpu()
#     params = params.clone().detach().cpu()

#     all_cur_params = (all_cur * params).sum(0) #(all_cur * params).prod(0)
#     all_pos_params = (all_pos * params).sum(0)
#     all_neg_params = (all_neg * params).sum(0)
#     left = 1 - F.cosine_similarity(all_cur_params, all_pos_params)
#     right = 1 - F.cosine_similarity(all_cur_params, all_neg_params)

#     # print(left.shape)
#     # print(right.shape)
#     left_ind = torch.argsort(left)#.item()
#     right_ind = torch.argsort(right, descending=True)#.item()

#     return left_ind, right_ind

def balance_pos_neg(positives, negatives, npos,nneg):
    _pos = list()
    _neg = list()
    # print(len(positives),len(negatives))
    # print('---',npos,nneg)
    poses = random.sample(positives, npos)

    for p in poses:
        negs = random.sample(negatives, nneg)
        _pos += [p] * nneg
        _neg += negs

    return torch.tensor(_pos), torch.tensor(_neg)

def train(word_data, EPOCHS, N_ANCH, alpha_mul, alpha_pow):
    device = torch.device("cpu")

    N_TEMPL = 18
    
    params_power = torch.FloatTensor(N_TEMPL,1,1).uniform_(1, 1)
    params_power = torch.tensor(params_power, device=device, requires_grad=True)

    params_mul = torch.FloatTensor(N_TEMPL,1,1).uniform_(1, 1)
    params_mul = torch.tensor(params_mul, device=device, requires_grad=True)
    # params = params

    # params = torch.FloatTensor(N_TEMPL,1,1).uniform_(1, 1)
    # params = torch.tensor(params, device=device, requires_grad=True)
    # params = torch.tensor(params, requires_grad=True)
    
    # optimizer = optim.SGD([params], lr=alpha)
    optimizer_mul = optim.SGD([params_mul], lr=alpha_mul)
    optimizer_power = optim.SGD([params_power], lr=alpha_pow)

    for epoch in range(EPOCHS):
        
        count = 0
        print('epoch:\t{}/{}'.format(epoch + 1, EPOCHS))
        n_batch = 0
        all_left = 0
        all_right = 0
        all_loss = 0
        back = False
        for word in tqdm(word_data):
            back = False
            n_batch = 0
            pos, neg, probs = word_data[word]
            probs = probs.cuda()
            for i in range(probs.shape[1]):
                if i not in pos or i not in neg:
                    continue
                # if len(pos[i]) > N_ANCH and len(neg[i]) > N_ANCH:
                if True:
                    n_batch = (n_batch + 1) % 100
                    _pos, _neg = get_hard_cases(probs, i, pos[i], neg[i], params_mul, params_power)
                    # poses,negs = balance_pos_neg(_pos, _neg, N_ANCH, 5)
                    # for pos, neg in zip([probs, _pos[:N_ANCH]],[_neg[:N_ANCH]]):
                    #     # print(len(pos),len(neg))
                    npos = min(len(pos[i]), 20)
                    __pos, __neg = balance_pos_neg(pos[i], neg[i], npos, 2)
                    # __pos, __neg = torch.tensor([]), torch.tensor([])
                    LEN = min(len(_pos), len(_neg), N_ANCH)
                    if not back:
                        back = True
                        left,right,loss = get_triplet_loss(probs, i, torch.cat([_pos[:LEN], __pos]), torch.cat([_neg[:LEN], __neg]), params_mul, params_power)
                        # left,right,loss = get_triplet_loss(probs, i, _pos[:LEN], _neg[:LEN], params)
                    else:
                        # l,r,lo = get_triplet_loss(probs, i, _pos[:LEN], _neg[:LEN], params)
                        l,r,lo = get_triplet_loss(probs, i, torch.cat([_pos[:LEN], __pos]), torch.cat([_neg[:LEN], __neg]), params_mul, params_power)
                        left += l
                        right += r
                        loss += lo
                    # all_loss += loss
                    
                if (n_batch == 99):
                    # print('batch goes here')
                    # print('{}\t{}\t{}'.format(left,right,loss))
                    back = False
                    count += 1
                    all_left += left
                    all_right += right
                    all_loss += loss.item()
                    # back = False
                    loss.backward()
                    # optimizer.step()
                    optimizer_mul.step()
                    optimizer_power.step()
                    
                
        print(params_mul.view(N_TEMPL))
        print(params_power.view(N_TEMPL))
        # print(torch.sum(params_mul.clone().detach()))
        # print(torch.sum(params.clone().detach()))
        # print(torch.sum(params.clone().detach()))
        print('triplet_loss:\t{}\t{}\t{}'.format(all_left/count,all_right/count,all_loss / count))
    return params

from sklearn.feature_extraction.text import TfidfVectorizer

substs_list = [
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask>-и-T-2ltr1f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-и-<mask>-2ltr2f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask>-или-T-2ltr2f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-или-<mask>-2ltr2f_topk150_fixspacesTrue.npz',

    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask><mask>-и-T-2ltr2f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-и-<mask><mask>-2ltr2f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask><mask>-или-T-2ltr2f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-или-<mask><mask>-2ltr2f_topk150_fixspacesTrue.npz',

    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask><mask><mask>-и-T-2ltr3f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-и-<mask><mask><mask>-2ltr3f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/<mask><mask><mask>-или-T-2ltr3f_topk150_fixspacesTrue.npz',
    '/home/y.kozhevnikov/bts-rnc/russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone/T-или-<mask><mask><mask>-2ltr3f_topk150_fixspacesTrue.npz'
]

un_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
# un_pairs = []

data_name = '/home/y.kozhevnikov/russe-wsi-kit/data/main/bts-rnc/traincsv'
topk = 200
min_df = 0.05
max_df = 0.98

per_word_vocab = dict()
for substitutes_dump in substs_list:
    loader = Substs_loader(data_name, lemmatizing_method='all', drop_duplicates=False, count_lemmas_weights=True)
    df = loader.get_substitutes(substitutes_dump, topk)
    substs_texts = df['substs']

    for word in df.word.unique():
        mask = (df.word == word)
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df)
        vec.fit(substs_texts[mask])
        words = set([w for w in vec.vocabulary_])
        if word not in per_word_vocab:
            per_word_vocab[word] = words
        else:
            per_word_vocab[word].update(words)

data = dict()

for substitutes_dump in substs_list:
    loader = Substs_loader(data_name, lemmatizing_method='all', drop_duplicates=False, count_lemmas_weights=True)
    df = loader.get_substitutes(substitutes_dump, topk)
    substs_texts = df['substs']
    
    per_word_data = dict()
    
    for word in df.word.unique():
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df, vocabulary=per_word_vocab[word])
        mask = (df.word == word)
        golds = df[mask]['gold_sense_id'].tolist()
        pos, neg = get_pos_neg_from_gold(golds)
        vectors = vec.fit_transform(substs_texts[mask]).toarray()
        if word not in data:
            data[word] = dict()
        data[word][substitutes_dump] = pos, neg, vectors

for word in data:
    for pair in un_pairs:
        f,s = pair
        union = data[word][substs_list[f]][2] * data[word][substs_list[s]][2]
        pos, neg, _ = data[word][substs_list[f]]
        data[word][substs_list[f] + '_' + substs_list[s]] = pos, neg, union

new_data = dict()

for word in data:
    vec_list = list([torch.FloatTensor(data[word][k][2]) for k in data[word]])
    vec_tensor = torch.stack(vec_list)
    pos, neg, _ = data[word][next(iter(data[word]))]
    new_data[word] = pos, neg, vec_tensor


train_set = ['лавка', 'лайка', 'лев', 'лира', 'мина', 'мишень', 'обед', 'оклад', 'опушка', 'полис'] #, 'лавка', 'лайка', 'лев', 'лира', 'мина'

train_words = {key : new_data[key] for key in train_set}

params = train(train_words, 20, 15, 0.001, 0.0001)
