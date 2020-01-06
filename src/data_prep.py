import io
import pickle
from typing import List

import fasttext
import numpy as np
import torch
from bpemb import BPEmb
from laserembeddings import Laser
from utills import read_data


class SentimentData(object):
    def __init__(self, en_x, en_y, es_x, es_y, cm_x, cm_y, test_x, test_y):
        self.en_x = en_x
        self.en_y = en_y
        self.es_x = es_x
        self.es_y = es_y
        self.cm_x = cm_x
        self.cm_y = cm_y
        self.test_x = test_x
        self.test_y = test_y

    @staticmethod
    def read_raw(train_path: str, test_path: str):
        return SentimentData(read_data(train_path, test_path))

    @staticmethod
    def read_pickle(path: str):
        with open(path) as f:
            object = pickle.load(f)
        return object

    def save(self, path: str):
        """
        Pickles the current class. Can be unpicked using read_pickle.
        Args:
            path:

        Returns:
        """
        with open(path, 'w') as f:
            pickle.dump(self, f)


def prep_laser(en_x: List[str], es_x: List[str], cm_x: List[str], test_x: List[str]) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """

    Args:
        en_x:
        es_x:
        cm_x:
        test_x:

    Returns:  en_x, es_x, cm_x, test_x

    """
    laser = Laser()
    en_x = get_laser_embeddings(en_x, "en", laser)
    es_x = get_laser_embeddings(es_x, "es", laser)
    cm_x = get_laser_embeddings(cm_x, "en", laser)
    test_x = get_laser_embeddings(test_x, "en", laser)
    return en_x, es_x, cm_x, test_x


def get_laser_embeddings(x: List[str], lang: str, laser=None) -> np.ndarray:
    if laser is None:
        laser = Laser()
    return laser.embed_sentences(sentences=x, lang=lang)


def prep_xlm(en_x: List[str], es_x: List[str], cm_x: List[str], test_x: List[str], return_all_hiddens: bool = False):
    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
    en_x = get_xlm_embeddings(en_x, xlmr, return_all_hiddens)
    es_x = get_xlm_embeddings(es_x, xlmr, return_all_hiddens)
    cm_x = get_xlm_embeddings(cm_x, xlmr, return_all_hiddens)
    test_x = get_xlm_embeddings(test_x, xlmr, return_all_hiddens)
    del xlmr
    return en_x, es_x, cm_x, test_x


def get_xlm_embeddings(x: List[str], xlmr=None, return_all_hiddens: bool = False):
    if xlmr is None:
        xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')

    embeddings = []
    for sentence in x:
        tokens = xlmr.encode(sentence)
        features = xlmr.extract_features(tokens, return_all_hiddens)
        embeddings.append(features.detach().numpy())
    return embeddings


def prep_multibpe(en_x: List[str], es_x: List[str], cm_x: List[str], test_x: List[str], vs=1000000, dim=300):
    multibpemb = BPEmb(lang="multi", vs=vs, dim=dim)
    en_x = get_multibpe_embeddings(en_x, multibpemb, vs, dim)
    es_x = get_multibpe_embeddings(es_x, multibpemb, vs, dim)
    cm_x = get_multibpe_embeddings(cm_x, multibpemb, vs, dim)
    test_x = get_multibpe_embeddings(test_x, multibpemb, vs, dim)
    return en_x, es_x, cm_x, test_x


def get_multibpe_embeddings(x: List[str], multibpemb=None, vs=1000000, dim=300):
    if multibpemb is None:
        multibpemb = BPEmb(lang="multi", vs=vs, dim=dim)

    embeddings = []
    for sentence in x:
        features = multibpemb.embed(sentence)
        embeddings.append(features)

    embeddings = pad(embeddings, [0 for _ in range(dim)], 32)
    return embeddings


def prep_fasttext(en_x: List[str], es_x: List[str], cm_x: List[str], test_x: List[str], path: str):
    ft = fasttext.load_model(path)
    en_x = get_fasttext_embeddings(en_x, ft, path)
    es_x = get_fasttext_embeddings(es_x, ft, path)
    cm_x = get_fasttext_embeddings(cm_x, ft, path)
    test_x = get_fasttext_embeddings(test_x, ft, path)
    return en_x, es_x, cm_x, test_x


def get_fasttext_embeddings(x: List[str], ft=None, path: str = None):
    if ft is None:
        if path is None:
            raise Exception("Both path and ft can't be None")
        ft = fasttext.load_model(path)

    embeddings = []
    for sentence in x:
        tokens = fasttext.tokenize(sentence)
        representation = []
        for token in tokens:
            representation.append(ft[token])
        embeddings.append(representation)

    embeddings = pad(embeddings, [0 for _ in range(100)], 32)
    return embeddings


def prep_muse():
    pass


def pad(x, pad_value, seq_len=26):
    for i in range(len(x)):
        if len(x[i]) > seq_len:
            x[i] = x[i][:seq_len]
        else:
            x[i] = list(x[i])
            while len(x[i]) < seq_len:
                x[i].append(pad_value)
    return np.array(x)


def get_muse(src_path, tgt_path, nmax):
    src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)

    # make combined embedding mattrix
    embedding_matrix = src_embeddings.copy().tolist()
    embedding_matrix.extend(tgt_embeddings.tolist())
    embedding_matrix = np.array(embedding_matrix)

    # make combined id2word and word2id
    id2word = src_id2word.copy()
    word2id = src_word2id.copy()

    next_id = len(id2word.keys())
    counter = len(id2word.keys())

    to_be_removed_id = []
    common_words = []

    for key in tgt_id2word:
        if tgt_id2word[key] in word2id:
            to_be_removed_id.append(counter)
            common_words.append(tgt_id2word[key])
            embedding_matrix[word2id[tgt_id2word[key]]] = (embedding_matrix[word2id[tgt_id2word[key]]] +
                                                           embedding_matrix[counter]) / 2
        else:
            id2word[next_id] = tgt_id2word[key]
            word2id[tgt_id2word[key]] = next_id
            next_id += 1
        counter += 1

    embedding_matrix = np.delete(embedding_matrix, to_be_removed_id, axis=0)

    return embedding_matrix, id2word, word2id, common_words


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id
