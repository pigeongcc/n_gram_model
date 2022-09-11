"""
- requirements.txt
- ideas to improve the model:
    1. hash word strings and store the dict {hash-number: word-string} on the disk to save RAM
    2. come up with a custom data type to decrease the size of each matrix p element
        (1 byte is a lot to store a probability value)
    3.

"""
import sys
import os
import numpy as np
import chardet
import pickle
import pymorphy2
from sklearn.cluster import DBSCAN
from sklearn import metrics


def upload_corpus(corpus_root: str):
    data = ''

    def read_txt(filepath):
        with open(filepath, 'rb') as file:
            result = chardet.detect(file.read(100))
        encoding = result['encoding']

        with open(filepath, 'r', encoding=encoding) as file:
            return file.read()

    for file in os.listdir(corpus_root):
        if file.endswith(".txt"):
            filepath = f"{corpus_root}/{file}"
            data += read_txt(filepath) + '\n'

    return data


punctuation_none = "\"#$%',:;@`~”“《》«»’‘№§"  # punctuation signs to remove
punctuation_space = "&()*+/<=>{|}[\]^_"  # punctuation signs to replace with spaces
punctuation_tag_sentence = ".!?…"  # punctuation signs to replace with special tags
tag_sent_end = '<s>'
sign_to_tag = {'.': tag_sent_end,
               '!': tag_sent_end,
               '?': tag_sent_end,
               '…': tag_sent_end
               }


def remove_punctuation(data: str):
    for sign in punctuation_none:
        data = data.replace(sign, '')
    for sign in punctuation_space:
        data = data.replace(sign, ' ')
    for sign in sign_to_tag:
        data = data.replace(sign, f' {sign_to_tag[sign]}')

    return data


def preprocess(data: str):
    data = data.lower()  # converting to lowercase
    data = remove_punctuation(data)
    data = data.replace('ё', 'е')

    return data


def initialize_ngram(words: set, N: int):
    word_to_ind = {}
    index = 0
    for word in words:
        word_to_ind[word] = index
        index += 1

    p_shape = tuple(len(word_to_ind) for _ in range(N))  # dimensions of p
    p = np.zeros(p_shape, dtype=np.uint8)

    return p, word_to_ind


def fill_p(p: np.ndarray, word_to_ind: dict, data: str, normalize=True):
    N = len(p.shape)
    data = data.split()
    n_grams = [data[i: i + N] for i in range(len(data) - N + 1)]

    for n_gram in n_grams:
        n_gram_ind = [word_to_ind[word] for word in n_gram]
        p[tuple(np.array(n_gram_ind).T)] += 1
    if normalize:
        # normalize by dividing values in Nth dimension by the number of (N-1)-gram occurrences
        nm1_grams = [data[i: i + (N - 1)] for i in range(len(data) - (N ) + 1)]
        nm1_grams_dict = {}
        for nm1_gram in nm1_grams:
            nm1_gram_items = tuple(nm1_gram)
            if nm1_gram_items not in nm1_grams_dict:
                nm1_grams_dict[nm1_gram_items] = 1
            else:
                nm1_grams_dict[nm1_gram_items] += 1

        def quantize(prob_vec_f, quants):
            prob_vec_int = np.zeros(prob_vec_f.shape, dtype=np.uint8)
            for i in range(len(prob_vec_f)):
                prob_int = None
                best_diff = None
                for quant in quants:
                    diff = abs(quant - prob_vec_f[i])
                    if prob_int is None or diff < best_diff:
                        prob_int = quant
                        best_diff = diff

                prob_vec_int[i] = prob_int

            return prob_vec_int

        for words_tuple in nm1_grams_dict.keys():
            ind_tuple = tuple(word_to_ind[word] for word in words_tuple)

            prob_vec = np.array(p[ind_tuple] * 255 / nm1_grams_dict[words_tuple])
            p[ind_tuple] = quantize(prob_vec, [i for i in range(0, 255, 12)])


def process_input(args):
    corpus_root = args[1]
    model_path = args[2]
    N = int(args[3])  # N-gram model
    return corpus_root, model_path, N


def save_model(model_path: str, vars):
    with open(model_path, 'wb') as file_pkl:
        pickle.dump(vars, file_pkl)


# dict narrowing the range of POS used
POS_TO_FEAT = {
    "NOUN": "POS_NOUN",
    "NPRO": "POS_NOUN",
    "ADJF": "POS_ADJ",
    "ADJS": "POS_ADJ",
    "COMP": "POS_ADJ",
    "PRTF": "POS_ADJ",
    "PRTS": "POS_ADJ",
    "VERB": "POS_VERB",
    "GRND": "POS_VERB",
    "INFN": "POS_INFN",
    "NUMR": "POS_NUMR",
    "ADVB": "POS_ADV",
    "PRED": "POS_PRED",
    "PREP": "POS_PREP",
    "CONJ": "POS_CONJ",
    "PRCL": "POS_PRCL",
    "INTJ": "POS_INTJ",
    "OTHER": "POS_OTHER"
}
# feature-to-index dict
FEAT_TO_IND = {
    "POS_NOUN": 0,
    "POS_ADJ": 1,
    "POS_VERB": 2,
    "POS_NUMR": 3,
    "POS_ADV": 4,
    "POS_PRED": 5,
    "POS_PREP": 6,
    "POS_CONJ": 7,
    "POS_PRCL": 8,
    "POS_INTJ": 9,
    "POS_INFN": 10,
    "POS_OTHER": 11,
    "NUMBER": 12,
    "CLUSTER": 13
}


def initialize_ml(num_of_examples: int):
    df = [ [0 for _ in range(len(FEAT_TO_IND))] for _ in range(num_of_examples)]
    return df


morph = pymorphy2.MorphAnalyzer()


def parse(word: str):
    parse_results = morph.parse(word)
    if len(parse_results) == 0:
        return None
    return parse_results[0].tag


WORD_TO_DF_IND = {}


def fill_df(df: list, words: list):
    for i in range(len(words)):
        word = words[i]
        WORD_TO_DF_IND[word] = i

        tag = parse(word)

        pos = tag.POS
        pos_ind = FEAT_TO_IND[POS_TO_FEAT.get(pos, "POS_OTHER")]
        df[i][pos_ind] = 1

        number = tag.number
        num_ind = FEAT_TO_IND["NUMBER"]
        df[i][num_ind] = 0 if number == 'sing' else 1


def clusterize(df: list):
    db = DBSCAN().fit(df)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, labels))

    for i in range(len(df)):
        df[i][FEAT_TO_IND["CLUSTER"]] = labels[i]

    return set(labels)


def compute_probs_clusters(data: str, df: list, clusters: set):
    data = data.split()
    num_of_clusters = len(clusters)
    p_clusters = [[0 for _ in range(num_of_clusters)] for _ in range(num_of_clusters)]
    for i in range(1, len(data)):
        word = data[i]
        word_prev = data[i-1]
        try:
            word_ind = WORD_TO_DF_IND[word]
            word_prev_ind = WORD_TO_DF_IND[word_prev]
        except KeyError:
            continue
        word_cluster = df[word_ind][FEAT_TO_IND["CLUSTER"]]
        word_prev_cluster = df[word_prev_ind][FEAT_TO_IND["CLUSTER"]]
        p_clusters[word_cluster][word_prev_cluster] += 1
    return p_clusters


def get_words_set(data: str):
    return set(data.split())


def fit_ml_model(data: str):
    # fill the df (table) with the words features
    words = list(get_words_set(data))
    df = initialize_ml(len(words))  # an old habit to call it df...
    fill_df(df, words)
    # run clusterization algorithm
    clusters = clusterize(df)
    # compute a matrix of clusters connection between each other using the text
    p_clusters = compute_probs_clusters(data, df, clusters)
    return df, p_clusters


def fit(args):
    corpus_root, model_path, N = process_input(args)
    # upload texts from the corpus into data variable
    data = upload_corpus(corpus_root)

    # preprocess the text data
    data = preprocess(data)

    """
    initialize an N-dimensional array p to store probabilities with 0s.
    word_to_ind{'word' : 0} is a dict containing pairs word-index for each word from the data.
    index is then used to access the word in array p
    """
    words = get_words_set(data)
    p, word_to_ind = initialize_ngram(words, N)

    fill_p(p, word_to_ind, data, True)

    df, p_clusters = fit_ml_model(data)

    save_model(model_path, [p, word_to_ind, df, WORD_TO_DF_IND, p_clusters])


if __name__ == '__main__':
    fit(sys.argv)

