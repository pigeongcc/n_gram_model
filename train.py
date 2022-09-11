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
from generate import generate


def upload_corpus(corpus_root: str):
    data = ''

    def read_txt(filepath):
        with open(filepath, 'rb') as file:
            result = chardet.detect(file.read(100000))
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


def remove_punctuation(data: str):
    for sign in punctuation_none:
        data = data.replace(sign, '')
    for sign in punctuation_space:
        data = data.replace(sign, ' ')
    for sign in punctuation_tag_sentence:
        tag = ' </s> <s>'  # sentence ended, new sentence began
        data = data.replace(sign, tag)

    return data


def preprocess(data: str):
    data = data.lower()  # converting to lowercase
    data = remove_punctuation(data)
    data = data.replace('ё', 'е')

    return data


def initialize_p_ind(words: set, N: int):
    ind = {}
    index = 0
    for word in words:
        ind[word] = index
        index += 1
    if verbose:
        print(ind)

    p_shape = tuple(len(ind) for _ in range(N))  # dimensions of p
    #p = np.zeros(p_shape)
    p = np.zeros(p_shape, dtype=np.uint8)

    return p, ind


def fill_p(p: np.ndarray, ind: dict, data: list[str], normalize=True):
    """
    :param data: data.split(). A list of words ordered as in text source
    """
    N = len(p.shape)
    n_grams = [data[i: i + N] for i in range(len(data) - N + 1)]
    if verbose:
        print(f"n-grams:\n{n_grams}\n")

    for n_gram in n_grams:
        n_gram_ind = [ind[word] for word in n_gram]
        p[tuple(np.array(n_gram_ind).T)] += 1
        #print(f"~~~~~~~ added 1 to p at {n_gram_ind}")
        #print(f"p now:\n{p}\n")
    if verbose:
        print(f"~~~~~~~ p before normalization:\n{p}\n")
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


        num_n_grams = len(n_grams)
        if verbose:
            print(f"number of words in data is {len(data)}")
            print(f"number of n_grams is {num_n_grams}")
            print(f"number of nm1_grams is {len(nm1_grams)}\n")

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
            ind_tuple = tuple(ind[word] for word in words_tuple)
            #print(f"~~~~~~~ IND TUPLE: {ind_tuple}")
            #print(f"~~ dividing  p[{ind_tuple}] by {nm1_grams_dict[words_tuple]} ...")
            #print(f"it was   {p[ind_tuple]}")
            #p[ind_tuple] /= nm1_grams_dict[words_tuple]
            #print(f"now it's {p[ind_tuple]}\n")

            prob_vec = np.array(p[ind_tuple] * 255 / nm1_grams_dict[words_tuple])
            p[ind_tuple] = quantize(prob_vec, [i for i in range(0, 255, 12)])
            """
            print("QUANT:")
            print(prob_vec)
            print(p[ind_tuple])
            """

        if verbose:
            print(f"nm1-grams:\n{nm1_grams}\n")
            print(f"nm1-grams-ctr:\n{nm1_grams_dict}\n")

        """
        p_sums = p.sum(axis=0)
        p = np.moveaxis(p, 1, 0)
        p = np.array([p[i]/(p_sums[i] if p_sums[i] != 0 else 1) for i in range(len(p))])
        p = np.moveaxis(p, 0, 1)

        # p_sums = p.sum(axis=tuple(range(p.ndim - 1)))
        #for i in range(p.shape[-1]):
            #p_sum = p_sums[i]
        #for p_sum in p_sums:
        super_tpl = tuple(range(p.ndim))
        print(f"~~~~~~~ super_tpl is {super_tpl}")
        for super_ind in tuple(range(p.ndim - 1)):
            p_sum = p_sums[super_ind]
            if p_sum > 1:
                axis = [_ for _ in range(p.ndim-1)]
                #axis.append(i)
                axis = tuple(np.array(axis).T)
                p[super_ind] /= p_sum
                print(f"~~~~~~~ divided by {p_sum} at {axis}")
                print(f"p now:\n{p}\n")
        """


def train(args):
    # upload texts from the corpus into data variable
    corpus_root = args[1]
    data = upload_corpus(corpus_root)
    #data = 'a b c b c'
    #data = 'a b c b a c b a d'
    if verbose:
        print(f"~~~~~~~ CORPUS IS READ ~~~~~~~\n{data}")

    # preprocess the text data
    data = preprocess(data)
    if verbose:
        print(f"~~~~~~~ DATA IS PREPROCESSED ~~~~~~~\n{data}")

    """
    initialize an N-dimensional array p to store probabilities with 0s.
    ind{'word' : 0} is a dict containing pairs word-index for each word from the data.
    index is then used to access the word in array p
    """
    data_split = data.split()
    words = set(data_split)
    N = int(args[2])  # N-gram model
    p, ind = initialize_p_ind(words, N)

    fill_p(p, ind, data_split, True)
    if verbose:
        with np.printoptions(threshold=sys.maxsize):
            print(f"~~~~~~~ p IS FILLED ~~~~~~~\np\n{p}\n\nind\n{ind}")
        print(f"p shape is {p.shape}\n~~~~~~~ GENERATOR ~~~~~~~\n")

    return p, ind, data, N


verbose = 0
if __name__ == '__main__':
    start_data = "для моделирования языка"
    gen_ctr = 150

    p, ind, data, N = train(sys.argv)
    generate(p, ind, start_data, N, gen_ctr)
