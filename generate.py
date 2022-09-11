"""
NOTES:
    - log probabilities
    - highest_prob=False parameter to decide whether we're composing random strings (fortune wheel), or best fit ones
    - пунктуация и орфография?
"""
import sys

import numpy as np
import train
import pickle


def textify(data: list):
    add_last_sent_end(data)
    textify_tags(data)
    text = ' '.join(data)   # join the words into one string
    text = remove_ugly_spaces(text)
    return text


def add_last_sent_end(data: list[str]):
    if data[len(data)-1] != '<s>':
        data.append('<s>')


tag_to_sign = {'<s>': '.'}
tags_capitalize = ['<s>']


def textify_tags(data: list[str]):
    for i in range(len(data)):
        if data[i] in tag_to_sign:
            tag = data[i]
            data[i] = tag_to_sign[tag]

            if tag in tags_capitalize and i != len(data)-1:
                data[i+1] = data[i+1].capitalize()

    return data


def remove_ugly_spaces(text: str):
    text = text.replace(' .', '.')
    text = text.replace('  ', ' ')
    return text


def reverse_dict(d: dict):
    return dict((v, k) for k, v in d.items())


def choose_word(prob_vec, choice_method: str, rand_ctr: int):
    if choice_method == 'roulette_wheel':
        return roulette_wheel(prob_vec, rand_ctr)
    elif choice_method == 'max_prob':
        return max_prob(prob_vec, rand_ctr)


def roulette_wheel(prob_vec, rand_ctr):
    prob_vec_f = np.array(prob_vec/sum(prob_vec))
    print(prob_vec_f)
    try:
        return np.random.choice(len(prob_vec_f), p=prob_vec_f), rand_ctr
    except ValueError:
        rand_ctr += 1
        return np.random.choice(len(prob_vec_f)), rand_ctr


def max_prob(prob_vec, rand_ctr):
    if np.sum(prob_vec) == 0:
        rand_ctr += 1
        return np.random.choice(len(prob_vec)), rand_ctr
    return prob_vec.argmax(), rand_ctr

def process_input(args):
    model_path = args[1]
    prefix_text = args[2]
    gen_len = int(args[3])

    with open(model_path, 'rb') as file_pkl:
        p, word_to_ind = pickle.load(file_pkl)

    return p, word_to_ind, prefix_text, gen_len


#def generate(p: np.ndarray, word_to_ind: dict, prefix_text: str, gen_ctr: int, choice_method='roulette_wheel'):
def generate(args, choice_method='roulette_wheel'):
    p, word_to_ind, prefix_text, gen_len = process_input(args)
    N = p.ndim

    # preprocess start data
    prefix_text = train.preprocess(prefix_text)
    prefix_text = prefix_text.split()

    print(prefix_text)

    ind_to_word = reverse_dict(word_to_ind)

    print(len(word_to_ind))
    print(len(ind_to_word))

    rand_ctr = 0
    for _ in range(gen_len):
        nm1_gram = prefix_text[-(N-1):]
        nm1_gram_ind = []
        for word in nm1_gram:
            if word in word_to_ind:
                nm1_gram_ind.append(word_to_ind[word])
            else:
                nm1_gram_ind.append(np.random.choice(list(ind_to_word.keys())))
        #print(f"nm1_gram: {nm1_gram}")
        #print(f"nm1_gram_ind: {nm1_gram_ind}")

        nm1_gram_ind = tuple(np.array(nm1_gram_ind).T)      # coordiates of (N-1)-gram in p matrix
        print(f"nm1_gram_ind: {nm1_gram_ind}")
        n_word_p_vec = p[nm1_gram_ind]   # vector of probabilities for Nth word
        print(f"n_word_p_vec: {n_word_p_vec}")
        n_word_ind, rand_ctr = choose_word(n_word_p_vec, choice_method, rand_ctr)
        print(f"n_word_ind: {n_word_ind}")
        n_word = ind_to_word[n_word_ind]
        print(f"n_word_p: {n_word}")

        prefix_text.append(n_word)

    print(' '.join(prefix_text))
    print()
    gen_text = textify(prefix_text)
    print(gen_text)
    print(rand_ctr)

if __name__ == '__main__':
    generate(sys.argv)