# compute probabilities using MLE n-gram parameter estimation
# (relative frequencies)

"""
NOTES:
    - log probabilities
    - highest_prob=False parameter to decide whether we're composing random strings (fortune wheel), or best fit ones
    - пунктуация и орфография?
"""

import numpy as np
import train

def roulette_wheel(prob_vec, rand_ctr):
    try:
        return np.random.choice(len(prob_vec), p=prob_vec), rand_ctr
    except:
        rand_ctr += 1
        return np.random.choice(len(prob_vec)), rand_ctr


def max_else_rand(prob_vec, rand_ctr):
    if np.sum(prob_vec) == 0:
        rand_ctr += 1
        return np.random.choice(len(prob_vec)), rand_ctr
    return prob_vec.argmax(), rand_ctr


def textify(data: list):
    text = ' '.join(data)
    return text


def generate(p: np.ndarray, word_to_ind: dict, gen_data: str, N: int, gen_ctr: int):
    # preprocess start data
    gen_data = train.preprocess(gen_data)
    gen_data = gen_data.split()

    print(gen_data)

    ind_to_word = dict((v, k) for k, v in word_to_ind.items())

    print(len(word_to_ind))
    print(len(ind_to_word))

    rand_ctr = 0
    for _ in range(gen_ctr):
        nm1_gram = gen_data[-(N-1):]
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
        #n_word_ind, rand_ctr = roulette_wheel(n_word_p_vec, rand_ctr)
        n_word_ind, rand_ctr = max_else_rand(n_word_p_vec, rand_ctr)
        print(f"n_word_ind: {n_word_ind}")
        n_word = ind_to_word[n_word_ind]
        print(f"n_word_p: {n_word}")

        gen_data.append(n_word)

    gen_text = textify(gen_data)
    print(gen_text)
    print(rand_ctr)
