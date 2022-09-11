import argparse
import numpy as np
import train
import pickle


tag_to_sign = {'<s>': '.'}
tags_capitalize = ['<s>']


def textify(data: list):
    def add_last_sent_end(data: list[str]):
        if data[len(data) - 1] != '<s>':
            data.append('<s>')

    def textify_tags(data: list[str]):
        for i in range(len(data)):
            if data[i] in tag_to_sign:
                tag = data[i]
                data[i] = tag_to_sign[tag]

                if tag in tags_capitalize and i != len(data) - 1:
                    data[i + 1] = data[i + 1].capitalize()

        return data

    def remove_ugly_spaces(text: str):
        text = text.replace(' .', '.')
        text = text.replace('  ', ' ')
        return text

    add_last_sent_end(data)
    textify_tags(data)
    data[0] = data[0].capitalize()
    text = ' '.join(data)   # join the words into one string
    text = remove_ugly_spaces(text)
    return text


def reverse_dict(d: dict):
    return dict((v, k) for k, v in d.items())


def get_cluster(word: str):
    return df[word_to_df_ind.get(word, randword())][13]


def choose_word(prob_vec, prev_word: str, rand_ctr: int, num_of_tries=5):
    max_cluster_rate = -1
    chosen_word_p_ind = -1

    word_p_inds, rand_ctr_change = roulette_wheel(prob_vec, size=num_of_tries, replace=False)
    for word_p_ind in word_p_inds:
        word_cluster = get_cluster(ind_to_word[word_p_ind])
        prev_word_cluster = get_cluster(prev_word)

        word_cluster_rate = p_clusters[word_cluster][prev_word_cluster]

        if word_cluster_rate > max_cluster_rate:
            chosen_word_p_ind = word_p_ind
            max_cluster_rate = word_cluster_rate

    rand_ctr += rand_ctr_change
    return chosen_word_p_ind, rand_ctr


def roulette_wheel(prob_vec: np.array, size: int, replace: bool):
    prob_vec_f = np.array(prob_vec/sum(prob_vec))
    choice_is_random = False
    try:
        return np.random.choice(len(prob_vec_f), p=prob_vec_f, size=size, replace=True), choice_is_random
    except ValueError as e:
        choice_is_random = True
        return np.random.choice(len(prob_vec_f), size=size, replace=replace), choice_is_random


def read_model(model_path):
    global p, word_to_p_ind, df, word_to_df_ind, p_clusters
    with open(model_path, 'rb') as file_pkl:
        p, word_to_p_ind, df, word_to_df_ind, p_clusters = pickle.load(file_pkl)


def randword():
    return word_to_p_ind[np.random.choice(list(word_to_p_ind.keys()))]


def generate(model_path: str, prefix_text: str, gen_len: int):
    read_model(model_path)
    N = p.ndim

    # preprocess start data
    prefix_text = train.preprocess(prefix_text)
    prefix_text = prefix_text.split()

    global ind_to_word
    ind_to_word = reverse_dict(word_to_p_ind)

    rand_ctr = 0
    for _ in range(gen_len):
        nm1_gram = prefix_text[-(N-1):]
        nm1_gram_ind = []
        for word in nm1_gram:
            if word in word_to_p_ind:
                nm1_gram_ind.append(word_to_p_ind.get(word, randword()))
            else:
                nm1_gram_ind.append(np.random.choice(list(ind_to_word.keys())))

        nm1_gram_ind = tuple(np.array(nm1_gram_ind).T)      # coordiates of (N-1)-gram in p matrix
        n_word_p_vec = p[nm1_gram_ind]   # vector of probabilities for Nth word
        nm1_word = nm1_gram[-1]
        n_word_ind, rand_ctr = choose_word(n_word_p_vec, nm1_word, rand_ctr)
        n_word = ind_to_word[n_word_ind]

        prefix_text.append(n_word)

    gen_text = textify(prefix_text)
    print(gen_text)
    print(f"{rand_ctr} words were randomly selected")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--length', type=int)
    args = parser.parse_args()

    model_path = args.model
    prefix_text = args.prefix
    gen_len = args.length

    generate(model_path, prefix_text, gen_len)