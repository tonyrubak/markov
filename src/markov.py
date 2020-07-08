import numpy as np
import json
import re
import string
import time
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz

def parse_word(word):
    trans = str.maketrans("","",string.punctuation)
    if not (word.startswith("<:") or word.startswith(":")):
        return word.translate(trans).lower()
    return word

def parse_line(text):
    return [*filter(lambda x: x != "",
                    map(parse_word,re.split(r"\s+", text.rstrip())))]

def build_markov(text):
    bow_dict = {}
    index_dict = {}
    inv_index = []
    # First build a bag-of-words model
    for line in text:
        line_processesd = parse_line(line)
        line_length = len(line_processesd)
        for (idx, word) in enumerate(line_processesd):
            if word in bow_dict:
                bow_dict[word]["_count"] += 1
            else:
                bow_dict[word] = {"_count": 1}
            if (idx < line_length - 1):
                next_word = line_processesd[idx+1]
                if next_word in bow_dict[word]:
                    bow_dict[word][next_word] += 1
                else:
                    bow_dict[word][next_word] = 1
    nwords = len(bow_dict.keys())
    # Next assign indexes to words based on word counts
    for (i, (k, _)) in enumerate(sorted(bow_dict.items(),
                                        key=lambda item: item[1]["_count"],
                                        reverse=True)):
        index_dict[k] = i
        inv_index.append(k)
    # Now actually build the markov chain model
    model = lil_matrix((nwords, nwords))
    for (idx, word) in enumerate(inv_index):
        word_idx = index_dict[word]
        for next_word in bow_dict[word]:
            if next_word == "_count":
                continue
            next_word_idx = index_dict[next_word]
            model[word_idx, next_word_idx] = bow_dict[word][next_word]
    return (index_dict, inv_index, model.tocsr())


def write_model(model):
    with open("model.json", "w") as out:
        json.dump({"index": model[0], "inverse": model[1]}, out)
    save_npz("model.npz", model[2])


def read_model():
    with open("data/model.json") as file:
        model = json.load(file)
    return (model["index"], model["inverse"], load_npz("data/model.npz"))


def generate_text(model, prefix):
    length = 10
    word_index = model[0]
    inv_index = model[1]
    matrix = model[2]
    word = word_index[prefix]
    res = prefix
    for _ in range(length):
        r = np.random.uniform()
        p = 0
        j = 0
        s = matrix[word, :].sum()
        if s == 0:
            break
        else:
            row = matrix[word, :] / matrix[word, :].sum()
        while p < r:
            p += row[0, j]
            j += 1
        res = " ".join([res, inv_index[j-1]])
        word = j - 1
    return res


def update_model(model, text):
    word_dict = model[0]
    inv_index = model[1]
    matrix = model[2].tolil()
    for line in text:
        line_processed = parse_line(line)
        line_length = len(line_processed)
        for (idx, word) in enumerate(line_processed):
            if idx >= line_length - 1:
                break
            if word in word_dict:
                word_idx = word_dict[word]
            else:
                inv_index.append(word)
                word_idx = word_dict[word] = len(inv_index) - 1
                matrix.resize((word_idx + 1, word_idx + 1))
            next_word = line_processed[idx+1]
            if next_word in word_dict:
                next_idx = word_dict[next_word]
            else:
                inv_index.append(next_word)
                next_idx = word_dict[next_word] = len(inv_index) - 1
                matrix.resize((next_idx + 1, next_idx + 1))
            matrix[word_idx, next_idx] += 1
    return (word_dict, inv_index, matrix.tocsr())


def reindex_model(model):
    inv_index = model[1]
    matrix = model[2]
    new_matrix = lil_matrix(matrix.shape)
    word_counts = zip(inv_index, matrix.sum(axis=1))
    new_inv_index = [k for (k, _) in sorted(word_counts,
                                            key=lambda item: item[1],
                                            reverse=True)]
    new_word_dict = dict([(word, idx)
                          for (idx, word) in enumerate(new_inv_index)])
    new_idxs = [*map(lambda x: new_word_dict[inv_index[x]],
                     range(matrix.shape[0]))]
    for row, col in zip(*matrix.nonzero()):
        new_matrix[new_idxs[row], new_idxs[col]] = matrix[row, col]
    return (new_word_dict, new_inv_index, new_matrix.tocsr())


mc = reindex_model(mc)
print(f"Load factor: {mc[2].nnz/(mc[2].shape[0] ** 2)}")

with open("data/log") as file:
    text = file.readlines()


def time_construction():
    print("Constructing markov chain model...")
    st = time.perf_counter()
    mc = build_markov(text)
    end = time.perf_counter()
    print(f"Constructed model in {end - st} seconds.")


for _ in range(1000):
    print(generate_text("dave", mc))
write_model(mc)

generate_text(mc,"dave")
[word for word in mc[0].keys() if "clap" in word]
