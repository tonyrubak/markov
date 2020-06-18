import numpy as np
import json
import re
import string
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz


def build_markov(text):
    bow_dict = {}
    index_dict = {}
    inv_index = []
    # First build a bag-of-words model
    for line in text:
        line_processesd = re.split(
            r"\W+",
            line.translate(
                str.maketrans("", "", string.punctuation)).lower().rstrip())
        line_length = len(line_processesd)
        for (idx,word) in enumerate(line_processesd):
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
    for (idx,word) in enumerate(inv_index):
        word_idx = index_dict[word]
        for next_word in bow_dict[word]:
            if next_word == "_count":
                continue
            next_word_idx = index_dict[next_word]
            model[word_idx,next_word_idx] = bow_dict[word][next_word]
    return (index_dict, inv_index, model.tocsr())


def write_model(model):
    with open("model.json", "w") as out:
        json.dump({"index": model[0], "inverse": model[1]}, out)
    save_npz("model.npz", model[2])


def read_model():
    with open("data/model.json") as file:
        model = json.load(file)
    return (model["index"], model["inverse"], load_npz("data/model.npz"))


def generate_text(prefix, model):
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
