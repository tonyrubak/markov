import numpy as np
import json, re, string
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz

def build_markov(text):
    bow_dict = {}
    index_dict = {}
    inv_index = []
    # First build a bag-of-words model
    for line in text:
        for word in re.split(r"\W+",line.translate(str.maketrans("","",string.punctuation)).lower().rstrip()):
            if word in bow_dict:
                bow_dict[word] += 1
            else:
                bow_dict[word] = 1
    nwords = len(bow_dict.keys())
    # Next assign indexes to words based on word counts
    for (i,(k,_)) in enumerate(sorted(bow_dict.items(), key = lambda item: item[1], reverse=True)):
        index_dict[k] = i
        inv_index.append(k)
    # Now actually build the markov chain model
    model = lil_matrix((nwords,nwords))
    for line in text:
        words = re.split(r"\W+",line.translate(str.maketrans("","",string.punctuation)).lower().rstrip())
        for i,w in enumerate(words[0:-1]):
            model[index_dict[w],index_dict[words[i+1]]] += 1
    return (index_dict,inv_index,model.tocsr())

def write_model(model):
    with open("model.json","w") as out:
        json.dump({"index": model[0], "inverse": model[1]},out)
    save_npz("model.npz",model[2])

def read_model():
    with open("data/model.json") as file:
        model = json.load(file)
    return (model["index"],model["inverse"],load_npz("data/model.npz"))

def generate_text(prefix,model):
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
        s = matrix[word,:].sum()
        if s == 0:
            break
        else:
            row = matrix[word,:] / matrix[word,:].sum()
        while p < r:
            p += row[0,j]
            j += 1
        res = " ".join([res, inv_index[j-1]])
        word = j - 1
    return res

with open("data/log") as file:
    text = file.readlines()

mc = build_markov(text)
for _ in range(1000):
    print(generate_text("synergy",mc))
write_model(mc)