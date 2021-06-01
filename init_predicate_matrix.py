import json
import numpy as np
import torch

def semantic_embedding():
    with open('../data/viroi_json/class_dict.json', 'r') as f:
        classes = json.load(f)

    objects = [None] * len(classes)
    for clas in classes.values():
        objects[clas['class_id'] - 1] = clas['name']

    word2int = {}
    for i, obj in enumerate(objects):
        word2int[obj] = i

    glove = {}
    vocab = len(objects)
    matrix_len = vocab
    weights_matrix = np.zeros((matrix_len, 300))
    # valid_vocab=[]
    # all_vocab=[]
    with open("glove.6B/glove.6B.300d.txt", 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            glove[word] = vect
            # all_vocab.append(word)
    invald_vocab={#"traffic light:",
    "fire hydrant":"hydrant",
    "stop sign": "sign",
    #"parking meter":"",
    "sports ball":"ball",
    "baseball bat":"bat",
    "baseball glove":"glove",
    "tennis racket":"racket",
    "wine glass":"glass",
    "hot dog":"hotdog",
    "potted plant":"plant",
    "dining table":"table",
    "cell phone":"cell-phone",
    "teddy bear":"teddybears",
    "hair drier":"drier",
    "door-stuff":"door",
    "floor-wood":"floor",
    "mirror-stuff":"mirror",
    "playingfield":"playfield",
    "wall-brick":"wall",
    "wall-stone":"wall",
    "wall-tile":"wall",
    "wall-wood":"wall",
    "water-other":"water",
    "window-blind":"window",
    "window-other":"window",
    "tree-merged":"trees",
    "fence-merged":"fence",
    "ceiling-merged":"ceiling",
    "sky-other-merged":"sky",
    "cabinet-merged":"cabinet",
    "table-merged":"table",
    "floor-other-merged":"floor",
    "pavement-merged":"pavement",
    "mountain-merged":"mountain",
    "grass-merged":"grass",
    "dirt-merged":"dirt",
    "paper-merged":"paper",
    "food-other-merged":"food",
    "building-other-merged":"building",
    "rock-merged":"rock",
    "wall-other-merged":"wall",
    "rug-merged":"rug"}
    for i, obj in enumerate(objects):
        try:
            if obj in invald_vocab:
                g_obj=invald_vocab[obj]
            else:
                g_obj=obj
            weights_matrix[word2int[obj]] = glove[g_obj]
            # valid_vocab.append(obj)
        except KeyError:
            print(obj)
            weights_matrix[word2int[obj]] = np.random.normal(
                scale=0.6, size=(300,))
    weights_matrix = torch.Tensor(weights_matrix)
    # json.dump(valid_vocab,open("glove_valid_vocab.json",'w'))
    # json.dump(all_vocab,open("glove_all_vocab.json",'w'))
    torch.save({"semantic_embedding":weights_matrix},"semantic_embedding.pth")

def predicate_embedding():
    with open('../data/viroi_json/relation_dict.json', 'r') as f:
        classes = json.load(f)

    objects = [None] * len(classes)
    for clas in classes.values():
        objects[clas['relation_id'] - 1] = clas['name'].replace(" ","-")

    word2int = {}
    for i, obj in enumerate(objects):
        word2int[obj] = i

    glove = {}
    vocab = len(objects)
    matrix_len = vocab
    weights_matrix = np.zeros((matrix_len, 300))
    # valid_vocab=[]
    # all_vocab=[]
    with open("glove.6B/glove.6B.300d.txt", 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            glove[word] = vect
            # all_vocab.append(word)
    invald_vocab = {

    }
    for i, obj in enumerate(objects):
        try:
            if obj in invald_vocab:
                g_obj = invald_vocab[obj]
            else:
                g_obj = obj
            if g_obj in glove:
                weights_matrix[word2int[obj]] = glove[g_obj]
            else:
                g_objs=g_obj.split("-")
                glove_embedding=glove[g_objs[0]]
                for i in range(1,len(g_objs)):
                    glove_embedding=glove_embedding+glove[g_objs[i]]
                weights_matrix[word2int[obj]] = glove_embedding
            # valid_vocab.append(obj)
        except KeyError:
            print(obj)
            weights_matrix[word2int[obj]] = np.random.normal(
                scale=0.6, size=(300,))
    weights_matrix = torch.Tensor(weights_matrix)
    # json.dump(valid_vocab,open("glove_valid_vocab.json",'w'))
    # json.dump(all_vocab,open("glove_all_vocab.json",'w'))
    torch.save({"predicate_embedding": weights_matrix}, "predicate_embedding.pth")

# predicate_embedding()
semantic_embedding()
