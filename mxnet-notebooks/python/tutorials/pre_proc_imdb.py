# imports and env-variable
import numpy as np
import os
import glob
import nltk
from nltk.tokenize import RegexpTokenizer
import urllib3
import tarfile

BASE_DIR = '/home/ubuntu/CyrusProjects/mxnet-notebooks/python/tutorials'
PATH='/aclImdb'

with open('aclImdb/imdb.vocab') as vocab_file:
    vocab_array = vocab_file.readlines()
vocab_array = [x.strip() for x in vocab_array]
print(len(vocab_array))


with open(BASE_DIR + PATH + '/imdb.vocab') as f:
    dic = f.readlines()
dic= [x.strip() for x in dic]
dic = np.array(np.sort(dic))
dic = np.unique(dic)
dic = dic.tolist()


# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
def sentence_array_creator(path, root_dir):
    os.chdir(path)
    sentences = []
    for file in list(glob.glob("*.txt")):
        with open(file, 'r') as f:
            sentences.append(f.readline().strip().lower())
    return sentences
    os.chdir(root_dir)

os.chdir(BASE_DIR)
root_dir = os.getcwd()

base_dir = BASE_DIR + PATH
pos_train_path = base_dir + '/train/pos'
neg_train_path = base_dir + '/train/neg'
pos_test_path = base_dir + '/test/pos'
neg_test_path = base_dir + '/test/neg'


pos_train_sentences = sentence_array_creator(pos_train_path, root_dir)
neg_train_sentences = sentence_array_creator(neg_train_path, root_dir)
pos_test_sentences = sentence_array_creator(pos_test_path, root_dir)
neg_test_sentences = sentence_array_creator(neg_test_path, root_dir)


print("{}, {}, {}, {}".format(len(pos_train_sentences), len(neg_train_sentences), len(pos_test_sentences), len(neg_test_sentences)))


def get_dic_index(word, dic):
    try:
        ret_val = dic.index(word)
    except ValueError: 
        ret_val = -1
    return ret_val



def sentence_encoder(sentences, vocab):
    bow = []
    t = RegexpTokenizer(r'\w+')
    prog = 0
    for sentence in sentences:
        if prog % 10 == 0:
            print(prog, sep=' ', end='-',flush=True)
        s = ''.join(sentence)
        tokens = t.tokenize(s)
        indexes = []
        for token in tokens:
            idx = get_dic_index(token, vocab)
            if idx != -1:
                indexes.append(idx)
        prog += 1
        bow.append(indexes)
    return bow

def sentence_tokenizer(sentences):
    bow = []
    t = RegexpTokenizer(r'\w+')
    for sentence in sentences:
        s = ''.join(sentence)
        tokens = t.tokenize(s)
        bow.append(tokens)
    return bow


def print_items(array):
    for i in array:
        print('{} \n'.format(i))
        
sentences = [
                ["elizabeth ashley is receiving phone calls from her nephew michael--he's crying, screaming and asking for help. the problem is michael died 15 years ago. <br /><br />this film scared me silly back in 1972 when it aired on abc. seeing it again, years later, it still works.<br /><br />the movie is a little slow and predictable, the deaths are very tame and there's a tacked-on happy ending, but this is a tv movie so you have to give it room. elizabeth ashley is excellent, ben gazzara is ok and it's fun to see michael douglas so young. and those telephone calls still scare the living daylights out of me. i actually had to turn a light on during one of them!<br /><br />a creepy little tv movie. worth seeing."],
                ["big fat liar is the best movie ever! it is funny, and cool. jason shepherd (frankie muniz) proves that he was not lying and goes to los angeles to get his paper back from marty wolf( paul giamatti). along with friend kaylee(amanda bynes), mess up his life since marty won't call jasons' dad and say he wrote the paper! yet it all turns out good and is a good movie to watch!"],
                ["out of any category, this is one demented and over the edge film, even in todays standards. filmed entirely in crap-o-rama, this film will blow your mind (and something else too!)<br /><br />the amount of hilarious bad taste and sleaze is astonishing. the dialog is breathtakingly fast and campy. you'll either love or hate this film, but give it go. i've seen it 4 times and absolutely love it. divine is in the quest for being the filthiest person alive, but so are her rivals too in this obscene and disgusting (but funny) and stylish little film. <br /><br />divine was phenomenal, and she will always be missed greatly. edith massey does the unforgettable performance as the egglady and don't forget the energetic mink stole!<br /><br />über crazy s**t! <br /><br />recommended also for you sick little puppies;<br /><br />female trouble <br /><br />desperate living <br /><br />polyester"]
            ]


print("now calculating bow_pos_train_sentences...")
bow_pos_train_sentences = sentence_encoder(pos_train_sentences, dic)
np.save(BASE_DIR + PATH + '/bow_pos_train_sentences_batch', bow_pos_train_sentences)

print("now calculating bow_neg_train_sentences...")
bow_neg_train_sentences = sentence_encoder(neg_train_sentences, dic)
np.save(BASE_DIR + PATH + '/bow_neg_train_sentences_batch', bow_neg_train_sentences)

print("now calculating bow_pos_test_sentences...")
bow_pos_test_sentences = sentence_encoder(pos_test_sentences, dic)
np.save(BASE_DIR + PATH + '/bow_pos_test_sentences_batch', bow_pos_test_sentences)

print("now calculating bow_neg_test_sentences...")
bow_neg_test_sentences = sentence_encoder(neg_test_sentences, dic)
np.save(BASE_DIR + PATH + '/bow_neg_test_sentences_batch', bow_neg_test_sentences)

print("DONE...")

print(len(bow_neg_test_sentences))
print(len(bow_neg_train_sentences))
print(len(bow_pos_test_sentences))
print(len(bow_pos_train_sentences))


print(len(np.load(BASE_DIR + PATH + '/bow_neg_test_sentences_batch.npy')))
print(len(np.load(BASE_DIR + PATH + '/bow_neg_train_sentences_batch.npy')))
print(len(np.load(BASE_DIR + PATH + '/bow_pos_test_sentences_batch.npy')))
print(len(np.load(BASE_DIR + PATH + '/bow_pos_train_sentences_batch.npy')))

