from math import nan
from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec
import os
import re
from nltk.corpus import wordnet,stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from django import forms
import pandas as pd
from joblib import load

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

wnl = WordNetLemmatizer()
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def stem(sentence):
    lemmas_sent = []
    tagged_sent  = pos_tag(sentence)
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos = wordnet_pos))
    return lemmas_sent

def display_pca_scatterplot(model, words, name, search_name):
    word_vectors = np.array([model.wv[w] for w in words if w != search_name])
    dot = np.array([model.wv[search_name]])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    twodim_dot = PCA().fit_transform(dot)[:2]

    plt.figure(figsize=(7, 4))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors = 'k', c = 'b', marker="x")
    plt.scatter(twodim_dot[0], twodim_dot[0], edgecolors = 'k', c = 'r', marker="x")
    plt.text(twodim_dot[0] + 0.05, twodim_dot[0] + 0.05, search_name)
    for word, (x, y) in zip(words[1:], twodim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.savefig(os.path.join(BASE_DIR + '/server/static', name))
    plt.close()

def text_preprocess(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and not(word.isdigit()))
    return stem([text])

def most_similar(wv_model, words, topn):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(wv_model.wv.most_similar(word, topn = topn), columns = [word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis = 1)
            sim_word = similar_df[word].to_list()
            sim_cos = similar_df['cos'].to_list()
            sim = [[i, round(j, 4)] for i, j in zip(sim_word, sim_cos)]
        except:
            print(word, "not found in Word2Vec model")
    return sim

def get_words_list():
    ALL_WORDS = []
    df = pd.read_csv('words.csv')
    for data in df.iloc:
        temp = []
        for word in data.to_list():
            if not(pd.isnull(word)):
                temp.append(word)
        ALL_WORDS.append(temp)
    return ALL_WORDS

def get_dict_list():
    df = pd.read_csv('dict_words.csv')
    sorted_word = df['0'].to_list()
    sorted_num = df['1'].to_list()
    return sorted_word, sorted_num

def MED(sent_01, sent_02):
    n = len(sent_01)
    m = len(sent_02)

    matrix = [[i + j for j in range(m + 1)] for i in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sent_01[i - 1] == sent_02[j - 1]: d = 0
            else: d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    distance_score = matrix[n][m]
   
    return distance_score

class FileUploadForm(forms.Form):
    file = forms.FileField(label = "File upload", widget=forms.ClearableFileInput(attrs={'multiple': True}))

def home(request):
    dict_word = {}
    global sorted_word
    error_msg = ''
    model = Word2Vec.load('word2vec.model')
    model_sg = Word2Vec.load('word2vec_sg.model')
    sorted_word, sorted_num = get_dict_list()
    if 'search' in request.POST:
        search = request.POST['search']
        search = text_preprocess(search)
        if search not in sorted_word:
            for word in sorted_word:
                try:
                    dict_word[word] = MED(word, search)
                except:
                    continue
            revise_word = sorted(dict_word.items(), key = lambda d: d[1])[0:5]
            revise_word = [i for i, j in revise_word]
            error_msg = 'No results for "'+ search + '". Do you want to search ' 
            return render(request, 'home.html', locals())
    elif 'revise' in request.POST:
        search = request.POST['revise']
    else:
        search = 'covid19'
    name = search + '.png'
    name2 = search + '_sg.png'
    if (not os.path.isfile(os.path.join(BASE_DIR + '/server/static', name))) or (not os.path.isfile(os.path.join(BASE_DIR + '/server/static', name2))):
        display_pca_scatterplot(model, sorted_word[:15], name, search)
        display_pca_scatterplot(model_sg, sorted_word[:15], name2, search)
    sim_word = most_similar(model, [search], 15)
    sim_word_sg = most_similar(model_sg, [search], 15)
    return render(request, 'home.html', locals())

def get_MLP(articles, model):
    MLP = []
    all_words = []
    for abstract in articles:
        all_words.append(text_preprocess(abstract))
    for i in range(len(all_words)):
        num = 0
        temp = [0] * 50
        for word in all_words[i][0].split():
            if word in model.wv and (word != 'covid19' and word != 'heartdisease'):
                num += 1
                for j in range(50):
                    temp[j] += model.wv[word][j]
        for j in range(50):
            try:
                temp[j] = round(temp[j] / num, 4)
            except:
                temp[j] = 0
        MLP.append(temp)
    return MLP

def down_stream(request):
    forms = FileUploadForm()
    if request.method == 'POST':
        forms = FileUploadForm(request.POST,request.FILES)
        if forms.is_valid():
            files = request.FILES.getlist('file')
            for f in files:
                file_name = str(f)
                model_sg = Word2Vec.load('word2vec_sg.model')
                data = pd.read_csv(f).dropna()
                titles = data['title'].to_list()
                abstracts = data['abstract'].to_list()
                MLP = get_MLP(abstracts, model_sg)
                clf = load('model_MLP.joblib') 
                predicteds = clf.predict(np.array(MLP))
                result = []
                for predicted in predicteds:
                    if predicted == 0:
                        result.append('covid-19')
                    elif predicted == 1:
                        result.append('heart disease')
                data_list = []
                for i in range(10):
                    try:
                        data_list.append([titles[i], abstracts[i], result[i]])
                    except:
                        i += 1
        else: 
            msg.append('The submitted file is invalid ( empty... ).')
    return render(request, 'down_stream.html', locals())
