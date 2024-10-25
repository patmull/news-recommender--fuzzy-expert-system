import docx
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from collections import defaultdict
from keybert import KeyBERT

from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords

czech_stopwords = load_cz_stopwords()

tokenizer = RegexpTokenizer(r'\w+')
final=[]
final=pd.DataFrame(final)


def readtxt(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


doc=readtxt('Umele-inteligentni-doporucovaci-systemy-Muller-bez-obrazku.docx')


def preprocess(sentence):
    sentence = sentence.lower()
    rem_num = re.sub('[0-9]+', '', sentence)#REMOVING NUMBERS
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 5 if not w in czech_stopwords]
    return " ".join(filtered_words)


#FREQUENT WORDS
def unigrams(filename):
    ngram_freq = nltk.FreqDist() #WE INITIALIZED A FREQUENCY COUNTER
    for ngram in nltk.ngrams(tokenizer.tokenize(filename), 1):
        ngram_freq[ngram] += 1
    freq_words=pd.DataFrame(ngram_freq.most_common(100))
    freq_words[0]=freq_words[0].astype('str')
    final['uni-grams'] = freq_words[0].map(lambda x: re.sub(r'\W+', '', x))
    return(final['uni-grams'])


#BIGRAMS
def bigrams(filename):
    words=tokenizer.tokenize(filename)

    def ngrams(words, n=2, padding=False):
        grams =  words
        return (tuple(grams[i:i+n]) for i in range(0, len(grams) - (n - 1)))

    counts = defaultdict(int)
    for ng in ngrams(words, 2, False):
        counts[ng] += 1

    bigram=[]
    for c, ng in sorted(((c, ng) for ng, c in counts.items()), reverse=True):
        bigram.append(ng)
    bigram=pd.DataFrame(bigram)
    bigram[3]="  "
    bigram['bigram'] =bigram[0].map(str) +bigram[3] + bigram[1].map(str)
    final['bi-grams']=bigram['bigram'].head(100)
    return(final['bi-grams'])


def keybert(doc):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=czech_stopwords, top_n=100)
    return keywords


def keybert_bi(doc):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=czech_stopwords, top_n=100)
    return keywords


def get_keywords(doc):
    print("""UNIGRAMS:""")
    unigrams_keywords = unigrams(doc)
    print(unigrams_keywords)
    print("""BIGRAMS:""")
    bigrams_keywords = bigrams(doc)
    print(bigrams_keywords)
    print("""KEYWORDS:""")
    keybert_keywords = keybert(doc)
    keybert_keywords = [item[0] for item in keybert_keywords]
    # TODO: Convert to series
    print(keybert_keywords)
    keybert_bi_keywords = keybert_bi(doc)
    keybert_bi_keywords = [item[0] for item in keybert_bi_keywords]
    print(keybert_bi_keywords)

preprocessed_doc = preprocess(doc)
get_keywords(preprocessed_doc)

