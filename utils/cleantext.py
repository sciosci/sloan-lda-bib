import pandas as pd
import numpy as np
import re
from gensim.utils import simple_preprocess
import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def break_copyright(text):
    if 'Copyright' in text:
        text = text.split('Copyright')[0]
    if 'copyright' in text:
        text = text.split('copyright')[0]
    if 'COPYRIGHT' in text:
        text = text.split('COPYRIGHT')[0]
    if '©' in text:
        text = text.split('©')[0]
    
    return(text)


def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def clean_abstract(text):
    text = remove_html_tags(break_copyright(text))
    text = re.sub('[,\.!?()-/%]', ' ', text)
    text = re.sub('{\a\b\t\r\v\f\n\d}', ' ', text)
    return(' '.join(text.lower().split()))
    
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))
        
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([str(token.lemma_) for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_ngrams(tokens, ngram_range = (1, 1), stop_words=None):
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        if min_n == 1:
            # no need to do any slicing for unigrams
            # just iterate through the original tokens
            tokens = list(original_tokens)
            min_n += 1
        else:
            tokens = []

        n_original_tokens = len(original_tokens)

        # bind method outside of loop to reduce overhead
        tokens_append = tokens.append
        space_join = " ".join

        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i: i + n]))

    return(tokens) 