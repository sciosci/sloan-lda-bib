import numpy as np
import gensim

def compute_coherence_values(corpus, dictionary, k, a = 'symmetric', b = None, passes = 3, workers = 2):

    lda_model = gensim.models.LdaMulticore(corpus=corpus_tfidf,
                                        id2word=dictionary,
                                        num_topics=k, 
                                        random_state=100,
                                        chunksize=100, alpha = a, eta = b, passes = passes, workers = passes)
    
    coherence_model_lda = CoherenceModel(model=lda_model, corpus = corpus, dictionary=id2word, coherence='u_mass')
    
    return(coherence_model_lda.get_coherence())