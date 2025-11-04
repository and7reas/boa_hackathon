class Config:

    '''
    contains the text cleaning regexes and the corresponding substring replacements
    '''
    REGEX_REPLACEMENTS_DICT = {
            " +" : " ", #replacing multiple spaces with single space
          
    }

    '''
    the id of the sentences transformer pre-trained model to use
    '''
    SENTENCE_TRANSFORMER_MODEL_ID = "all-MiniLM-L6-v2"


    '''
    the tfidf configuration applied to each of the clean versions of the documentation sentences
    '''
    TFIDF_VEC_CONF = {
        "min_df" : 1,
        "ngram_range" : (1,2)
    }

    '''
    the number of most similar documentations returned
    '''
    NUM_OF_MOST_SIM_DOCS_RETURNED = 5

    '''
    the path to the local location where the database documentations embeddings are saved
    '''
    DATABASE_EMBDEDDINGS_PATH = ""

    

    

