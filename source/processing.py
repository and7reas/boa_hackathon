from sentence_transformers import SentenceTransformer, util
import re
import nltk
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os

from config import Config

nltk.download('punkt')



class Processing:


    def clean_doc_text(text,
                       regex_replacements_dict = None):
        '''
        performs basic cleaning to input document text

        Args:
                                          text (str) : the raw documentation text
            regex_replacements_dict (Dict[str, str]) : dictionary of the form (regex to identify text to be replaced --> new substring acting as replacement)

        Returns:
                                                 str : the clean version of the text
        '''
        regex_replacements_dict = regex_replacements_dict if regex_replacements_dict is not None else Config.REGEX_REPLACEMENTS_DICT
        
        text = text.lower()
        for regex, repl in regex_replacements_dict.items:
            text = re.sub(regex, repl)
        return text

    
    def split_into_sentences(text):
        '''
        splits the cleaned version of the documentation into sentences

        Args:
            text (str) : the clean version of the documentation

        Returns:
             List[str] : list of the sentences included inside the documentation
        '''
        sentences = nltk.sent_tokenize(text)
        clean_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]
        return clean_sentences



    def create_sentence_embeddings(sentences,
                                   st_model_id = None):
        '''
        Creates embeddings for the given sentences

        Args:
            sentences (List[str]) : the document sentences for which word embeddings will be created
                st_model_id (str) : the sentence transformers model ID to be used
 
        Returns:
                         np.array : array containing the embeddings for each input sentence
        '''

        st_model_id = st_model_id if st_model_id is not None else Config.SENTENCE_TRANSFORMER_MODEL_ID

        model = SentenceTransformer(st_model_id)

        emb_vectors = model.encode(sentences,
                                   normalize_embeddings = True,
                                   convert_to_numpy = True)

        return emb_vectors

    
    def fit_tfidf(sentences,
                  min_df = None,
                  ngram_range = None):
        '''
        applies tfidf over the superset of sentences comprising the document

        Args:
            sentences (List[str]) : list containing all the sentences contained in the documentation
                     min_df (int) : the minimum document frequency a term must have to be considered
    ngram_range (Tuple[int, int]) : tuple of the form (lowest degree of n-gram being considered, maximum degree of ngram considered)
        Returns:
                  TfIdfVectorizer : the fitted tfidf vectorizer's object                    
        '''
        min_df = min_df if min_df is not None else Config.TFIDF_VEC_CONF["min_df"]
        ngram_range = ngram_range if ngram_range is not None else Config.TFIDF_VEC_CONF["ngram_range"]

        tfidt_model = TfidfVectorizer(min_df = min_df,
                                      ngram_range = ngram_range)

        fitted_tfidf_model = tfidf_model.fit(sentences)
        return fitted_tfidf_model

    
    def get_sentences_tfidf_sum(sentence):
        '''
        returns the sum of the tfidf values associated with the input sentences with the aim to using them as sentence weights

        Args: 
                    sentence (str) : the clean version of the sentence
     tfidf_model (TfidfVectorizer) : the fitted tfidf model

        Returns:
                             float : the sum of the tfidf values associated with the particular sentence
        '''
        sentence_tfidf_vec = tfidf_model.transform([sentence])
        tfidf_values_sum = float(sentence_tfidf_vec.sum())
        return tfidf_values_sum

    
    def generate_final_documentation_embedding(sentences,
                                               sentences_embeddings,
                                               fitted_tfidf_model):
        '''
        generates the final embedding corresponding to a particular documentation

        Args:
                               sentences (List[str]) : list containing the clean text version of the sentences of the documentation
                     sentences_embeddings (np.array) : array containing the sentences transformer embeddings of each sentence
                fitted_tfidf_model (TfIdfVectorizer) : the fitted tfidf model on the documentation sentences

        Returns:
                                            np.array : the documentation's tfidf-weighted embedding
        '''
        sentences_weights = [get_sentences_tfidf_sum(sentence) for sentence in sentences]
        documentation_embedding = (sentences_embeddings * sentences_weights[:, None]).sum(axis = 0) / weights.sum()
        return documentation_embedding

    
    def get_top_similar_documentations(chosen_documentation_embedding,
                                       database_documentation_embeddings,
                                       database_documentation_ids,
                                       top_n = None):
        '''
        gets the top <top_n> most similar documentation embeddings to the documentation of interest

        Args:
               chosen_documentation_embedding (np.array) : vector embedding of the documentation of interest
            database_documentation_embeddings (np.array) : array containing the embeddings of the documentations that the given documentation will compare against
                  database_documentation_ids (List[str]) : list containing the ids of the database documentations
                                             top_n (int) : the number of most similar documentations to be returned

        Returns:
                                   List[str, float] : list of tuples of the form (documentation IDs, cosine similarity scores) in descending order of similarity scores
        '''
        top_n = top_n if top_n is not None else Config.NUM_OF_MOST_SIM_DOCS_RETURNED

        cosine_similarities = util.cos_sim(chosen_documentation_embedding,
                                           database_documentation_embeddings)

        docids_similarities = list(zip(database_documentation_ids,
                                       cosine_similarities))
        sorted_docids_similarities = sorted(docids_similarities,
                                           key = lambda x: (x[1], x[0]),
                                           reverse = True)
        final_sorted_docids_similarities = sorted_docids_similarities[:top_n] if len(sorted_docids_similarities) >= top_n else sorted_docids_similarities

        return final_sorted_docids_similarities

    
    def compare_documentation_with_existing(doc_text,
                                            vectors_path = None):
        '''
        compares a new documentation with the existing ones in the database

        Args:
                     doc_text (str) : the raw string documentation added by the user
                 vectors_path (str) : the path to the local location storing the documentation embeddings

        Returns:
  Tuple[np.array, List[str, float]] : tuple of the form (new documentation's word embedding, list of tuples of the form (documentation IDs, cosine similarity scores) in descending order of similarity scores)
        '''
        vectors_path = vectors_path if vectors_path is not None else Config.DATABASE_EMBDEDDINGS_PATH
        
        # basic cleaning of the text
        clean_doc_text = clean_doc_text(text = doc_text)

        # splitting the documentation text to sentences
        doc_sentences = split_into_sentences(text = clean_doc_text)

        # creating the documentation sentences embeddings
        doc_sentences_embeddings = create_sentence_embeddings(sentences = doc_sentences)

        # fitting tfidf model using the documentation's sentences
        fitted_tfidf_model = fit_tfidf(sentences = doc_sentences)

        # creating the documentation's embedding
        doc_embedding = generate_final_documentation_embedding(sentences = doc_sentences,
                                                               sentences_embeddings = doc_sentences_embeddings,
                                                               fitted_tfidf_model = fitted_tfidf_model)

        
        # getting the most similar documentations from the database 

        database_documentation_ids, database_documentation_embeddings = load_database_documentation_embeddings()

        if len(database_documentation_ids) > 0 :
        
            doc_sim_list = get_top_similar_documentations(chosen_documentation_embedding = doc_embedding,
                                                          database_documentation_embeddings = database_documentation_embeddings,
                                                          database_documentation_ids = database_documentation_ids)
    
            return (doc_embedding, 
                    doc_sim_list)
        return (doc_embedding, [])


    def save_documentation_embedding(document_id,
                                     doc_embedding,
                                     vectors_path = None):
        '''
        saves the documentation's embedding to the existing documentation's database

        Args:
              documentation_id (str) : the identifier of the documentation to be saved
            doc_embedding (np.array) : the documentation's embedding to be saved to the embeddings database
                  vectors_path (str) : the path to the local location storing the documentation embeddings

        Returns:
                                None 
        '''
        vectors_path = vectors_path if vectors_path is not None else Config.DATABASE_EMBDEDDINGS_PATH

        try:
            with open("{}/{}.pkl".format(vectors_path, document_id), "wb") as f:
                pickle.dump(doc_embedding, f)
            print("the file was saved successfully...")

        except Exception as e:
            print("Error when the documentation's embedding...")



    def load_database_documentation_embeddings(vectors_path = None):
        '''
        loads all the database embeddings

        Args:
            vectors_path (str) : the path to the local location storing the documentation embeddings

        Returns:
                      np.array : array containing the embeddings of all the database documentations
        '''
        def load_embedding(vector_file_name):
            '''
            loads a particular documentation embedding

            Args:
                vector_file_name (str) : the name of the database documentation of interest

            Returns:
             Tuple[str, np.array] : tuple of the form (the documentation_id, array containing the documentation's embedding)
            '''
            with open("{}/{}".format(vectors_path, vector_path), "rb") as f:
                doc_embedding = pickle.dump(f)
                doc_id = vector_path.split(".pkl")[0]
                return (doc_id, doc_embedding)
            return None
        vectors_path = vectors_path if vectors_path is not None else Config.DATABASE_EMBDEDDINGS_PATH

        vector_file_names = os.listdir(vectors_path)
 
        db_doc_embeddings = [load_embedding(vector_file) for vector_file_name in vector_file_names]

        doc_ids, doc_embeddings = zip(*db_doc_embeddings))

        return (list(doc_ids), list(doc_embeddings))

        

        

        

        
        


    

        
        
        
        