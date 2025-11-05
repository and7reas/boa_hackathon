import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from openai import OpenAI
from pydantic import BaseModel, Field

from config import Config
from doc_info import DocInfo


nltk.download('punkt')
nltk.download('punkt_tab')



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

        for regex, repl in regex_replacements_dict.items():
            text = re.sub(regex, repl, text)
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

        model = SentenceTransformer(st_model_id, device = "cpu")

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

        tfidf_model = TfidfVectorizer(min_df = min_df,
                                      ngram_range = ngram_range)

        fitted_tfidf_model = tfidf_model.fit(sentences)
        return fitted_tfidf_model

    
    def get_sentences_tfidf_sum(sentence,
                                tfidf_model):
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
        sentences_weights = [Processing.get_sentences_tfidf_sum(sentence, fitted_tfidf_model) for sentence in sentences]
        sentences_weights = np.array(sentences_weights).reshape(-1, 1)

        documentation_embedding = (sentences_embeddings * sentences_weights).sum(axis = 0) / sentences_weights.sum()
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
                                           database_documentation_embeddings).tolist()[0]

        docids_similarities = list(zip(database_documentation_ids,
                                       cosine_similarities))
        
        sorted_docids_similarities = sorted(docids_similarities,
                                           key = lambda x: (x[1], x[0]),
                                           reverse = True)

        final_sorted_docids_similarities = sorted_docids_similarities[:top_n] if len(sorted_docids_similarities) >= top_n else sorted_docids_similarities

        return final_sorted_docids_similarities

    def restructure_document_and_receive_feedback(raw_doc_text,
                                                  system_prompt = None,
                                                  content_prompt_format = None,
                                                  llm_model_id = None):
        '''
        restructures the document based on the prompt guidelines and receives feedback

        Args:
            raw_doc_text (str) : the raw documentation text submitted by the user
           system_prompt (str) : the system prompt giving high level information to the llm
   content_prompt_format (str) : the format of the prompt used for giving the tasks details to the llm
            llm_model_id (str) : the id of the openai model used

        Returns:
               Tuple[str, str] : tuple of the form (the structured form of the input document, the feedback in terms of the document's contents)
            
        '''
        system_prompt = system_prompt if system_prompt is not None else Config.SYSTEM_PROMPT
        content_prompt_format = content_prompt_format if content_prompt_format is not None else Config.CONTENT_PROMPT_FORMAT
        llm_model_id = llm_model_id if llm_model_id is not None else Config.OPENAI_MODEL_USED

        client = OpenAI()

        st.text("querying the llm for generating the better structure and the required feedback...")

        response = client.chat.completions.create(
    model = llm_model_id,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": content_prompt_format.format(raw_doc_text),
        },
    ],
    response_format={"type": "json_object"},
)

        output_json = response.choices[0].message.content
        '''

        output_json = """
       "standardized_document": {
    "What is this API?": "The InvTrack API provides access to inventory management features, including retrieving product information and current stock levels. The API returns product details and stock data in JSON format.",
    "Why is it useful?": "The API enables automation of inventory tracking, ensures up-to-date product information, and reduces manual lookups. It facilitates efficient integration of inventory data into other business systems, improving operational accuracy and decision-making.",
    "Who is it for?": "This API is intended for developers building inventory management tools, system integrators, or business teams looking to automate product and stock data retrieval within their platforms.",
    "How does it work?": "To use the API, first register for an InvTrack account. Upon registration, obtain an API key. Authenticate requests by including the API key in the query string of each call. Use provided endpoints to fetch or update inventory information. Responses are returned in JSON, and there is a rate limit of 2000 requests per day.",
    "API Endpoints": [
      {
        "Name": "Products",
        "URL": "/products",
        "Method": "GET",
        "Function": "Retrieve product catalog"
      },
      {
        "Name": "Stock Levels",
        "URL": "/stock-levels",
        "Method": "GET",
        "Function": "Check current stock levels"
      }
    ],
    "Example API Call": "GET https://api.invtrack.com/products?key=ABC123\nHeaders:\n  Accept: application/json\nResponse:\n{\n  \"products\": [\n    {\"id\": 1, \"name\": \"Laptop\", \"stock\": 50},\n    {\"id\": 2, \"name\": \"Mouse\", \"stock\": 200}\n  ]\n}"
  """
        '''

        #st.text("model output: {}".format(output_json))

        st.text("extracting and validating model output...")

        doc_info = DocInfo.model_validate_json(output_json)

        doc_structured_form = doc_info.standardized_document
        doc_feedback = doc_info.missing_information_feedback

        return (str(doc_structured_form), 
                str(doc_feedback))

        

        
    
    
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

        st.text("generating the documentation's embedding...")
        # basic cleaning of the text
        doc_text_clean = Processing.clean_doc_text(text = doc_text)

        # splitting the documentation text to sentences
        doc_sentences = Processing.split_into_sentences(text = doc_text_clean)

        # creating the documentation sentences embeddings
        doc_sentences_embeddings = Processing.create_sentence_embeddings(sentences = doc_sentences)

        # fitting tfidf model using the documentation's sentences
        fitted_tfidf_model = Processing.fit_tfidf(sentences = doc_sentences)

        
        # creating the documentation's embedding
        doc_embedding = Processing.generate_final_documentation_embedding(sentences = doc_sentences,
                                                               sentences_embeddings = doc_sentences_embeddings,
                                                               fitted_tfidf_model = fitted_tfidf_model)


        # getting the most similar documentations from the database 
        database_documentation_ids, database_documentation_embeddings = Processing.load_database_documentation_embeddings()

        if len(database_documentation_ids) > 0 :
            
            st.text("extracting top most similar database documentations...")
            doc_sim_list = Processing.get_top_similar_documentations(chosen_documentation_embedding = doc_embedding,
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
            st.text("saving documentation embedding...")
            with open("{}/{}.pkl".format(vectors_path, document_id), "wb") as f:
                pickle.dump(doc_embedding, f)
            st.text("the file was saved successfully...")

        except Exception as e:
            st.text("Error when saving the documentation's embedding...")



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
            with open("{}/{}".format(vectors_path, vector_file_name), "rb") as f:
                doc_embedding = pickle.load(f)
                doc_id = vector_file_name.split(".pkl")[0]
                return (doc_id, doc_embedding)
            return None
        vectors_path = vectors_path if vectors_path is not None else Config.DATABASE_EMBDEDDINGS_PATH

        vector_file_names = os.listdir(vectors_path)
 
        db_doc_embeddings = [load_embedding(vector_file_name) for vector_file_name in vector_file_names if not vector_file_name.startswith(".")]

        if len(db_doc_embeddings) > 0:
            doc_ids, doc_embeddings = zip(*db_doc_embeddings)
    
            return (list(doc_ids), list(doc_embeddings))
        return ([], [])

        

        

        

        
        


    

        
        
        
        