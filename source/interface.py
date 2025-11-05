import streamlit as st
st.text("importing packages")
from processing import Processing
from plotting import Plotting

st.title("standardAIzer")

doc_title = st.text_input(label = "The title of the documentation",
                          value = st.session_state.get("documentation_title", ""))

doc_text = st.text_area(label = "Input Standardization",
                        value = st.session_state.get("input_documentation", ""),
                        height = 250)

feedback_text = st.text_area(label = "standardAIzer feedback",
                             value = st.session_state.get("genai_feedback", ""))

if "similarity_plot" in st.session_state:
    st.pyplot(st.session_state["similarity_plot"])

submit_button = st.button(label = "Submit",
                          disabled = doc_text == "" or doc_title == "")


save_button = st.button(label = "Save",
                        disabled = feedback_text == "" or doc_title == "")

if submit_button:

    # using the llm to restructure the document and obtain feedback

    structured_doc_text, feedback = Processing.restructure_document_and_receive_feedback(raw_doc_text = doc_text)
    st.session_state["input_documentation"] = str(structured_doc_text)
    st.session_state["genai_feedback"] = feedback

    # comparing the documentation with existing documentations and extract the most similar ones

    doc_embedding, doc_sim_list = Processing.compare_documentation_with_existing(doc_text = structured_doc_text)

    # visualizing the similarity with existing documentations
    if len(doc_sim_list) > 0:
        sim_fig = Plotting.plot_sim_scores(doc_sim_scores = doc_sim_list)

        st.session_state["similarity_plot"] = sim_fig
    st.session_state["documentation_embedding"] = doc_embedding

    st.rerun()

if save_button:
    # saving the re-structured documentation embedding to the 'database'
    Processing.save_documentation_embedding(document_id = session_state["documentation_title"],
                                            doc_embedding = session_state["documentation_embedding"])

    st.session_state["documentation_title"] = ""
    st.session_state["input_documentation"] = ""
    st.session_state["genai_feedback"] = ""

    st.rerun()

    

    

    







