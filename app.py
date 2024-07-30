# -*- coding: utf-8 -*-
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from transformers import pipeline
from eval_metrics import *

# Set up the Streamlit page configuration
st.set_page_config(page_title="Chat with My Data", layout="wide")

# Initialize Streamlit session state
def init_session_state():
    if 'user_turn' not in st.session_state:
        st.session_state['user_turn'] = False
    if 'pdf' not in st.session_state:
        st.session_state['pdf'] = None
    if 'embed_model' not in st.session_state:
        st.session_state['embed_model'] = None
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = None
    if 'eval_models' not in st.session_state:
        st.session_state['eval_models'] = {
            'app_metrics': AppMetrics(),
        #     'guards': IOGuards(),
        #     'textstat': TextStat(),
        #     'comparison': ComparisonMetrics(),
        #     'traditional_pipeline': TraditionalPipelines(),
        #     'llm_eval': None,  # Placeholder for LLMasEvaluator, will be set later
          }

init_session_state()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@AppMetrics.measure_execution_time
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if st.session_state['embed_model'] is None:
        st.session_state['embed_model'] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state['embed_model'],
                                    distance_strategy=DistanceStrategy.DOT_PRODUCT)
    vector_store.save_local("faiss_index")

def get_inference_pipeline():
    token = st.session_state['api_key']
    return pipeline("text2text-generation", model="google/flan-t5-base", token=token)

@AppMetrics.measure_execution_time
def llm_output(pipeline, docs, user_question):
    if not isinstance(docs, list):
        st.error("docs should be a list")
        return "Error generating response"

    if len(docs) > 0 and isinstance(docs[0], dict):
        context = "".join([doc.get('page_content', '') for doc in docs])
    elif len(docs) > 0 and isinstance(docs[0], str):
        context = "".join(docs)
    else:
        st.error("Unexpected format for docs")
        return "Error generating response"

    prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
    try:
        response = pipeline(prompt, max_length=150)
        return response[0]['generated_text'].strip()
    except Exception as e:
        st.error(f"Error during model generation: {e}")
        return "Error generating response"

@AppMetrics.measure_execution_time
def fetch_context(new_db, user_question, k=3, distance_strategy=DistanceStrategy.DOT_PRODUCT):
    return new_db.similarity_search_with_score(user_question, k=k, distance_strategy=distance_strategy)

@st.cache_data
def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", st.session_state['embed_model'],
                              allow_dangerous_deserialization=True,
                              distance_strategy=DistanceStrategy.DOT_PRODUCT)
    contexts_with_scores, exec_time = fetch_context(new_db, user_question, k=3,
                                                    distance_strategy=DistanceStrategy.DOT_PRODUCT)
    st.session_state["eval_models"]["app_metrics"].exec_times["chunk_fetch_time"] = exec_time

    # Extract text from document objects if necessary
    docs = [item.page_content for item, score in contexts_with_scores]

    generator = get_inference_pipeline()
    response, llm_exec_time = llm_output(generator, docs, user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["llm_resp_time"] = llm_exec_time

    st.write("Reply: ", response)

    ctx = ""
    for item, score in contexts_with_scores:
        if len(item.page_content.strip()):
            ctx += f"<li>Similarity Score: {round(float(score), 2)}<br>Context: {item.page_content}<br>&nbsp</li>"

    with st.expander("Click to see the context passed"):
        st.markdown(f"""<ol>{ctx}</ol>""", unsafe_allow_html=True)

    return contexts_with_scores, response

def evaluate_all(query, context_lis, response):
    if "app_metrics" not in st.session_state["eval_models"]:
        st.session_state["eval_models"]["app_metrics"] = AppMetrics()

    guards = st.session_state["eval_models"].get("guards", IOGuards())
    stat = st.session_state["eval_models"].get("textstat", TextStat())
    comp = st.session_state["eval_models"].get("comparison", ComparisonMetrics())
    traditional_pipeline = st.session_state["eval_models"].get("traditional_pipeline", TraditionalPipelines())
    appmet = st.session_state["eval_models"].get("app_metrics", AppMetrics())
    llm_eval = st.session_state["eval_models"].get("llm_eval", None)
    context = "\n\n".join(context_lis) if len(context_lis) else "no context"

    RESULT = {}

    # Ensure inputs are strings
    query = str(query)
    context = str(context)
    response = str(response)

    # Traditional Pipeline Evaluations
    if traditional_pipeline:
        RESULT["traditional_pipeline"] = {
            "topics": traditional_pipeline.topics(response),
            "ner": traditional_pipeline.NER(response),
            "pos": traditional_pipeline.POS(response),
            "summary": traditional_pipeline.Summarizer(response),
        }

    # Guard and Comparison metrics
    RESULT["guards"] = {
        "query_injection": guards.prompt_injection_classif(query),
        "context_injection": guards.prompt_injection_classif(context),
        "query_bias": guards.bias(query),
        "context_bias": guards.bias(context),
        "response_bias": guards.bias(response),
        "query_regex": guards.detect_pattern(query),
        "context_regex": guards.detect_pattern(context),
        "response_regex": guards.detect_pattern(response),
        "query_toxicity": guards.toxicity(query),
        "context_toxicity": guards.toxicity(context),
        "response_toxicity": guards.toxicity(response),
        "query_sentiment": guards.sentiment(query),
        "query_polarity": guards.polarity(query),
        "context_polarity": guards.polarity(context),
        "response_polarity": guards.polarity(response),
        "query_response_hallucination": comp.hallucinations(query, response),
        "context_response_hallucination": comp.hallucinations(context, response),
        "query_response_contradiction": comp.contradiction(query, response),
        "context_response_contradiction": comp.contradiction(context, response),
    }

    RESULT["guards"].update(guards.harmful_refusal_guards(query, context, response))

    tmp = {}
    for key, val in comp.ref_focussed_metrics(query, response).items():
        tmp[f"query_response_{key}"] = val

    for key, val in comp.ref_focussed_metrics(context, response).items():
        tmp[f"context_response_{key}"] = val

    RESULT["reference_based_metrics"] = tmp

    tmp = {}
    for key, val in comp.string_similarity(query, response).items():
        tmp[f"query_response_{key}"] = val

    for key, val in comp.string_similarity(context, response).items():
        tmp[f"context_response_{key}"] = val

    RESULT["string_similarities"] = tmp

    tmp = {}
    for key, val in stat.calculate_text_stat(response).items():
        tmp[f"result_{key}"] = val
    RESULT["response_text_stats"] = tmp

    RESULT["execution_times"] = st.session_state["eval_models"].get("app_metrics", AppMetrics()).exec_times

    # Add LLMasEvaluator results if available
    if llm_eval:
        llm_eval = LLMasEvaluator(llm=get_inference_pipeline())
        llm_results = llm_eval.evaluate(query, context_lis, response)
        RESULT["llm_eval"] = llm_results
    else:
        llm_results = {"error": "LLMasEvaluator not initialized."}

    return RESULT

def main():
    st.markdown("""## RAG Pipeline Example""")

    st.info("Note: This is a minimal demo focusing on ***EVALUATION*** so you can do simple Document QA using FLAN-T5 without any persistent memory hence no multi-turn chat is available. If the question is out of context from the document, this will not work so ask questions related to the document only. You can optimize the workflow by using Re-Rankers, Chunking Strategy, Better models etc but this app runs on CPU right now easily and is about, again, ***EVALUATION***", icon="ℹ️")

    st.error("WARNING: If you reload the page, everything (model, PDF, key) will have to be loaded again. That's how `streamlit` works", icon="🚨")

    with st.sidebar:
        st.title("Menu:")
        st.session_state['api_key'] = st.text_input("Enter your Hugging Face API Key:", type="password", key="api_key_input")
        if st.session_state['api_key']:
            st.session_state["pdf"] = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")

        if st.session_state["pdf"]:
            if st.session_state["embed_model"] is None:
                with st.spinner("Setting up `all-MiniLM-L6-v2` for the first time"):
                    st.session_state["embed_model"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            with st.spinner("Processing PDF files into chunks and creating `FAISS` Index..."):
                raw_text = get_pdf_text(st.session_state["pdf"])
                text_chunks, exec_time = get_text_chunks(raw_text)
                st.session_state["eval_models"]["app_metrics"].exec_times["chunk_creation_time"] = exec_time

                get_vector_store(text_chunks)
                st.success("Done")

    if not st.session_state['api_key']:
        st.warning("Please enter your Hugging Face API Key to proceed.")
    elif not st.session_state["pdf"]:
        st.warning("Upload a PDF file")
    else:
        st.markdown("""#### Ask a Question from the PDF file""")
        user_question = st.text_input("", key="user_question")

        if user_question and st.session_state['api_key']:  # Ensure API key and user question are provided
            contexts_with_scores, response = user_input(user_question)

            st.warning("There are 5 major types of metrics computed below having multiple sub-metrics", icon="🤖")
            metric_calc = st.button("Load Models & Compute Evaluation Metrics")
            if metric_calc:
                if st.session_state["eval_models"].get("llm_eval") is None:
                    # Ensure you pass the required `llm` argument
                    st.session_state["eval_models"]["llm_eval"] = LLMasEvaluator(llm=get_inference_pipeline())

                with st.spinner("Calculating all the metrics. Please wait ...."):
                    eval_result = evaluate_all(user_question, [item.page_content for item, score in contexts_with_scores], response)
                    st.balloons()

                st.json(eval_result)

if __name__ == "__main__":
    main()