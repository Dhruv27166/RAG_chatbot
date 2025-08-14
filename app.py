import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context". Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def compute_bleu(reference, candidate):
    return corpus_bleu([candidate], [[reference]]).score

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    if docs:
        top_document = docs[0]
        context = top_document.page_content
    else:
        context = "No context found"

    generated_response = response["output_text"]

    data = {
        "query": user_question,
        "context": context,
        "response": generated_response
    }

    save_json(data)

    print("Context:", context)
    print("Generated Response:", generated_response)

    reference = context
    candidate = generated_response
    bleu_score = compute_bleu(reference, candidate)
    rouge_scores = compute_rouge(reference, candidate)
    P, R, F1 = score([candidate], [reference], lang='en', verbose=True)

    print("BLEU Score:", bleu_score)
    print("ROUGE Scores:", rouge_scores)
    print("BERT Scores - Precision:", P, "Recall:", R, "F1:", F1)

    st.write("Generated Response: ", generated_response)
    st.write("BLEU Score: ", bleu_score)
    st.write("ROUGE-1 Score: ", rouge_scores['rouge1'].fmeasure)
    st.write(f"Precision: {P.mean().item()}")
    st.write(f"Recall: {R.mean().item()}")
    st.write(f"F1 Score: {F1.mean().item()}")

def save_json(data, filename="output_data.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.header("Athina AI Assignment (PDF QA CHATBOT)")

    user_question = st.text_input("Ask a Question from the PDF File")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
