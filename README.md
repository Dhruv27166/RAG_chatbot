# PDF QA Chatbot

A **Streamlit-based PDF Question-Answering chatbot** that allows you to upload PDF documents, process their content into embeddings, and then ask natural language questions. The chatbot retrieves relevant context and generates detailed answers, while also evaluating the responses using **BLEU**, **ROUGE**, and **BERTScore** metrics.

---

## Features
- **Multiple PDF Upload** – Upload one or more PDF documents at once.
- **Text Extraction** – Extracts text from all PDF pages.
- **Text Chunking** – Splits large documents into smaller chunks for efficient retrieval.
- **Vector Store** – Uses **FAISS** for fast similarity search over embedded text.
- **Conversational QA** – Retrieves relevant context and answers questions using **Google Gemini**.
- **Evaluation Metrics** – Automatically computes:
  - BLEU Score
  - ROUGE Score
  - BERTScore (Precision, Recall, F1)
- **JSON Logging** – Saves query, context, and generated response to a JSON file.

---

## Project Structure
```
.
├── app.py                # Main Streamlit application
├── output_data.json      # Saved Q&A and evaluation results
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Variables
Create a `.env` file in the root directory and add your **Google API key**:

```
GOOGLE_API_KEY=your_google_api_key_here
```

You can get a Google API key from the [Google AI Studio](https://aistudio.google.com/).

---

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

1. Upload one or more PDF files from the **sidebar**.
2. Click **"Submit & Process"** to extract and store embeddings.
3. Enter your question in the text input box.
4. View:
   - Generated answer
   - BLEU, ROUGE, and BERTScore metrics
   - Logged output in `output_data.json`

---

## Example Output

**Console log:**
```
Context: [Top relevant document chunk...]
Generated Response: [Model's answer...]
BLEU Score: 56.43
ROUGE Scores: {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
BERT Scores - Precision: tensor([0.93]) Recall: tensor([0.91]) F1: tensor([0.92])
```

**Streamlit UI:**
- Generated response displayed
- BLEU score displayed
- ROUGE-1 score displayed
- Precision, Recall, F1 displayed

---

## Dependencies
- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain_google_genai`
- `google-generativeai`
- `faiss-cpu`
- `sacrebleu`
- `rouge-score`
- `bert-score`
- `python-dotenv`
---
## Acknowledgements
- [LangChain](https://www.langchain.com/) for text processing and chain management.
- [Google Generative AI](https://aistudio.google.com/) for embeddings and chat model.
- [FAISS](https://faiss.ai/) for vector similarity search.
- [Streamlit](https://streamlit.io/) for the interactive UI.
