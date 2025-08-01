# Required installations (in terminal):
# pip install gradio
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2

import gradio as gr
import pickle
import docx
import PyPDF2
import re

# Load your models
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()

# File extract functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return file.read().decode("latin-1")

# Master function
def predict_resume_category(resume_text, uploaded_file):
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        elif name.endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_file)
        elif name.endswith(".txt"):
            resume_text = extract_text_from_txt(uploaded_file)
        else:
            return "❌ Unsupported file type."

    if not resume_text.strip():
        return "⚠️ Please provide resume text or upload a file."

    cleaned = cleanResume(resume_text)
    vec = tfidf.transform([cleaned]).toarray()
    pred_label = svc_model.predict(vec)
    category = le.inverse_transform(pred_label)[0]
    return f"✅ Predicted Category: **{category}**"

# Gradio Interface
interface = gr.Interface(
    fn=predict_resume_category,
    inputs=[
        gr.Textbox(label="Paste Resume Text (optional)", lines=10, placeholder="Paste resume content here..."),
        gr.File(label="Upload Resume (PDF, DOCX, TXT)", file_types=[".pdf", ".docx", ".txt"])
    ],
    outputs=gr.Markdown(label="Prediction"),
    title="📄 Resume Category Predictor",
    description="Upload or paste resume content to predict the job category using a pre-trained ML model."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()