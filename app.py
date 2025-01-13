from flask import Flask, request, render_template, jsonify
import torch
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, MBartForConditionalGeneration, MBart50TokenizerFast
from LDict import find_legal_terms, legal_terms_lower
import nltk
import re,os, logging

# Set environment variables for writable directories
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
nltk.data.path.append("/tmp/nltk_data")

logging.basicConfig(level=logging.ERROR)

# Download necessary NLTK data
nltk.download('punkt', download_dir="/tmp/nltk_data")
nltk.download('punkt_tab', download_dir="/tmp/nltk_data")


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"

#Method 1 model
pegasus_ckpt = "google/pegasus-cnn_dailymail"
tokenizer_pegasus = AutoTokenizer.from_pretrained(pegasus_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(pegasus_ckpt).to(device)

# Method 2 model
port_tokenizer= AutoTokenizer.from_pretrained("stjiris/t5-portuguese-legal-summarization")
model_port = AutoModelForSeq2SeqLM.from_pretrained("stjiris/t5-portuguese-legal-summarization").to(device)

#paraphrase
t5_ckpt = "t5-base"
tokenizer_t5 = T5Tokenizer.from_pretrained(t5_ckpt)
model_t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt).to(device)

#Translation Model
mbart_ckpt = "facebook/mbart-large-50-one-to-many-mmt"
tokenizer_mbart = MBart50TokenizerFast.from_pretrained(mbart_ckpt,src_lang="en_XX")
model_mbart = MBartForConditionalGeneration.from_pretrained(mbart_ckpt).to(device)


def simplify_text(input_text):
    matches = find_legal_terms(input_text)
    tokens = word_tokenize(input_text)
    simplified_tokens = [f"{token} ({legal_terms_lower[token.lower()]})" if token.lower() in matches else token for token in tokens]
    return ' '.join(simplified_tokens)

def remove_parentheses(text):
    p1 = re.sub(r"[()]", "", text)
    p2 = re.sub(r"\s+", " ", p1).strip()
    p3 = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", p2, flags=re.IGNORECASE)
    return p3

def summarize_text(text, method):
    if method == "method2":
        #Sumarry Model2
        inputs_legal = port_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids_legal = model_port.generate(inputs_legal["input_ids"], max_length=250, num_beams=4, early_stopping=True)
        Summarized_method2 = port_tokenizer.decode(summary_ids_legal[0], skip_special_tokens=True)
        cleaned_summary2 = remove_parentheses(Summarized_method2)
        #Paraphrase
        p_inputs = tokenizer_t5.encode(cleaned_summary2, return_tensors="pt", max_length=512, truncation=True)
        p_summary_ids = model_t5.generate(p_inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        method2 = tokenizer_t5.decode(p_summary_ids[0], skip_special_tokens=True)
        return method2

    elif method == "method1":
        summarization_pipeline = pipeline('summarization', model=model_pegasus, tokenizer=tokenizer_pegasus, device=0 if device == "cuda" else -1)
        method1 = summarization_pipeline(text, max_length=100, min_length=30, truncation=True)[0]['summary_text']
        cleaned_summary1 = remove_parentheses(method1)
        return cleaned_summary1


def translate_to_hindi(text):
    inputs = tokenizer_mbart([text], return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model_mbart.generate(**inputs, forced_bos_token_id=tokenizer_mbart.lang_code_to_id["hi_IN"])
    
    # Select the first sequence from the generated tokens
    translation = tokenizer_mbart.decode(translated_tokens[0], skip_special_tokens=True)  
    return translation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_text = request.form['input_text']
            method = request.form['method']
            
            simplified_text = simplify_text(input_text)
            summarized_text = summarize_text(simplified_text, method)

            return jsonify({
                "summarized_text": summarized_text, })
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data['text']
        translated_text = translate_to_hindi(text)

        return jsonify({
            "translated_text": translated_text})
    except Exception as e:
        logging.error(f"Error occurred during translation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5003)