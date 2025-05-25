from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sqlite3
import wikipedia
from googletrans import Translator

app = Flask(__name__)

# GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Traducción
translator = Translator()

# Memoria con SQLite
conn = sqlite3.connect("memory.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS memory (prompt TEXT, response TEXT)")
conn.commit()

# Generador de respuesta multilingüe
def generate_response(prompt):
    # Detectar idioma
    lang = translator.detect(prompt).lang
    translated_prompt = translator.translate(prompt, dest='en').text

    # Generar con GPT-2
    inputs = tokenizer.encode(translated_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7)
    english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Traducir de regreso
    final_response = translator.translate(english_response, dest=lang).text
    return final_response

# Wikipedia
def wiki_answer(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None

# Rutas
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    if any(q in user_input.lower() for q in ["qué", "quién", "dónde", "cuándo", "what", "who", "when", "where"]):
        wiki = wiki_answer(user_input)
        response = wiki if wiki else generate_response(user_input)
    else:
        response = generate_response(user_input)

    c.execute("INSERT INTO memory (prompt, response) VALUES (?, ?)", (user_input, response))
    conn.commit()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
