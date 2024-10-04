from openai import OpenAI
import os
from streamlit import secrets

OPENAI_API_KEY = secrets["OPENAI_API_KEY"]

def generate_prompt():
    
    prompt = """
        Tu es un enseignant expérimenté et bienveillant, spécialisé dans la correction des textes d'étudiants.
        Ta tâche consiste à fournir trois éléments essentiels dans un format Markdown, en gardant la réponse courte, bienveillante, et concise, uniquement comme ceci :

        1. **Vocabulaire :** Donne une analyse du vocabulaire utilisé, en mentionnant les points forts et les suggestions d'amélioration.

        2. **Grammaire :** Corrige toutes les erreurs de francais. Utilise le format suivant :
        - "mot incorrect" → "mot correct" (explication rapide de l'erreur).
        Si le texte ne contient aucune erreur, indique que le texte est correct avec le format suivant :
        **Grammaire :** Le texte est correct. Félicitations !

        3. **Appréciation générale :** Fournis un commentaire sur la clarté et la pertinence du texte, ainsi que des encouragements pour l'étudiant.

        Assure-toi que ta réponse est bien structurée, concise, bienveillante, et entièrement formatée en Markdown.
        Commence directement avec **Vocabulaire**, **Grammaire**, et **Appréciation générale**.
        Donne moi uniquement la réponse comme output

        """
    return prompt

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_content(text) -> dict:
    prompt = generate_prompt() 
    messages = [
        {"role": "assistant", "content": prompt},
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.2,
        max_tokens=800,
        frequency_penalty=0.0,
    )
    response_text = response.choices[0].message.content
    return response_text
