from huggingface_hub import InferenceClient


def make_prompt(text):
    return f"""
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

    Texte soumis : {text}
    """

def correct_text(text, hf_token):
    client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
    output = client.text_generation(make_prompt(text), max_new_tokens=4000)
    return output
