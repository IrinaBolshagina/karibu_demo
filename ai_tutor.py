from huggingface_hub import InferenceClient


# def make_prompt(question, answer):
#     return f"""Évalue ma réponse.
#     Question : "{question}"
#     Ma réponse : "{answer}"
#     Évalue cette réponse en suivant exactement ce format :
#     <b>Compréhension</b>
#     <p>[Analyse si la réponse correspond à la question posée]</p>
#     <b>Vocabulaire</b>
#     <p>[Analyse du vocabulaire utilisé]</p>
#     <b>Grammaire</b>
#     <p>[Corrections nécessaires OU "Le texte est correct. Félicitations !"]</p>
#     <b>Appréciation générale</b>
#     <p>[Bref commentaire encourageant]</p>
#     Importante: Évalue uniquement le texte fourni, sans le réécrire ni en générer un nouveau.
#     """

# def correct_text(question, answer, hf_token):
#     client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
#     output = client.chat.completions.create(
#     messages=[
#         {"role": "system", "content": "Tu es un enseignant qui évalue ma réponse."},
#         {"role": "user", "content": make_prompt(question, answer)},
#     ],
#     max_tokens=1024,)
#     return output['choices'][0]['message']['content']

def clean_text_for_prompt(text):
    # Remove potential problematic characters while preserving French accents
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    return text.strip()

def correct_text(question, answer, hf_token):
    prompt = f"""Évalue ma réponse.
    Question : "{clean_text_for_prompt(question)}"
    Ma réponse : "{clean_text_for_prompt(answer)}"
    Évalue cette réponse en suivant exactement ce format :

    <b>Compréhension</b>
    <p>[Analyse si la réponse correspond à la question posée]</p>

    <b>Vocabulaire</b>
    <p>[Analyse du vocabulaire utilisé]</p>

    <b>Grammaire</b>
    <p>[Corrections nécessaires OU "Le texte est correct. Félicitations !"]</p>

    <b>Appréciation générale</b>
    <p>[Bref commentaire encourageant]</p>

    Important : Respectez strictement ce format et gardez les espaces entre les sections."""

        
    client = InferenceClient(
        model="meta-llama/Llama-3.3-70B-Instruct", 
        token=hf_token
        )

    output = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Tu es un enseignant qui évalue ma réponse."},
        {"role": "user", "content": prompt},
    ],

    max_tokens=1024,)
    result = output.choices[0].message.content
    return result    