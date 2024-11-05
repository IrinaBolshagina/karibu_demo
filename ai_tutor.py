from huggingface_hub import InferenceClient


def make_prompt(question, answer):
    return f"""<s>[INST] Tu es un enseignant qui doit évaluer la réponse d'un étudiant.

Question posée à l'étudiant : "{question}"

Réponse de l'étudiant : "{answer}"

Évalue cette réponse en suivant exactement ce format :

**Compréhension**
[Analyse si la réponse correspond à la question posée]

**Vocabulaire**
[Analyse du vocabulaire utilisé]

**Grammaire**
[Corrections nécessaires OU "Le texte est correct. Félicitations !"]

**Appréciation générale**
[Bref commentaire encourageant]

Importante: Évalue uniquement le texte fourni, sans le réécrire ni en générer un nouveau.[/INST]</s>"""

def correct_text(text, hf_token):
    client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
    output = client.text_generation(make_prompt(text), max_new_tokens=4000)
    return output
