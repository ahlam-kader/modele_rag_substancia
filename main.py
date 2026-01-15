import os
import fitz
import re
import faiss
import numpy as np
import pickle
import shutil
import random
from typing import List, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
import uvicorn

app = FastAPI()

# --- Modèles pour FastAPI ---
class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage]
    user_profile: dict  # Reçu du front-end

# --- Classe Moteur RAG ---
class SubstanciaEngineUniversal:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        # self.encoder supprimé car on utilise l'API Mistral Embeddings (gain de RAM)
        self.chunks = []
        self.index = None
        self.documents_meta = {}
        self.last_question = ""
        self.last_context = ""
        self.last_answer = ""
        self.user_profile = {}
        self.load_index_if_exists()

    def clean_text_expert(self, text):
        text = re.sub(r'[|]', '', text)
        text = re.sub(r'\(\s*\$(.*?)\$\s*\)', r'$\1$', text)
        text = re.sub(r'\s+', ' ', text)
        def format_matrix(matrix_str):
            rows = matrix_str.strip().split(';')
            rows_latex = [" & ".join(r.strip().split()) for r in rows if r.strip()]
            return "\\begin{bmatrix}" + " \\\\ ".join(rows_latex) + "\\end{bmatrix}"
        text = re.sub(r'MATRIX\[(.*?)\]', format_matrix, text)
        return text.strip()

    def get_embedding(self, texts):
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts
        )
        return [np.array(item.embedding, dtype="float32") for item in response.data]

    def load_index_if_exists(self):
        if os.path.exists("faiss_index.idx") and os.path.exists("chunks_meta.pkl"):
            self.index = faiss.read_index("faiss_index.idx")
            with open("chunks_meta.pkl", "rb") as f:
                self.chunks = pickle.load(f)

    def summarizer(self, text):
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": f"Résume ce texte en 100 mots max : {text}"}]
        )
        return response.choices[0].message.content

    def ingest_pdfs(self, file_paths, summarizer=None):
        if not file_paths: return "❌ Aucun fichier sélectionné."
        self.chunks = []
        for path in file_paths:
            doc = fitz.open(path)
            fname = os.path.basename(path)
            metadata = doc.metadata
            author = metadata.get("author", "Inconnu")
            date = metadata.get("creationDate", "Date inconnue")
            for i, page in enumerate(doc):
                content = self.clean_text_expert(page.get_text("text"))
                if len(content) > 50:
                    self.chunks.append({
                        "source": fname,
                        "page": i + 1,
                        "content": content,
                        "author": author,
                        "date": date
                    })
            self.documents_meta[fname] = {"author": author, "date": date, "summary": "", "full_text": ""}
            doc.close()

        if not self.chunks: return "❌ PDF vides."

        contents = [c["content"] for c in self.chunks]
        # Ingestion par lots pour éviter de surcharger l'API si le PDF est énorme
        embeddings = np.vstack(self.get_embedding(contents)).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, "faiss_index.idx")
        with open("chunks_meta.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        return f"✅ {len(self.chunks)} sections indexées."

    def ask(self, question, chat_history):

        if self.index is None:
            return "❌ Veuillez charger des documents PDF.", chat_history

        q_lower = question.lower()
        is_follow_up = any(k in q_lower for k in ["explique-moi plus", "explique plus", "développe", "approfondis", "encore", "plus", "explique"])
        is_summary = any(k in q_lower for k in ["résumé", "resume", "résumer", "résume"])
        is_reco = any(k in q_lower for k in ["recommandation", "conseil", "recommande", "recommende", "recommend"])
        is_quiz = any(k in q_lower for k in ["quiz", "qcm"])
        is_all_docs = any(k in q_lower for k in ["tous les documents", "tous les pdf", "le dossier"])
        is_qa = any(k in q_lower for k in ["question reponse", "question réponse", "questions réponses", "questions reponses", "question_reponse", "question-réponse", "QR", "Q-R"])

        current_context = ""
        sources_md = ""

        if (is_summary or is_follow_up or is_quiz or is_reco or is_qa) and self.last_context:
            current_context = self.last_context
            if is_summary:
                current_context = self.clean_text_expert(self.last_context)
                instruction = f"Résume le texte suivant : {current_context}"
                sources_md = "*(Résumé basé sur le dernier sujet traité)*"
            elif is_follow_up:
                current_context = self.clean_text_expert(self.last_context)
                instruction = f"Explique en détail le texte suivant : {current_context}"
                sources_md = "*(Explication détaillée basée sur le dernier sujet traité)*"
            elif is_quiz:
                current_context = self.clean_text_expert(self.last_context)
                instruction = f"Génère un quiz basé sur le texte suivant : {current_context}"
                sources_md = "*(Quiz basé sur le dernier sujet traité)*"
            elif is_reco:
                current_context = self.clean_text_expert(self.last_context)
                instruction = f"Donne des recommandations basées sur le texte suivant : {current_context}"
                sources_md = "*(Recommandation basée sur le dernier sujet traité)*"
            elif is_qa:
                current_context = self.clean_text_expert(self.last_context)
                instruction = f"Génère des questions-réponses basées sur le texte suivant : {current_context}"
                sources_md = "*(Questions-Réponses basées sur le dernier sujet traité)*"

        else:
            q_emb = np.vstack(self.get_embedding([question])).astype("float32")
            faiss.normalize_L2(q_emb)
            scores, idx = self.index.search(q_emb, min(8, len(self.chunks)))
            retrieved = [self.chunks[i]["content"] for score, i in zip(scores[0], idx[0]) if score >= 0.8]
            
            if retrieved:
                current_context = self.clean_text_expert("\n\n".join(retrieved))
                instruction = f"Réponds à la question : {question}"
                sources_md = "*(Sources extraites des documents)*"
            else:
                current_context = "Aucune source fiable trouvée."
                instruction = question
        
        self.last_context = current_context
        salt = random.randint(1, 9999)
        style_prof = random.choice(["Expert pédagogique clair avec exemples et analogies",
            "Expert mathématique précis avec exercices",
            "Expert motivant et interactif avec quiz",
            "Expert pratique avec conseils concrets et applications",
            "Professeur explicatif avec résumés courts et exercices",
            "Professeur créatif avec anecdotes et exemples visuels",
            "Expert technique détaillé avec démonstrations",
            "Professeur didactique et structuré avec questions interactives",
            "Expert motivant avec mini-projets pratiques",
            "Professeur analytique avec points clés et recommandations"
        ])
        
        profile_text = "\n".join([f"- {k}: {v}" for k, v in self.user_profile.items()])

        past_interactions = ""
        for msg in chat_history[-3:]:
            past_interactions += f"{msg['role']}: {msg['content'][:100]}...\n"
        
        random_seed = salt
        query = question

        system_prompt = f"""
        RÈGLES CRITIQUES :
        1. Utilise UNIQUEMENT le document ou les documents fournis pour les faits techniques.
        2. VARIABILITÉ : Ne réponds jamais deux fois de la même façon (Seed: {salt}). Change tes analogies et exemples.
        3. LaTeX :
          - $ ... $ → inline
          - $$ ... $$ → isolé sur une ligne
          - Ne jamais utiliser | ou ( ) autour du LaTeX
        4. STRUCTURE :
          - Introduction concise
          - Développement (selon l'approche de l'utilisateur)
          - Points clés / Résumé
          - Défi / Exercice / Quiz : Propose toujours 1-2 questions simples ou exercice pratique liés au sujet, avec correction ou indice. Si le texte est court, adapte un exemple ou une recommandation pratique.
        5. Important pour les réponses sans source fiable :
          - Si aucune source fiable n’est trouvée, afficher **uniquement** le message "Aucune source fiable trouvée..." + contextualisation brève + prochaine étape.
          - **Ne jamais** ajouter d’exercices, de quiz, de recommandations ou d’emojis.
          - L’utilisateur doit uniquement voir la phrase contextualisée et la demande de précision.


        Tu es un professeur expert ({style_prof}) en pédagogie et en éducation, capable de devenir spécialiste dans n'importe quel domaine : mathématiques, physique, chimie, informatique, médecine, biologie, langues, littérature, arts, etc.

        IMPORTANT (VISION ET CONTINUITÉ) :
        Analyse la relation entre les questions dans une même conversation pour comprendre le contexte et l'intention de l'utilisateur.
        Utilise les interactions précédentes si l'utilisateur dit 'plus', 'encore', 'détaille' ou pose une question liée :
        {past_interactions}

        AFFICHAGE MATHÉMATIQUE ET MULTIMÉDIA :
        - Formules : $inline$ et $$isolé$$ uniquement
        - Matrices : transformer en LaTeX propre
        - Tableaux : Markdown ou LaTeX formaté
        - Images : indiquer 'Image extraite page X'

        LANGUE ET PRIORITÉS :
        1. Identifie la langue de la question et des documents (PDF).
        2. RÈGLE D'OR : Réponds prioritairement dans la langue de la QUESTION de l'utilisateur.
        3. Si la question est ambiguë, utilise la langue principale des DOCUMENTS fournis.
        4. En dernier recours, utilise la langue spécifiée dans le PROFIL UTILISATEUR : {self.user_profile.get('langue', 'Français')}.
        5. Tu as l'autorisation de répondre en Français, Anglais ou Arabe (selon le choix de l'utilisateur). Adapte ton alphabet (latin ou arabe) parfaitement au contexte.
        - Si langue non supportée : "Je ne peux répondre qu’en français ou en anglais."

        PÉDAGOGIE ET INTERACTIVITÉ :
        - Réponses structurées : Introduction → Développement → Points clés
        - Propose systématiquement 'Exercice' à la fin
        - Corrige la réponse de l'élève avec bienveillance si exercice fait
        - Utilise analogies, exemples, exercices, résumés, schémas textuels si nécessaire

        FLEXIBILITÉ :
        - Devient expert immédiat dans le domaine de la question
        - Réponses variées : analogies, exemples, exercices, résumés, schémas textuels

        PROFIL UTILISATEUR :
        {profile_text}

        VARIABILITÉ : {random_seed}

        En tant que Professeur Universitaire, crée un support complet basé sur ces analyses :
        {current_context}

        FLEXIBILITÉ PÉDAGOGIQUE :
        - Même pour une réponse courte, termine toujours par :
        Exercice / Quiz
        - 1 à 2 questions ou exercice
        - Correction ou indice
        - Recommandation pratique
        - Si possible, illustre avec tableau ou exemple chiffré
        - Si texte source petit, adapte un mini-exemple pour clarifier

        Structure ton cours ainsi :
        # Titre du Cours Magistral : {query}
        ## 1. Introduction et Concepts Clés
        ## 2. Développement Technique et Mathématique
        ## 3. Recommandations et Bonnes Pratiques
        ## 4. Sources Consultées (Fichier et Page)
        Rôle : "Professeur Expert Universel Multi-Agent". Maîtrise absolue dans tous les domaines.
        Analyse rigoureuse : précise systématiquement la page, reconstruis matrices cassées, formate tableaux.
        Mode interactif : propose menu Résumé, explications détaillées, Quiz (multi-choix et 1 seul choix), Questions-Réponses (10-12 questions avec 6-7 lignes d'explications).
        Mentorat : sois encourageant et pédagogique.

        EXEMPLE DE STRUCTURE ATTENDUE :
        # Titre : Fonction de coût k-means
        ## 1. Introduction
        Le k-means est un algorithme de clustering non supervisé...
        ## 2. Développement
        Formule matricielle : $$E(Z,M)=||X-ZM||_F^2$$
        Explication ligne par ligne, exemple chiffré, tableau des centres et clusters.
        ## 3. Points clés
        - K-means minimise la distance intra-cluster
        - Z est une matrice binaire
        - M contient les centres
        ## 4. Exercice
        1. Calcule $$E(k)$$ pour X=[[1,1],[2,1]] et M=[[1.5,1.0]]
        2. Question conceptuelle : pourquoi utiliser la norme de Frobenius ?
        3. quiz : vrai/faux - "Z contient des probabilités"
        ## 5. Sources
        - hhhhhh.pdf, p.00
        """

        user_prompt = f"""
        CONTEXTE :\n{current_context}\n{profile_text}\n\nDEMANDE : {instruction}
        {"(Note : Utilise un maximum d'extraits différents du contexte pour cette analyse globale)" if is_all_docs else ""}

        QUESTION ACTUELLE DE L'UTILISATEUR :
        {question}

        INSTRUCTIONS DE RÉPONSE :
        1. Si la question est "plus", "encore", "détaille", "continue", "more", regarde l'historique fourni pour approfondir le sujet précédent.
        2. Reformule la question pour montrer compréhension.
        3. Réponse détaillée basée sur les documents fournis ou tes connaissances d'expert.
        4. Terminer par des points clés et un Exercice.
        """

        response = self.client.chat.complete(
            model="mistral-large-latest",
            temperature=0.8,
            messages=[
                {"role": "system", "content": system_prompt},
                *[{"role": m["role"], "content": m["content"]} for m in chat_history],
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.choices[0].message.content
        return answer, sources_md

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
engine = SubstanciaEngineUniversal(MISTRAL_KEY)

# --- Endpoints ---
@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    os.makedirs("temp", exist_ok=True)
    file_paths = []
    for file in files:
        path = f"temp/{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(path)
    res = engine.ingest_pdfs(file_paths)
    return {"status": res}

@app.post("/ask")
async def ask_api(request: AskRequest):
    engine.user_profile = request.user_profile
    history = [{"role": m.role, "content": m.content} for m in request.chat_history]
    answer, sources = engine.ask(request.question, history)
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
