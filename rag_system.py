# rag_system.py
import os
import json
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataloader import CourseDataLoader
from text_processor import TextProcessor


class VectorRAG:
    """
    Système RAG vectoriel avec rechargement automatique des données
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', corpus_dir: str = "./rag/corpus"):
        self.model_name = model_name
        self.corpus_dir = corpus_dir
        self.model = None
        self.index = None
        self.meta = None
        self.last_modified = 0

        # Initialisation des composants
        self.data_loader = CourseDataLoader(corpus_dir)
        self.text_processor = TextProcessor()

        print(f"🔄 Initialisation du système RAG...")
        self._load_model()
        self._check_and_reload_if_needed()

    def _load_model(self):
        """Charge le modèle SentenceTransformer"""
        if self.model is None:
            print(f"🔄 Chargement du modèle {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Modèle chargé")

    def _get_corpus_last_modified(self) -> float:
        """Obtient le timestamp de dernière modification du corpus"""
        if not os.path.exists(self.corpus_dir):
            return 0

        latest_time = 0
        for root, dirs, files in os.walk(self.corpus_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    latest_time = max(latest_time, file_time)
        return latest_time

    def _check_and_reload_if_needed(self) -> bool:
        """Vérifie si le corpus a été modifié et recharge si nécessaire"""
        current_modified = self._get_corpus_last_modified()

        if current_modified > self.last_modified:
            print("🔄 Changements détectés dans le corpus, rechargement...")
            self.last_modified = current_modified
            self._build_index()
            return True
        return False

    def _build_index(self):
        """Construit l'index FAISS à partir du corpus"""
        try:
            # Chargement des données
            course_blocks = self.data_loader.load_all_course_blocks()
            if not course_blocks:
                print("⚠️ Aucun fichier de cours trouvé")
                self.index = None
                self.meta = []
                return

            # Préparation des textes
            texts, meta = self.text_processor.prepare_texts_for_embedding(course_blocks)
            if not texts:
                print("⚠️ Aucun contenu à indexer")
                self.index = None
                self.meta = []
                return

            # Génération des embeddings
            print(f"🔄 Génération des embeddings pour {len(texts)} chunks...")
            embeddings = self.model.encode(texts,
                                           convert_to_numpy=True,
                                           show_progress_bar=True)

            # Construction de l'index FAISS
            print("🔄 Construction de l'index FAISS...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.meta = meta

            print(f"✅ Index créé avec {len(texts)} documents")

        except Exception as e:
            print(f"❌ Erreur lors de la construction de l'index: {e}")
            self.index = None
            self.meta = []

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Recherche les documents similaires à la requête

        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats à retourner

        Returns:
            Liste des documents pertinents avec scores de similarité
        """
        # Vérification du rechargement
        self._check_and_reload_if_needed()

        if self.index is None or not self.meta:
            return []

        try:
            query_emb = self.model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_emb.astype('float32'), k)

            results = []
            for i, distance in zip(indices[0], distances[0]):
                if i < len(self.meta):  # Vérification de sécurité
                    result = self.meta[i].copy()
                    result['similarity_score'] = float(distance)
                    results.append(result)

            return results

        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return []

    def build_rag_prompt(self, results: List[Dict], question: str) -> str:
        """
        Construit le prompt RAG à partir des résultats de recherche

        Args:
            results: Résultats de la recherche vectorielle
            question: Question de l'utilisateur

        Returns:
            Prompt formaté pour le LLM
        """
        if not results:
            return f"""Réponds à la question suivante du mieux que tu peux :

Question : {question}

Réponse :"""

        context = ""
        for i, r in enumerate(results, 1):
            context += f"\n### Extrait {i} ({r['file']}) – {r['type']}\n{r['content']}\n"

        prompt = f"""Voici des extraits de cours universitaires pertinents :

{context}

Instructions :
- Réponds à la question en te basant principalement sur ces extraits
- Structure ta réponse de manière claire et pédagogique
- Si la réponse n'est pas complètement dans les extraits, utilise tes connaissances générales mais indique-le
- Cite les sources quand c'est pertinent

Question : {question}

Réponse :"""

        return prompt.strip()

    def get_stats(self) -> Dict:
        """Retourne les statistiques du système"""
        return {
            "model_name": self.model_name,
            "corpus_dir": self.corpus_dir,
            "indexed_documents": len(self.meta) if self.meta else 0,
            "last_reload": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_modified)),
            "index_ready": self.index is not None
        }

    def force_reload(self):
        """Force le rechargement du corpus"""
        print("🔄 Rechargement forcé du corpus...")
        self.last_modified = 0  # Force le rechargement
        self._check_and_reload_if_needed()