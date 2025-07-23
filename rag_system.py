# rag_system.py - Version avec Cosine Similarity
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
    Système RAG vectoriel avec Cosine Similarity et rechargement automatique
    """

    def __init__(self, model_name: str, corpus_dir: str, similarity_metric: str = "cosine"):
        self.model_name = model_name
        self.corpus_dir = corpus_dir
        self.similarity_metric = similarity_metric  # "cosine" ou "l2"
        self.model = None
        self.index = None
        self.meta = None
        self.last_modified = 0

        # Initialisation des composants
        self.data_loader = CourseDataLoader(corpus_dir)
        self.text_processor = TextProcessor()

        print(f"🔄 Initialisation du système RAG avec {similarity_metric} similarity...")
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

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalise les embeddings pour utiliser cosine similarity avec dot product"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _build_index(self):
        """Construit l'index FAISS à partir du corpus"""
        try:
            course_blocks = self.data_loader.load_all_course_blocks()
            if not course_blocks:
                print("⚠️ Aucun fichier de cours trouvé")
                self.index = None
                self.meta = []
                return

            texts, meta = self.text_processor.prepare_texts_for_embedding(course_blocks)
            if not texts:
                print("⚠️ Aucun contenu à indexer")
                self.index = None
                self.meta = []
                return

            print(f"🔄 Génération des embeddings pour {len(texts)} chunks...")
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

            dimension = embeddings.shape[1]

            # Choix de l'index selon la métrique
            if self.similarity_metric == "cosine":
                # Pour cosine similarity, on normalise les embeddings
                embeddings = self._normalize_embeddings(embeddings)
                # IndexFlatIP calcule le dot product (équivalent à cosine sur vecteurs normalisés)
                self.index = faiss.IndexFlatIP(dimension)
                print("📐 Utilisation de Cosine Similarity (IndexFlatIP)")
            else:
                # Distance L2 classique
                self.index = faiss.IndexFlatL2(dimension)
                print("📏 Utilisation de L2 Distance")

            self.index.add(embeddings.astype('float32'))
            self.meta = meta

            print(f"✅ Index créé avec {len(texts)} documents")

        except Exception as e:
            print(f"❌ Erreur lors de la construction de l'index: {e}")
            self.index = None
            self.meta = []

    def _check_and_reload_if_needed(self, force: bool = False) -> bool:
        """Vérifie si le corpus a été modifié et recharge si nécessaire"""
        current_modified = self._get_corpus_last_modified()

        if force or current_modified > self.last_modified:
            print("🔄 Rechargement des données du corpus...")
            self.last_modified = current_modified
            self._build_index()
            return True
        return False

    def search_similar(self, query: str, k: int = 5, include_neighbors: bool = True) -> List[Dict]:
        """
        Recherche les documents similaires à la requête avec cosine similarity

        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats principaux à retourner
            include_neighbors: Ajoute les extraits voisins si True

        Returns:
            Liste enrichie de documents pertinents avec scores
        """
        self._check_and_reload_if_needed()

        if self.index is None or not self.meta:
            return []

        try:
            query_emb = self.model.encode([query], convert_to_numpy=True)

            # Normaliser la requête si on utilise cosine similarity
            if self.similarity_metric == "cosine":
                query_emb = self._normalize_embeddings(query_emb)

            scores, indices = self.index.search(query_emb.astype('float32'), k)

            seen_indices = set()
            results = []

            for i, score in zip(indices[0], scores[0]):
                if i >= len(self.meta):
                    continue

                # Ajouter chunk principal
                if i not in seen_indices:
                    result = self.meta[i].copy()

                    # Conversion du score selon la métrique
                    if self.similarity_metric == "cosine":
                        # IndexFlatIP retourne le dot product (plus élevé = plus similaire)
                        result['similarity_score'] = float(score)
                        result['cosine_similarity'] = float(score)  # Score déjà normalisé
                    else:
                        # IndexFlatL2 retourne la distance (plus bas = plus similaire)
                        result['similarity_score'] = float(score)
                        result['l2_distance'] = float(score)

                    results.append(result)
                    seen_indices.add(i)

                # Ajouter les voisins
                if include_neighbors:
                    for neighbor_offset in [-1, 1]:
                        ni = i + neighbor_offset
                        if 0 <= ni < len(self.meta) and ni not in seen_indices:
                            neighbor = self.meta[ni].copy()
                            neighbor['similarity_score'] = float(score)
                            results.append(neighbor)
                            seen_indices.add(ni)

            # Trier par score (décroissant pour cosine, croissant pour L2)
            if self.similarity_metric == "cosine":
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
            else:
                results.sort(key=lambda x: x['similarity_score'])

            return results

        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return []

    def calculate_cosine_similarity_manual(self, text1: str, text2: str) -> float:
        """
        Calcule manuellement la cosine similarity entre deux textes
        Utile pour debug ou comparaisons
        """
        emb1 = self.model.encode([text1], convert_to_numpy=True)
        emb2 = self.model.encode([text2], convert_to_numpy=True)

        # Calcul manuel du cosine
        dot_product = np.dot(emb1[0], emb2[0])
        norm1 = np.linalg.norm(emb1[0])
        norm2 = np.linalg.norm(emb2[0])

        return float(dot_product / (norm1 * norm2))

    def build_rag_prompt(self, results: List[Dict], question: str) -> str:
        """
        Construit le prompt RAG à partir des résultats avec info de similarité
        """
        if not results:
            return f"""Réponds à la question suivante du mieux que tu peux :\n\nQuestion : {question}\n\nRéponse :"""

        context = ""
        for i, r in enumerate(results, 1):
            score_info = ""
            if 'cosine_similarity' in r:
                score_info = f" (similarité: {r['cosine_similarity']:.3f})"
            elif 'l2_distance' in r:
                score_info = f" (distance: {r['l2_distance']:.3f})"

            context += f"\n### Extrait {i} ({r['file']}) – {r['type']}{score_info}\n{r['content']}\n"

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
        """Retourne des statistiques sur le système"""
        return {
            "model_name": self.model_name,
            "corpus_dir": self.corpus_dir,
            "similarity_metric": self.similarity_metric,
            "indexed_documents": len(self.meta) if self.meta else 0,
            "last_reload": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_modified)),
            "index_ready": self.index is not None
        }

    def force_reload(self):
        """Force le rechargement du corpus"""
        print("🔄 Rechargement forcé...")
        self.last_modified = 0
        self._check_and_reload_if_needed(force=True)
