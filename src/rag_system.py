import os
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataloader import CourseDataLoader
from text_processor import TextProcessor


class VectorRAG:
    """
    Représente un système de recherche dense utilisant une approche Retrieval-Augmented Generation (RAG).
    Il permet d’associer les requêtes des utilisateurs à des documents pertinents
    dans un corpus pré-indexé, en s’appuyant sur des embeddings et des mesures de similarité
    pour une récupération efficace de l’information.

    Cette classe intègre des fonctions pour gérer l’indexation des documents,
    retrouver des documents similaires selon les requêtes, et calculer la similarité cosinus.
    Elle utilise un modèle SentenceTransformer pour générer les embeddings, FAISS pour l’index vectoriel,
    et inclut des mécanismes de gestion et mise à jour du corpus.

    :ivar model_name: Nom du modèle transformer utilisé pour générer les embeddings.
    :type model_name: str
    :ivar corpus_dir: Chemin du dossier contenant les fichiers du corpus à indexer.
    :type corpus_dir: str
    :ivar similarity_metric: Métrique utilisée pour le calcul de similarité. Par défaut "cosine".
    :type similarity_metric: str
    :ivar model: Instance du modèle SentenceTransformer pour la génération d’embeddings.
    :type model: Optional[SentenceTransformer]
    :ivar index: Index FAISS utilisé pour stocker et rechercher les vecteurs d’embeddings.
    :type index: Optional[faiss.Index]
    :ivar meta: Métadonnées associées aux documents indexés pour faciliter la récupération.
    :type meta: Optional[List[Dict]]
    :ivar last_modified: Timestamp de la dernière modification du corpus.
    :type last_modified: float
    :ivar data_loader: Instance responsable du chargement des données du cours depuis le disque.
    :type data_loader: CourseDataLoader
    :ivar text_processor: Composant qui traite et prépare les textes pour la génération d’embeddings.
    :type text_processor: TextProcessor
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
        """
        Charge un modèle SentenceTransformer s’il n’est pas déjà chargé.

        Cette méthode vérifie si le modèle est actuellement chargé.
        Si le modèle n’est pas chargé (c’est-à-dire s’il est `None`),
        un modèle SentenceTransformer correspondant à l’attribut `model_name`
        est chargé et assigné à l’attribut `model`.

        :raises RuntimeError: En cas de problème lors du chargement du modèle.

        :return: None
        """
        if self.model is None:
            print(f"🔄 Chargement du modèle {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Modèle chargé")

    def _get_corpus_last_modified(self) -> float:
        """
        Détermine la date de la dernière modification parmi tous les fichiers markdown (.md)
        dans un répertoire donné ainsi que ses sous-répertoires.
        Si le répertoire n’existe pas, retourne 0.

        :param self: L’instance de la classe contenant l’attribut du répertoire du corpus.
        :return: Le timestamp (en secondes depuis l’époque) du fichier markdown le plus récemment modifié,
                 ou 0 si le répertoire n’existe pas ou ne contient aucun fichier markdown.
        :rtype: float
        """
        if not os.path.exists(self.corpus_dir):
            return 0

        latest_time = 0
        for root, dirs, files in os.walk(self.corpus_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    latest_time = max(latest_time, file_time)
        return latest_time

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
       Normalise les embeddings pour en faire des vecteurs unitaires.

        Cette méthode prend un tableau NumPy 2D d’embeddings et normalise chaque vecteur
        pour qu’il ait une norme égale à 1. Cela garantit que tous les embeddings se trouvent
        sur l’hypersphère unité, ce qui est souvent nécessaire pour les calculs de similarité.

        :param embeddings: Tableau NumPy 2D de forme (n_samples, n_features), où chaque ligne représente un vecteur embedding.
        :return: Tableau NumPy 2D de forme (n_samples, n_features), où chaque ligne est un vecteur unitaire normalisé correspondant à la ligne d’entrée.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _build_index(self):
        """
        Construit l’index des embeddings textuels à partir des blocs de cours chargés.

        Cette fonction prépare et encode les données textuelles pour la recherche.
        Elle traite le texte, génère les embeddings avec un modèle prédéfini,
        et les stocke dans un index FAISS adapté selon la métrique de similarité choisie.
        Les embeddings peuvent être configurés pour utiliser la similarité cosinus ou la distance euclidienne.
        Si aucun bloc de cours ou texte valide n’est trouvé, l’index et les métadonnées sont réinitialisés.

        :raises Exception: En cas d’erreur lors du traitement des données, de la génération des embeddings ou de la création de l’index.
        """
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

    def _check_and_reload_if_needed(self) -> bool:
        """
        Vérifie et recharge les données si nécessaire, en fonction du timestamp de modification ou
        si le rechargement est forcé. Cette méthode contrôle la date de dernière modification du corpus
        et décide de reconstruire l’index selon les changements détectés ou si un rechargement est demandé.

        :return: Booléen indiquant si les données ont été rechargées ou non.
        """
        current_modified = self._get_corpus_last_modified()

        if current_modified > self.last_modified:
            print("🔄 Rechargement des données du corpus...")
            self.last_modified = current_modified
            self._build_index()
            return True
        else:
            print("❌ Rechargement non necessaire")
        return False

    def search_similar(self, query: str, k: int = 5, include_neighbors: bool = True) -> List[Dict]:
        """
        Recherche des éléments similaires dans les données indexées à partir de la requête fournie,
        et retourne une liste des correspondances avec leurs scores de similarité.
        Cette fonction utilise le modèle pré-entraîné pour générer les embeddings,
        et recherche dans l’index les résultats les plus pertinents selon la métrique de similarité.
        Le résultat peut optionnellement inclure les éléments voisins adjacents aux entrées correspondantes dans les métadonnées.

        :param query: La requête textuelle à rechercher. Elle est encodée en embedding avant la comparaison de similarité.
        :type query: str
        :param k: Nombre de meilleurs résultats à retourner. Par défaut 5.
        :type k: int
        :param include_neighbors: Indique si les éléments voisins doivent être inclus dans les résultats. Par défaut True.
        :type include_neighbors: bool
        :return: Une liste de dictionnaires représentant les résultats de la recherche.
                 Chaque dictionnaire contient les métadonnées de l’élément trouvé et son score de similarité calculé.
                 Les voisins, s’ils sont inclus, ont le même score de similarité.
        :rtype: List[Dict]
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
        Calcule la similarité cosinus entre deux textes d’entrée en utilisant le modèle fourni,
        sans utiliser de fonctions préconstruites de similarité cosinus.

        La similarité cosinus est calculée manuellement à partir du produit scalaire des vecteurs
        et de leurs magnitudes respectives.

        :param text1: Premier texte d’entrée pour le calcul.
        :type text1: str
        :param text2: Second texte d’entrée pour le calcul.
        :type text2: str
        :return: Le score de similarité cosinus entre les deux textes en nombre flottant.
        :rtype: float
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
        Construit un prompt pour répondre à une question en utilisant les résultats fournis.
        Cette fonction génère un contexte structuré à partir des résultats, qui sert de base à la réponse,
        tout en fournissant des instructions détaillées sur la manière dont la réponse doit être formulée.

        :param results: Liste de dictionnaires représentant des extraits pertinents de cours universitaires.
                        Chaque dictionnaire contient des attributs tels que 'content', 'file', 'type',
                        et éventuellement 'cosine_similarity' ou 'l2_distance' pour l’information de pertinence.
        :type results: List[Dict]
        :param question: La question à laquelle il faut répondre en s’appuyant sur les résultats et le contexte fournis.
        :type question: str
        :return: Un prompt structuré et détaillé utilisé pour générer une réponse claire et pédagogique.
        :rtype: str
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

        prompt = f"""Contexte :

{context}

Instructions :
- Réponds à la question
- Structure ta réponse de manière concise et pédagogique
- Utilise uniquement les informations contenues dans le contexte
- Cite les sources quand c'est pertinent

Question : {question}

Réponse :"""

        return prompt.strip()

    def get_stats(self) -> Dict:
        """
        Fournit des métadonnées statistiques sur l’état actuel de l’objet.

        :return: Un dictionnaire contenant les métadonnées suivantes :
                 - model_name : Le nom du modèle utilisé.
                 - corpus_dir : Le chemin du répertoire du corpus.
                 - similarity_metric : L’identifiant de la métrique de similarité utilisée.
                 - indexed_documents : Le nombre de documents indexés.
                 - last_reload : Une chaîne de caractères représentant le timestamp de la dernière modification.
                 - index_ready : Un booléen indiquant si l’index est prêt (initialisé).
        :rtype: Dict
        """
        return {
            "model_name": self.model_name,
            "corpus_dir": self.corpus_dir,
            "similarity_metric": self.similarity_metric,
            "indexed_documents": len(self.meta) if self.meta else 0,
            "last_reload": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_modified)),
            "index_ready": self.index is not None
        }

    def force_reload(self):
        """
        Force le rechargement d’une ressource ou d’une configuration,
        en contournant toutes les vérifications conditionnelles qui empêcheraient normalement ce rechargement.
        Elle définit l’attribut ``last_modified`` à 0 et déclenche un rechargement forcé
        en appelant le mécanisme de rechargement.

        :raises RuntimeError: En cas d’erreur inattendue lors de l’exécution du rechargement.
        """
        print("🔄 Rechargement forcé...")
        self.last_modified = 0
        self._check_and_reload_if_needed()