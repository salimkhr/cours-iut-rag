# text_processor.py
import re
from typing import List, Dict, Tuple


class TextProcessor:
    """
    Traite des données textuelles en les segmentant en blocs optimisés pour l’indexation via des embeddings,
    tout en respectant une stratégie hiérarchique de découpage.

    Cette classe gère la segmentation du texte par paragraphes, par phrases, et inclut un découpage forcé
    en dernier recours si les segments sont trop longs. Elle propose également des fonctionnalités de prétraitement,
    de déduplication, de filtrage, ainsi qu’une analyse statistique des blocs de texte générés.

    :ivar chunk_size: Taille maximale d’un bloc de texte.
    :type chunk_size: int

    :ivar min_chunk_size: Taille minimale pour qu’un bloc soit considéré comme valide.
    :type min_chunk_size: int

    :ivar overlap: Nombre de caractères de chevauchement entre deux blocs consécutifs.
    :type overlap: int
    """

    def __init__(self, chunk_size: int = 300, min_chunk_size: int = 50, overlap: int = 50):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Découpe un texte en chunks optimisés pour l'embedding

        Args:
            text: Texte à découper

        Returns:
            Liste des chunks de texte
        """
        if not text.strip():
            return []

        # Nettoyage préliminaire
        text = self._clean_text(text)

        # Stratégie de découpage hiérarchique
        chunks = []

        # 1. Essayer de découper par paragraphes
        paragraphs = self._split_by_paragraphs(text)

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                if len(paragraph) >= self.min_chunk_size:
                    chunks.append(paragraph)
            else:
                # 2. Découper par phrases
                sentence_chunks = self._split_by_sentences(paragraph)
                chunks.extend(sentence_chunks)

        return self._post_process_chunks(chunks)

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte des caractères indésirables"""
        # Suppression des caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)

        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)

        # Suppression des espaces en début/fin
        text = text.strip()

        return text

    def _split_by_paragraphs(self, text: str) -> List[str]:
        # Découpage sur double saut de ligne ou points suivis de majuscule,
        # en conservant le point à la fin du paragraphe
        parts = re.split(r'(\n\n+|(?<=\.)(?= +[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ]))', text)

        paragraphs = []
        current = ""
        for part in parts:
            if re.match(r'\n\n+', part):
                if current.strip():
                    paragraphs.append(current.strip())
                current = ""
            elif re.match(r'(?<=\.)(?= +[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ])', part):
                # séparation par phrase : on termine le paragraphe en gardant le point
                if current.strip():
                    paragraphs.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            paragraphs.append(current.strip())
        return paragraphs

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Découpe un texte long en chunks basés sur les phrases

        Args:
            text: Texte à découper

        Returns:
            Liste des chunks
        """
        # Découpage approximatif par phrases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ])', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Test si on peut ajouter la phrase au chunk actuel
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Sauvegarder le chunk actuel s'il est assez long
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Commencer un nouveau chunk
                if len(sentence) <= self.chunk_size:
                    current_chunk = sentence
                else:
                    # Phrase trop longue, découpage forcé
                    forced_chunks = self._force_split(sentence)
                    chunks.extend(forced_chunks[:-1])
                    current_chunk = forced_chunks[-1] if forced_chunks else ""

        # Ajouter le dernier chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """
        Découpage forcé d'un texte trop long

        Args:
            text: Texte à découper

        Returns:
            Liste des chunks
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 pour l'espace

            if current_length + word_length <= self.chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Post-traitement des chunks (déduplication, filtrage)
        
        Args:
            chunks: Liste des chunks bruts
        
        Returns:
            Liste des chunks traités
        """
        # Filtrage des chunks trop courts
        filtered_chunks = (
            chunk for chunk in chunks
            if len(chunk) >= self.min_chunk_size
        )

        # Déduplication basique
        seen = set()
        unique_chunks = []
        for chunk in filtered_chunks:
            chunk_normalized = chunk.lower().strip()
            if chunk_normalized not in seen:
                seen.add(chunk_normalized)
                unique_chunks.append(chunk)
        return unique_chunks

    def prepare_texts_for_embedding(self, course_blocks: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Prépare tous les textes pour l'embedding

        Args:
            course_blocks: Liste des blocs de cours chargés

        Returns:
            Tuple (textes, métadonnées)
        """
        texts = []
        meta = []

        total_blocks = sum(len(course.get("blocks", [])) for course in course_blocks)
        processed_blocks = 0

        print(f"📊 Traitement de {total_blocks} blocs...")

        for course in course_blocks:
            course_title = course.get("title", "Sans titre")
            course_file = course.get("file", "unknown")

            for block in course.get("blocks", []):
                processed_blocks += 1

                content = block.get("content", "").strip()
                block_type = block.get("type", "unknown")

                if not content:
                    continue

                # Découpage en chunks
                chunks = self.chunk_text(content)

                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    meta.append({
                        "file": course_file,
                        "course_title": course_title,
                        "type": block_type,
                        "content": chunk,
                        "original_content": content[:200] + "..." if len(content) > 200 else content,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })

        print(f"✅ {len(texts)} chunks créés à partir de {processed_blocks} blocs")
        return texts, meta

    def get_chunk_stats(self, chunks: List[str]) -> Dict:
        """
        Calcule des statistiques sur les chunks

        Args:
            chunks: Liste des chunks

        Returns:
            Dictionnaire des statistiques
        """
        if not chunks:
            return {"count": 0}

        lengths = [len(chunk) for chunk in chunks]

        return {
            "count": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths)
        }