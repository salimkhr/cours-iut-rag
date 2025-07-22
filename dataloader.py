import os
from typing import List, Dict
import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension

class CourseDataLoader:
    """
    Gestionnaire de chargement des données de cours depuis des fichiers Markdown
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir

    def load_all_course_blocks(self) -> List[Dict]:
        course_blocks = []

        print(self.corpus_dir)

        if not os.path.exists(self.corpus_dir):
            print(f"⚠️ Répertoire corpus introuvable: {self.corpus_dir}")
            return course_blocks

        total_files = 0
        loaded_files = 0

        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                if file.endswith(".md"):
                    print(f"📊 Traitement de {file}")
                    total_files += 1
                    full_path = os.path.join(root, file)

                    try:
                        with open(full_path, encoding="utf-8") as f:
                            md_content = f.read()

                        blocks = self._parse_markdown_blocks(md_content)
                        if blocks:
                            course_blocks.append({
                                "file": os.path.relpath(full_path, self.corpus_dir),
                                "full_path": full_path,
                                "blocks": blocks,
                            })
                            loaded_files += 1
                        else:
                            print(f"⚠️ Aucun bloc extrait dans {full_path}")

                    except Exception as e:
                        print(f"❌ Erreur lors du chargement de {full_path}: {e}")

        print(f"📊 {loaded_files}/{total_files} fichiers chargés depuis {self.corpus_dir}")
        return course_blocks

    def _parse_markdown_blocks(self, md_content: str) -> List[Dict]:
        """
        Parse le contenu Markdown et extrait les blocs (titre, texte, liste, code, table).

        Ici on fait simple : découpe par lignes, détecte les blocs de code, titres, listes.

        Returns:
            Liste des blocs sous forme de dictionnaires avec 'type' et 'content'
        """
        blocks = []
        lines = md_content.splitlines()
        buffer = []
        current_type = None

        def flush_buffer():
            nonlocal buffer, current_type
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    blocks.append({"type": current_type or "content", "content": content})
                buffer = []
                current_type = None

        in_code_block = False
        code_block_lang = None

        for line in lines:
            # Détection bloc code markdown ```
            if line.startswith("```"):
                if not in_code_block:
                    flush_buffer()
                    in_code_block = True
                    code_block_lang = line[3:].strip()
                    current_type = "code"
                    buffer = []
                else:
                    # fin du bloc code
                    flush_buffer()
                    in_code_block = False
                    code_block_lang = None
                continue

            if in_code_block:
                buffer.append(line)
                continue

            # Détection titre Markdown
            if line.startswith("#"):
                flush_buffer()
                level = len(line) - len(line.lstrip('#'))
                content = line.lstrip('#').strip()
                blocks.append({"type": "heading", "level": level, "content": content})
                continue

            # Détection liste (lignes commençant par - ou *)
            if line.strip().startswith(("-", "*")):
                if current_type != "list":
                    flush_buffer()
                    current_type = "list"
                    buffer = []
                buffer.append(line.strip())
                continue

            # Ligne vide : flush buffer
            if not line.strip():
                flush_buffer()
                continue

            # Par défaut texte
            if current_type not in (None, "content"):
                flush_buffer()
                current_type = "content"
                buffer = []

            current_type = current_type or "content"
            buffer.append(line)

        flush_buffer()
        return blocks

    # Les autres méthodes (validation, infos, listing) peuvent être adaptées selon les besoins
