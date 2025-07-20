# dataloader.py
import os
import json
from typing import List, Dict


class CourseDataLoader:
    """
    Gestionnaire de chargement des données de cours depuis les fichiers JSON
    """

    def __init__(self, corpus_dir: str = "./rag/corpus"):
        self.corpus_dir = corpus_dir

    def load_all_course_blocks(self) -> List[Dict]:
        """
        Charge tous les fichiers JSON du corpus

        Returns:
            Liste des blocs de cours avec métadonnées
        """
        course_blocks = []

        if not os.path.exists(self.corpus_dir):
            print(f"⚠️ Répertoire corpus introuvable: {self.corpus_dir}")
            return course_blocks

        total_files = 0
        loaded_files = 0

        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                if file.endswith(".json"):
                    total_files += 1
                    full_path = os.path.join(root, file)

                    try:
                        with open(full_path, encoding="utf-8") as f:
                            data = json.load(f)

                        # Validation de base de la structure
                        if self._validate_course_structure(data, full_path):
                            data["file"] = os.path.relpath(full_path, self.corpus_dir)
                            data["full_path"] = full_path
                            course_blocks.append(data)
                            loaded_files += 1

                    except json.JSONDecodeError as e:
                        print(f"❌ Erreur JSON dans {full_path}: {e}")
                    except Exception as e:
                        print(f"❌ Erreur lors du chargement de {full_path}: {e}")

        print(f"📊 {loaded_files}/{total_files} fichiers chargés depuis {self.corpus_dir}")
        return course_blocks

    def _validate_course_structure(self, data: Dict, file_path: str) -> bool:
        """
        Valide la structure d'un fichier de cours

        Args:
            data: Données JSON chargées
            file_path: Chemin du fichier pour les messages d'erreur

        Returns:
            True si la structure est valide
        """
        if not isinstance(data, dict):
            print(f"⚠️ Structure invalide dans {file_path}: doit être un objet JSON")
            return False

        if "blocks" not in data:
            print(f"⚠️ Clé 'blocks' manquante dans {file_path}")
            return False

        if not isinstance(data["blocks"], list):
            print(f"⚠️ 'blocks' doit être une liste dans {file_path}")
            return False

        # Validation des blocs
        valid_blocks = 0
        for i, block in enumerate(data["blocks"]):
            if self._validate_block_structure(block, file_path, i):
                valid_blocks += 1

        if valid_blocks == 0:
            print(f"⚠️ Aucun bloc valide trouvé dans {file_path}")
            return False

        return True

    def _validate_block_structure(self, block: Dict, file_path: str, block_index: int) -> bool:
        """
        Valide la structure d'un bloc de contenu

        Args:
            block: Bloc de contenu
            file_path: Chemin du fichier
            block_index: Index du bloc

        Returns:
            True si le bloc est valide
        """
        if not isinstance(block, dict):
            print(f"⚠️ Bloc {block_index} invalide dans {file_path}: doit être un objet")
            return False

        required_fields = ["type", "content"]
        for field in required_fields:
            if field not in block:
                print(f"⚠️ Champ '{field}' manquant dans le bloc {block_index} de {file_path}")
                return False

        if not block["content"].strip():
            print(f"⚠️ Contenu vide dans le bloc {block_index} de {file_path}")
            return False

        return True

    def get_file_info(self, file_path: str) -> Dict:
        """
        Obtient des informations sur un fichier de cours

        Args:
            file_path: Chemin vers le fichier JSON

        Returns:
            Dictionnaire avec les informations du fichier
        """
        full_path = os.path.join(self.corpus_dir, file_path)

        if not os.path.exists(full_path):
            return {"error": "Fichier introuvable"}

        try:
            stat = os.stat(full_path)
            with open(full_path, encoding="utf-8") as f:
                data = json.load(f)

            return {
                "file_path": file_path,
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
                "blocks_count": len(data.get("blocks", [])),
                "title": data.get("title", "Sans titre"),
                "description": data.get("description", ""),
            }

        except Exception as e:
            return {"error": str(e)}

    def list_corpus_files(self) -> List[Dict]:
        """
        Liste tous les fichiers du corpus avec leurs informations

        Returns:
            Liste des informations de fichiers
        """
        files_info = []

        if not os.path.exists(self.corpus_dir):
            return files_info

        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.corpus_dir)
                    info = self.get_file_info(relative_path)
                    files_info.append(info)

        return sorted(files_info, key=lambda x: x.get("file_path", ""))