import os
from typing import List, Dict
import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension
import re


def _add_list_item(stack, item, indent, level_indent=2):
    """
    Ajoute un item à une liste hiérarchique en fonction de son niveau d'indentation.
    """
    level = indent // level_indent

    def insert(stack, level, item):
        if level == 0:
            stack.append(item)
        else:
            last = stack[-1]
            if isinstance(last, str):
                last = {"content": last, "subitems": []}
                stack[-1] = last
            elif "subitems" not in last:
                last["subitems"] = []
            insert(last["subitems"], level - 1, item)

    insert(stack, level, item)


def _parse_markdown_table(table_lines):
    def split_row(line):
        return [cell.strip() for cell in line.strip().strip('|').split('|')]

    rows = [split_row(line) for line in table_lines if '|' in line]
    if len(rows) >= 2 and re.match(r"^-{2,}", rows[1][0]):
        headers = rows[0]
        data_rows = rows[2:]  # skip separator
    else:
        headers = []
        data_rows = rows

    return {
        "type": "table",
        "headers": headers,
        "rows": data_rows
    }

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
        blocks = []
        lines = md_content.splitlines()
        buffer = []
        current_type = None
        list_stack = []

        def flush_buffer():
            nonlocal buffer, current_type, list_stack
            if current_type in ("list", "ordered_list") and list_stack:
                blocks.append({"type": current_type, "items": list_stack})
                list_stack = []
            elif buffer:
                content = "\n".join(buffer).strip()
                if content:
                    blocks.append({"type": current_type or "content", "content": content})
            elif current_type == "table" and buffer:
                table_block = _parse_markdown_table(buffer)
                blocks.append(table_block)
            buffer = []
            current_type = None

        in_code_block = False

        for line in lines:
            raw_line = line
            line = line.rstrip("\n")
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Bloc de code
            if stripped.startswith("```"):
                if not in_code_block:
                    flush_buffer()
                    in_code_block = True
                    current_type = "code"
                    buffer = []
                else:
                    flush_buffer()
                    in_code_block = False
                continue

            if in_code_block:
                buffer.append(raw_line)
                continue

            # Titre
            if stripped.startswith("#"):
                flush_buffer()
                level = len(stripped) - len(stripped.lstrip("#"))
                content = stripped.lstrip("#").strip()
                blocks.append({"type": "heading", "level": level, "content": content})
                continue

            # Tableau Markdown
            # Détection de ligne de tableau (ligne contenant des |)
            if "|" in line and re.search(r"\|", stripped):
                if current_type != "table":
                    flush_buffer()
                    current_type = "table"
                    buffer = []
                buffer.append(line)
                continue

            # En fin de bloc ou ligne vide : si tableau, on le parse
            if not stripped and current_type == "table":
                if buffer:
                    table_block = _parse_markdown_table(buffer)
                    blocks.append(table_block)
                    buffer = []
                    current_type = None
                continue

            # Liste ordonnée
            if re.match(r"^\d+\.\s", stripped):
                match = re.match(r"^(\d+\.)\s+(.*)", stripped)
                item = match.group(2)
                if current_type != "ordered_list":
                    flush_buffer()
                    current_type = "ordered_list"
                    list_stack = []
                _add_list_item(list_stack, item, indent)
                continue

            # Liste non ordonnée
            if re.match(r"^[-*+]\s", stripped):
                item = re.sub(r"^[-*+]\s+", "", stripped)
                if current_type != "list":
                    flush_buffer()
                    current_type = "list"
                    list_stack = []
                _add_list_item(list_stack, item, indent)
                continue

            # Ligne vide
            if not stripped:
                flush_buffer()
                continue

            # Par défaut : texte
            if current_type not in (None, "content"):
                flush_buffer()
            current_type = "content"
            buffer.append(line)

        flush_buffer()
        return blocks

    # Les autres méthodes (validation, infos, listing) peuvent être adaptées selon les besoins
