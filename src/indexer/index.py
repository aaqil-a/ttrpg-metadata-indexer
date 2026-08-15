import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
from indexer.types import Adventure
from urllib.parse import quote

def safe_index_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '-', name)


def create_adventure_files(
    adventures: List[Adventure],
    indices: List[str],
    root: Path,
    max_entries: int = 50,
    hierarchical: bool = True,
) -> None:
    adventure_dir = root / "adventures"
    adventure_dir.mkdir(parents=True, exist_ok=True)

    for adventure in tqdm(adventures, desc="Writing adventures to markdown"):
        path = adventure_dir / f"{adventure.slug}.md"

        path.write_text(
            adventure_to_markdown(adventure),
            encoding="utf-8",
        )

    index_files = {}

    for index in indices:
        if not hasattr(adventures[0], index):
            print(f"Invalid attribute {index} skipped...")
            continue

        if index == "other_args":
            print(f"Indexing on other_args is not supported.")
            continue

        index_files[index] = create_indexes(
            adventures,
            root,
            index,
            lambda a: getattr(a, index),
            max_entries=max_entries,
            hierarchical=hierarchical,
        )

    if indices:
        create_master_index(index_files, root)


def create_master_index(
    index_files: Dict[str, Set[Path]],
    root: Path,
) -> None:
    lines: List[str] = []

    def build_tree(paths: Set[Path], base_dir: Path):
        root_node = {'children': {}, 'files': []}

        for p in paths:
            try:
                rel_path = p.relative_to(base_dir)
            except Exception:
                rel_path = p.relative_to(root)

            parts = list(rel_path.parts)
            node = root_node
            for part in parts[:-1]:
                node = node['children'].setdefault(part, {'children': {}, 'files': []})

            file_name = parts[-1]
            rel = quote(p.relative_to(root).as_posix())
            node['files'].append({'path': p, 'name': file_name, 'stem': p.stem, 'rel': rel})

        return root_node

    def render_node(node, indent: int, out_lines: List[str], dir_name: str | None = None):
        prefix = '  ' * indent
        if dir_name is not None:
            display = dir_name.replace('-', ' ')
            out_lines.append(f"{prefix}- {display}")
            indent += 1
            prefix = '  ' * indent

        files = node.get('files', [])
        groups: Dict[str, List[dict]] = {}
        for f in files:
            m = re.match(r"^(.*?)(?:-(\d+))?$", f['stem'])
            base = m.group(1) if m else f['stem']
            groups.setdefault(base, []).append(f)

        for base in sorted(groups.keys(), key=lambda s: s.lower()):
            members = groups[base]
            if len(members) > 1:
                out_lines.append(f"{prefix}- {base.replace('-', ' ')}")
                for mem in sorted(members, key=lambda x: x['name']):
                    out_lines.append(f"{prefix}  - [{mem['stem'].replace('-', ' ')}]({mem['rel']})")
            else:
                mem = members[0]
                out_lines.append(f"{prefix}- [{mem['stem'].replace('-', ' ')}]({mem['rel']})")

        for child_name in sorted(node['children'].keys(), key=lambda s: s.lower()):
            render_node(node['children'][child_name], indent, out_lines, child_name)

    for index, paths in index_files.items():
        if not paths:
            continue

        lines.append(f"## By {index}")
        lines.append("")

        base_dir = root / index
        tree = build_tree(paths, base_dir)
        render_node(tree, 0, lines)
        lines.append("")

    body = "\n".join(lines)

    with open(root / "index.md", "w", encoding="utf-8") as file:
        file.write(body)

def create_indexes(
    adventures: List[Adventure],
    root: Path,
    directory_name: str,
    group_by,
    max_entries: int = 50,
    hierarchical: bool = True,
) -> Set[Path]:
    adventure_dir = root / "adventures"

    index_dir = root / directory_name
    index_dir.mkdir(parents=True, exist_ok=True)

    index_files = set()
    groups: dict[str, list[Adventure]] = defaultdict(list)

    for adventure in adventures:
        value = group_by(adventure)

        if isinstance(value, (list, tuple, set)):
            group_values = value
        else:
            group_values = [value]

        for group in group_values:
            if group is None:
                group = "Other"

            groups[str(group).lower()].append(adventure)

    def build_trie(keys: List[str]):
        root = {'children': {}, 'groups': [], 'count': 0}

        for g in keys:
            tokens = g.split()
            node = root
            for t in tokens:
                node = node['children'].setdefault(t, {'children': {}, 'groups': [], 'count': 0})
            node['groups'].append(g)

        def compute_counts(node):
            total = len(node['groups'])
            for child in node['children'].values():
                total += compute_counts(child)
            node['count'] = total
            return total

        compute_counts(root)
        return root

    def assign_dirs(root):
        mapping: Dict[str, Path] = {}

        def dfs(node, dir_parts: List[str]):
            # assign groups that end at this node
            for g in node['groups']:
                # apply safe name to directory parts
                safe_parts = [safe_index_name(p) for p in dir_parts]
                mapping[g] = Path('/'.join(safe_parts))

            for token, child in node['children'].items():
                # create a directory for this token only if the subtree contains more than one group
                should_dir = child['count'] > 1
                next_parts = dir_parts + [token] if should_dir else dir_parts
                dfs(child, next_parts)

        dfs(root, [])
        return mapping

    group_keys = list(groups.keys())
    if hierarchical:
        trie_root = build_trie(group_keys)
        group_dir_map = assign_dirs(trie_root)
    else:
        # no hierarchical grouping; map every group to top-level (empty path)
        group_dir_map = {g: Path("") for g in group_keys}

    for group, group_adventures in tqdm(
        groups.items(),
        desc=f"Creating indices for {directory_name}",
    ):
        safe_name = safe_index_name(group)

        sorted_adventures = sorted(
            group_adventures,
            key=lambda a: (
                a.start_level is None,
                a.start_level or 0,
                a.title.lower(),
            ),
        )

        if max_entries is None or max_entries <= 0:
            chunks = [sorted_adventures]
        else:
            chunks = [
                sorted_adventures[i : i + max_entries]
                for i in range(0, len(sorted_adventures), max_entries)
            ]

        for idx, chunk in enumerate(chunks, start=1):
            suffix = f"-{idx}" if len(chunks) > 1 else ""
            dir_parts = group_dir_map.get(group, Path())
            target_dir = index_dir / dir_parts
            target_dir.mkdir(parents=True, exist_ok=True)

            index_path = target_dir / f"{safe_name}{suffix}.md"

            title = f"{group}"
            if len(chunks) > 1:
                title = f"{group} ({idx}/{len(chunks)})"

            lines = [
                f"# {title}",
                "",
                "| Adventure | Start Level | End Level | Environments |",
                "|---|---:|---:|---|",
            ]

            for adventure in chunk:
                adventure_file = adventure_dir / f"{adventure.slug}.md"
                rel_path = os.path.relpath(adventure_file, start=target_dir)
                rel_posix = rel_path.replace(os.path.sep, '/')

                start_level = adventure.start_level or "—"
                end_level = adventure.end_level or "—"

                lines.append(
                    f"| [{adventure.title}]({quote(rel_posix)}) "
                    f"| {start_level} "
                    f"| {end_level} "
                    f"| {', '.join(adventure.environments)} |"
                )

            index_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            index_files.add(index_path)

    return index_files

def adventure_to_markdown(adv: Adventure) -> str:
    def esc(s: str) -> str:
        if s is None:
            return ""
        return s.replace('\\', '\\\\').replace('"', '\\"')

    front: List[str] = []
    front.append("---")
    front.append(f'slug: "{esc(adv.slug)}"')
    front.append(f'title: "{esc(adv.title)}"')

    front.append("authors:")
    for a in (adv.authors or []):
        front.append(f'  - "{esc(a)}"')

    front.append("environments:")
    for e in (adv.environments or []):
        front.append(f'  - "{esc(e)}"')

    if adv.start_level is None:
        front.append("start_level: null")
    else:
        front.append(f"start_level: {adv.start_level}")

    if adv.end_level is None:
        front.append("end_level: null")
    else:
        front.append(f"end_level: {adv.end_level}")

    front.append("creatures:")
    for c in (adv.creatures or []):
        front.append(f'  - "{esc(c)}"')

    front.append(f'downloaded_from: "{esc(adv.downloaded_from)}"')
    front.append("---")

    body: List[str] = []
    body.append(f"# {adv.title}")

    if adv.environments:
        body.append("")
        body.append("| " + " | ".join(adv.environments) + " |")
        body.append("| " + " | ".join(["---"] * len(adv.environments)) + " |")

    body.append("")
    start_disp = adv.start_level if adv.start_level is not None else "—"
    end_disp = adv.end_level if adv.end_level is not None else "—"
    body.append(f"**Levels {start_disp} - {end_disp}**   ")
    body.append(f"_By {', '.join(adv.authors or [])}_")
    body.append("```")
    body.append(adv.description or "")
    body.append("```")
    body.append("")
    body.append("## Creatures")
    body.append("\n".join([f"- {creature}" for creature in adv.creatures]))
    body.append("## Additional Data")
    body.append("```")
    body.append(f"Downloaded from: {adv.downloaded_from}")
    for key, value in (adv.other_args or {}).items():
        body.append(f"{key}: {value}")
    body.append("```")

    return "\n".join(front + [""] + body) + "\n"
