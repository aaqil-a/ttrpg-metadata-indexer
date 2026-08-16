import re
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
from indexer.types import Adventure
from urllib.parse import quote

def safe_index_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '-', name)


def _level_range_label(chunk: List[Adventure]) -> str:
    def fmt(a: Adventure) -> str:
        return str(a.start_level) if a.start_level is not None else "—"

    lo = fmt(chunk[0])
    hi = fmt(chunk[-1])
    
    return f"Start level {lo if lo == hi else f"{lo} - {hi}"}"


def _title_range_label(chunk: List[Adventure]) -> str:
    def initial(a: Adventure) -> str:
        title = (a.title or "").strip()
        return title[0].upper() if title else "#"

    lo = initial(chunk[0])
    hi = initial(chunk[-1])
    return lo if lo == hi else f"{lo} - {hi}"


def chunk_range_label(chunk: List[Adventure], vary_by_level: bool) -> str:
    return _level_range_label(chunk) if vary_by_level else _title_range_label(chunk)


def create_adventure_files(
    adventures: List[Adventure],
    indices: List[str],
    root: Path,
    max_entries: int = 50,
    hierarchical: bool = True,
) -> None:
    adventure_dir = root / "adventures"
    adventure_dir.mkdir(parents=True, exist_ok=True)
    master_index_path: Path = root / "master.md"

    for adventure in tqdm(adventures, desc="Writing adventures to markdown"):
        path = adventure_dir / f"{adventure.slug}.md"

        path.write_text(
            adventure_to_markdown(adventure, adventure_dir, master_index_path),
            encoding="utf-8",
        )

    index_files: Dict[str, Path] = {}

    for index in indices:
        if not hasattr(adventures[0], index):
            print(f"Invalid attribute {index} skipped...")
            continue

        if index == "other_args":
            print(f"Indexing on other_args is not supported.")
            continue

        index_files[index] = create_indices(
            adventures,
            root,
            index,
            lambda a: getattr(a, index),
            max_entries=max_entries,
            hierarchical=hierarchical,
            master_index_path=master_index_path,
        )

    if indices:
        create_master_index(index_files, root, master_index_path)


def create_master_index(
    submaster_files: Dict[str, Path],
    root: Path,
    file_name: str | Path = "master.md",
    title: str = "Master Index"
) -> Path:
    body: List[str] = []
    body.append(f"# {title}")
    for name, path in submaster_files.items():
        body.append(f"### [By {name}]({_get_rel_path(root, path)})")

    body.append("")

    if isinstance(file_name, Path):
        file_path = file_name
    else:
        file_path = root / file_name

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(body))

    return file_path

def create_submaster_index(
    index_files: Set[Path],
    root: Path,
    path: Path,
    title: str,
    master_index_path: Path | None = None,
    chunk_labels: Dict[Path, str] | None = None,
    display_names: Dict[Path, str] | None = None,
) -> Path:
    lines: List[str] = []
    chunk_labels = chunk_labels or {}
    display_names = display_names or {}

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
        def try_int(s: str):
            try:
                return int(s)
            except Exception:
                return None

        def base_sort_key(s: str):
            v = try_int(s)
            if v is not None:
                return (0, v)
            return (1, s.lower())

        def member_sort_key(mem: dict):
            parts = mem['stem'].split('-')
            key = []
            for p in parts:
                iv = try_int(p)
                key.append(iv if iv is not None else p.lower())
            return tuple(key)

        def fallback_display(mem: dict) -> str:
            if try_int(base) is not None:
                return base
            return mem['stem'].replace('-', ' ')

        for base in sorted(groups.keys(), key=base_sort_key):
            members = groups[base]
            if len(members) > 1:
                umbrella_display = display_names.get(
                    members[0]['path'],
                    base if try_int(base) is not None else base.replace('-', ' '),
                )
                out_lines.append(f"{prefix}- {umbrella_display}")
                for mem in sorted(members, key=member_sort_key):
                    mm = re.match(r"^(.*?)(?:-(\d+))?$", mem['stem'])
                    suffix = mm.group(2) if mm else None
                    label = chunk_labels.get(mem['path'])
                    base_display = display_names.get(mem['path'], fallback_display(mem))
                    if label:
                        display = f"{base_display} ({label})"
                    elif try_int(base) is not None and suffix is not None:
                        display = f"{base_display} (part {int(suffix)})"
                    else:
                        display = base_display
                    out_lines.append(f"{prefix}  - [{display}]({mem['rel']})")
            else:
                mem = members[0]
                display = display_names.get(mem['path'], fallback_display(mem))
                out_lines.append(f"{prefix}- [{display}]({mem['rel']})")

        for child_name in sorted(node['children'].keys(), key=lambda s: s.lower()):
            render_node(node['children'][child_name], indent, out_lines, child_name)

    lines.append(f"# {title}")
    lines.append("")

    if master_index_path:
        lines.append(f"[Back to {master_index_path.stem}]({_get_rel_path(root, master_index_path)})")

    tree = build_tree(index_files, root / "indices")
    render_node(tree, 0, lines)

    body = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as file:
        file.write(body)

    return path


def create_indices(
    adventures: List[Adventure],
    root: Path,
    directory_name: str,
    group_by,
    max_entries: int = 50,
    hierarchical: bool = True,
    master_index_path: Path | None = None
) -> Path:
    adventure_dir = root / "adventures"

    index_dir = root / directory_name / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_files: Set[Path] = set()
    chunk_labels: Dict[Path, str] = {}
    display_names: Dict[Path, str] = {}
    original_names: Dict[str, str] = {}
    groups: dict[str, list[Adventure]] = defaultdict(list)

    submaster_path = root / directory_name / f"{directory_name}_index.md"

    for adventure in adventures:
        value = group_by(adventure)

        if isinstance(value, (list, tuple, set)):
            group_values = value
        else:
            group_values = [value]

        for group in group_values:
            if group is None:
                group = "Other"

            key = str(group).lower()
            groups[key].append(adventure)
            original_names.setdefault(key, str(group))

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
            for g in node['groups']:
                safe_parts = [safe_index_name(p) for p in dir_parts]
                mapping[g] = Path('/'.join(safe_parts))

            for token, child in node['children'].items():
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

        def split_into_chunks(items: List[Adventure]) -> List[List[Adventure]]:
            if not items:
                return []
            if max_entries is None or max_entries <= 0:
                return [items]
            return [
                items[i : i + max_entries]
                for i in range(0, len(items), max_entries)
            ]

        leveled = [a for a in sorted_adventures if a.start_level is not None]
        unleveled = [a for a in sorted_adventures if a.start_level is None]
        chunks = split_into_chunks(leveled) + split_into_chunks(unleveled)

        vary_by_level = len({a.start_level for a in sorted_adventures}) > 1
        group_chunks: List[tuple[Path, List[Adventure]]] = []

        for idx, chunk in enumerate(chunks, start=1):
            suffix = f"-{idx}" if len(chunks) > 1 else ""
            dir_parts = group_dir_map.get(group, Path())
            target_dir = index_dir / dir_parts
            target_dir.mkdir(parents=True, exist_ok=True)

            index_path = target_dir / f"{safe_name}{suffix}.md"

            display_name = original_names.get(group, group)
            title = f"{display_name}"
            if len(chunks) > 1:
                title = f"{display_name} ({idx}/{len(chunks)})"

            lines = [
                f"# {title}",
                "",
                f"[Back to {submaster_path.stem}]({_get_rel_path(target_dir, submaster_path)})",
                "| Adventure | Start Level | End Level | Environments | Downloaded From |",
                "|---|---:|---:|---|---|",
            ]

            for adventure in chunk:
                adventure_file = adventure_dir / f"{adventure.slug}.md"

                start_level = adventure.start_level or "—"
                end_level = adventure.end_level or "—"

                lines.append(
                    f"| [{adventure.title}]({_get_rel_path(target_dir, adventure_file)}) "
                    f"| {start_level} "
                    f"| {end_level} "
                    f"| {', '.join(adventure.environments)} "
                    f"| {adventure.downloaded_from} |"
                )

            index_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            index_files.add(index_path)
            display_names[index_path] = display_name
            if len(chunks) > 1:
                group_chunks.append((index_path, chunk))

        primary = {p: chunk_range_label(c, vary_by_level) for p, c in group_chunks}
        duplicated = {lbl for lbl, n in Counter(primary.values()).items() if n > 1}
        for path, chunk in group_chunks:
            label = primary[path]
            if vary_by_level and label in duplicated:
                label = f"{label} ({_title_range_label(chunk)})"
            chunk_labels[path] = label

    index_master = create_submaster_index(
        index_files,
        root / directory_name,
        submaster_path,
        directory_name,
        master_index_path=master_index_path,
        chunk_labels=chunk_labels,
        display_names=display_names,
    )

    return index_master

def adventure_to_markdown(
    adv: Adventure,
    root: Path | None = None,
    master_index_path: Path | None = None,
) -> str:
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
    body.append("")
    if root and master_index_path:
        body.append(f"[Back to {master_index_path.stem}]({_get_rel_path(root, master_index_path)})")

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


def _get_rel_path(root: Path, file: Path) -> str:
    rel_path = os.path.relpath(file, start=root)
    rel_posix = rel_path.replace(os.path.sep, '/')
    return quote(rel_posix)
