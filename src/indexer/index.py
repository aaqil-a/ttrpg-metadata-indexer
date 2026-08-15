import re
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

        index_files[index] = create_indexes(
            adventures,
            root,
            index,
            lambda a: getattr(a, index),
        )

    if indices:
        create_master_index(index_files, root)


def create_master_index(
    index_files: Dict[str, Set[Path]],
    root: Path,
) -> None:
    lines: List[str] = []

    for index, paths in index_files.items():
        if not paths:
            continue

        stems = [p.stem for p in paths]

        def is_integer_string(s: str) -> bool:
            return s.lstrip("+-").isdigit()

        numeric_count = sum(1 for s in stems if is_integer_string(s))

        if numeric_count >= (len(stems) / 2):
            def sort_key(p: Path):
                s = p.stem
                if is_integer_string(s):
                    return (0, int(s))
                return (1, s.lower())

            sorted_paths = sorted(paths, key=sort_key)
        else:
            sorted_paths = sorted(paths, key=lambda p: p.stem.lower())

        lines.append(f"## By {index}")
        lines.append("")

        cols = 5
        lines.append("| " + " | ".join([""] * cols) + " |")
        lines.append("|" + "|".join(["---"] * cols) + "|")

        row_cells: List[str] = []

        for path in sorted_paths:
            rel = quote(path.relative_to(root).as_posix())
            row_cells.append(f"[{path.stem}]({rel})")

            if len(row_cells) == cols:
                lines.append("| " + " | ".join(row_cells) + " |")
                row_cells = []

        if row_cells:
            row_cells += [""] * (cols - len(row_cells))
            lines.append("| " + " | ".join(row_cells) + " |")

        lines.append("")

    body = "\n".join(lines)

    with open(root / "index.md", "w", encoding="utf-8") as file:
        file.write(body)

def create_indexes(
    adventures: List[Adventure],
    root: Path,
    directory_name: str,
    group_by,
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

            groups[str(group)].append(adventure)

    for group, group_adventures in tqdm(
        groups.items(),
        desc=f"Creating indices for {directory_name}",
    ):
        index_path = index_dir / f"{safe_index_name(group)}.md"

        lines = [
            f"# {group}",
            "",
            "| Adventure | Start Level | End Level | Environments |",
            "|---|---:|---:|---|",
        ]

        for adventure in sorted(
            group_adventures,
            key=lambda a: (
                a.start_level is None,
                a.start_level or 0,
                a.title.lower(),
            ),
        ):
            adventure_path = (
                Path("..")
                / adventure_dir.name
                / f"{adventure.slug}.md"
            )

            start_level = adventure.start_level or "—"
            end_level = adventure.end_level or "—"

            lines.append(
                f"| [{adventure.title}]({adventure_path.as_posix()}) "
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
    body.append("## Additional Data")
    body.append("```")
    body.append(f"Downloaded from: {adv.downloaded_from}")
    for key, value in (adv.other_args or {}).items():
        body.append(f"{key}: {value}")
    body.append("```")

    return "\n".join(front + [""] + body) + "\n"
