from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_DESC_LENGTH = 1500

@dataclass
class Adventure:
    slug: str = field(metadata={"no_index": True})
    title: str = field(metadata={"no_index": True})
    description: str = field(metadata={"no_index": True})
    authors: List[str]
    environments: List[str]
    start_level: Optional[int]
    end_level: Optional[int]
    creatures: List[str]
    downloaded_from: str
    other_args: Dict[str, Any] = field(metadata={"no_index": True})
