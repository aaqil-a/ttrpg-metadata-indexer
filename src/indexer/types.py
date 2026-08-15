from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Adventure:
    slug: str
    title: str
    description: str
    authors: List[str]
    environments: List[str]
    start_level: Optional[int]
    end_level: Optional[int]
    creatures: List[str]
    downloaded_from: str
    other_args: Dict[str, Any]
