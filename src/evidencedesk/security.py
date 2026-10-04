import re

INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(previous|prior|above)",
    r"forget (everything|all) (above|before)",
    r"reveal your (instructions|prompt|system)",
    r"you are now",
    r"new instructions?:",
    r"</?(system|assistant|user)>",
)

_COMPILED_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS)

FENCE_OPEN = "<<<UNTRUSTED_DOCUMENT"
FENCE_CLOSE = "UNTRUSTED_DOCUMENT>>>"


def scan_for_injection(text: str) -> list[str]:
    """Liefert die Muster, die auf einen Prompt-Injection-Versuch hindeuten.

    Erkennen heisst nicht blockieren: Treffer werden protokolliert, damit Auffaelligkeiten
    sichtbar sind, ohne legitime Dokumente zu verwerfen.
    """
    return [pattern for pattern in INJECTION_PATTERNS if re.search(pattern, text, re.IGNORECASE)]


def wrap_untrusted(chunk_id: str, text: str) -> str:
    """Rahmt abgerufenen Text als nicht vertrauenswuerdige Daten ein."""
    return f"{FENCE_OPEN} id={chunk_id}\n{text}\n{FENCE_CLOSE}"
