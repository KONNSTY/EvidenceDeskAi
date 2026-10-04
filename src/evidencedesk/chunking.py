from dataclasses import dataclass

_APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    token_count: int


def approximate_tokens(text: str) -> int:
    """Grobe Token-Schaetzung ohne Tokenizer-Abhaengigkeit."""
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def chunk_text(text: str, *, size: int, overlap: int) -> list[Chunk]:
    """Zerlegt Text in ueberlappende, wortgrenzen-treue Chunks.

    ``size`` und ``overlap`` sind in Zeichen. Der Ueberlapp greift auf ganze Woerter zurueck,
    damit kein Wort zerschnitten wird.
    """
    if size <= 0:
        raise ValueError("size muss positiv sein")
    if not 0 <= overlap < size:
        raise ValueError("overlap muss 0 <= overlap < size erfuellen")

    words = " ".join(text.split()).split(" ")
    if words == [""]:
        return []

    chunks: list[Chunk] = []
    start = 0
    total = len(words)
    while start < total:
        end = start
        length = 0
        while end < total:
            addition = len(words[end]) + (1 if end > start else 0)
            if end > start and length + addition > size:
                break
            length += addition
            end += 1

        content = " ".join(words[start:end])
        chunks.append(
            Chunk(ordinal=len(chunks), content=content, token_count=approximate_tokens(content))
        )
        if end >= total:
            break

        back = end
        overlap_length = 0
        while back > start + 1 and overlap_length + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            overlap_length += len(words[back]) + 1
        start = max(back, start + 1)

    return chunks
