from collections.abc import Generator
from collections.abc import Iterable


def generate_chunks[T](iterable: Iterable[T], chunk_size: int) -> Generator[tuple[T, ...]]:
    """Yield successive n-sized chunks from any iterable"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield tuple(chunk)
            chunk = []
    if len(chunk) > 0:
        yield tuple(chunk)
