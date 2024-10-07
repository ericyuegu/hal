from typing import Generator
from typing import Iterable
from typing import Tuple
from typing import TypeVar

T = TypeVar("T")


def generate_chunks(iterable: Iterable[T], chunk_size: int) -> Generator[Tuple[T, ...], None, None]:
    """Yield successive n-sized chunks from any iterable"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield tuple(chunk)
            chunk = []
    if len(chunk) > 0:
        yield tuple(chunk)
