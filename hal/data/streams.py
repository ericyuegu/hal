import os
from collections.abc import Callable

from streaming.base.stream import Stream

from hal.local_paths import REPO_DIR


def _aws_bucket() -> str:
    aws_bucket = os.getenv("AWS_BUCKET")
    if aws_bucket is None:
        raise RuntimeError("AWS_BUCKET environment variable is required to construct remote data streams")
    return aws_bucket


def _remote(path: str) -> str:
    return f"s3://{_aws_bucket()}/hal/{path}"


def _stream(path: str, *, proportion: float = 1.0) -> Stream:
    return Stream(
        remote=_remote(path),
        local=f"{REPO_DIR}/data/{path}",
        proportion=proportion,
        keep_zip=True,
    )


class StreamRegistry:
    STREAMS: dict[str, Stream] = {}
    STREAM_FACTORIES: dict[str, Callable[[], Stream]] = {}

    @classmethod
    def register(cls, name: str, stream: Stream | Callable[[], Stream]) -> None:
        if name in cls.STREAMS or name in cls.STREAM_FACTORIES:
            raise ValueError(f"Stream {name} already registered")
        if isinstance(stream, Stream):
            cls.STREAMS[name] = stream
        else:
            cls.STREAM_FACTORIES[name] = stream

    @classmethod
    def get(cls, name: str) -> Stream:
        if name in cls.STREAMS:
            return cls.STREAMS[name]
        if name in cls.STREAM_FACTORIES:
            stream = cls.STREAM_FACTORIES[name]()
            cls.STREAMS[name] = stream
            return stream
        raise ValueError(f"Stream {name} not registered")


StreamRegistry.register("ranked-platinum", lambda: _stream("ranked/platinum"))
StreamRegistry.register("ranked-diamond", lambda: _stream("ranked/diamond"))
StreamRegistry.register("ranked-master", lambda: _stream("ranked/master"))
StreamRegistry.register("cody", lambda: _stream("top_players/Cody"))
StreamRegistry.register("mang0", lambda: _stream("top_players/mang0", proportion=2.0))
