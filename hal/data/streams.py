import os
from typing import Dict

from streaming.base.stream import Stream

from hal.local_paths import REPO_DIR

AWS_BUCKET = os.getenv("AWS_BUCKET")
assert AWS_BUCKET is not None, "AWS_BUCKET environment variable is not set"


class StreamRegistry:
    STREAMS: Dict[str, Stream] = {}

    @classmethod
    def register(cls, name: str, streams: Stream) -> None:
        if name in cls.STREAMS:
            raise ValueError(f"Stream {name} already registered")
        cls.STREAMS[name] = streams

    @classmethod
    def get(cls, name: str) -> Stream:
        if name in cls.STREAMS:
            return cls.STREAMS[name]
        raise ValueError(f"Stream {name} not registered")


### Ranked


RankedPlatinumStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/platinum",
    local=f"{REPO_DIR}/data/ranked/platinum",
    proportion=1.0,
    keep_zip=True,
)


RankedDiamondStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/diamond",
    local=f"{REPO_DIR}/data/ranked/diamond",
    proportion=1.0,
    keep_zip=True,
)


RankedMasterStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/master",
    local=f"{REPO_DIR}/data/ranked/master",
    proportion=1.0,
    keep_zip=True,
)


StreamRegistry.register("ranked-platinum", RankedPlatinumStream)
StreamRegistry.register("ranked-diamond", RankedDiamondStream)
StreamRegistry.register("ranked-master", RankedMasterStream)


### Top players


AkloStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Aklo",
    local=f"{REPO_DIR}/data/top_players/Aklo",
    proportion=1.0,
    keep_zip=True,
)

AmsaStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/aMSa",
    local=f"{REPO_DIR}/data/top_players/aMSa",
    proportion=1.0,
    keep_zip=True,
)

CodyStream = Stream(
    # remote=f"s3://{AWS_BUCKET}/hal/top_players/Cody",
    local=f"{REPO_DIR}/data/top_players/Cody",
    proportion=1.0,
    keep_zip=True,
)

FranzStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Franz",
    local=f"{REPO_DIR}/data/top_players/Franz",
    proportion=1.0,
    keep_zip=True,
)

FrenzyStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Frenzy",
    local=f"{REPO_DIR}/data/top_players/Frenzy",
    proportion=1.0,
    keep_zip=True,
)

KodorinStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Kodorin",
    local=f"{REPO_DIR}/data/top_players/Kodorin",
    proportion=1.0,
    keep_zip=True,
)

Mang0Stream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/mang0",
    local=f"{REPO_DIR}/data/top_players/mang0",
    proportion=2.0,
    keep_zip=True,
)

MorsecodeStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Morsecode",
    local=f"{REPO_DIR}/data/top_players/Morsecode",
    proportion=1.0,
    keep_zip=True,
)

SFATStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/SFAT",
    local=f"{REPO_DIR}/data/top_players/SFAT",
    proportion=1.0,
    keep_zip=True,
)

SolobattleStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Solobattle",
    local=f"{REPO_DIR}/data/top_players/Solobattle",
    proportion=1.0,
    keep_zip=True,
)

YCZStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/YCZ",
    local=f"{REPO_DIR}/data/top_players/YCZ",
    proportion=1.0,
    keep_zip=True,
)

StreamRegistry.register("cody", CodyStream)
StreamRegistry.register("mang0", Mang0Stream)
