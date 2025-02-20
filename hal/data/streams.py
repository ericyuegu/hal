import os
from typing import Dict

from streaming.base.stream import Stream

from hal.emulator_paths import REMOTE_REPO_DIR

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
    local=f"{REMOTE_REPO_DIR}/data/ranked/platinum",
    proportion=1.0,
)


RankedDiamondStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/diamond",
    local=f"{REMOTE_REPO_DIR}/data/ranked/diamond",
    proportion=1.0,
)


RankedMasterStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/master",
    local=f"{REMOTE_REPO_DIR}/data/ranked/master",
    proportion=1.0,
)


StreamRegistry.register("ranked-platinum", RankedPlatinumStream)
StreamRegistry.register("ranked-diamond", RankedDiamondStream)
StreamRegistry.register("ranked-master", RankedMasterStream)


### Top players


AkloStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Aklo",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Aklo",
    proportion=1.0,
)

AmsaStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/aMSa",
    local=f"{REMOTE_REPO_DIR}/data/top_players/aMSa",
    proportion=1.0,
)

CodyStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Cody",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Cody",
    proportion=1.0,
)

FranzStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Franz",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Franz",
    proportion=1.0,
)

FrenzyStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Frenzy",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Frenzy",
    proportion=1.0,
)

KodorinStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Kodorin",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Kodorin",
    proportion=1.0,
)

Mang0Stream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/mang0",
    local=f"{REMOTE_REPO_DIR}/data/top_players/mang0",
    proportion=2.0,
)

MorsecodeStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Morsecode",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Morsecode",
    proportion=1.0,
)

SFATStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/SFAT",
    local=f"{REMOTE_REPO_DIR}/data/top_players/SFAT",
    proportion=1.0,
)

SolobattleStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Solobattle",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Solobattle",
    proportion=1.0,
)

YCZStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/YCZ",
    local=f"{REMOTE_REPO_DIR}/data/top_players/YCZ",
    proportion=1.0,
)

StreamRegistry.register("cody", CodyStream)
StreamRegistry.register("mang0", Mang0Stream)
