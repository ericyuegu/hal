import os
from typing import Dict
from typing import Sequence

from streaming.base.stream import Stream

from hal.emulator_paths import REMOTE_REPO_DIR

AWS_BUCKET = os.getenv("AWS_BUCKET")
assert AWS_BUCKET is not None, "AWS_BUCKET environment variable is not set"


class StreamRegistry:
    STREAMS: Dict[str, Sequence[Stream]] = {}

    @classmethod
    def register(cls, name: str, streams: Sequence[Stream]) -> None:
        if name in cls.STREAMS:
            raise ValueError(f"Stream {name} already registered")
        cls.STREAMS[name] = streams

    @classmethod
    def get(cls, name: str) -> Sequence[Stream]:
        if name in cls.STREAMS:
            return cls.STREAMS[name]
        raise ValueError(f"Stream {name} not registered")


### Ranked


RankedPlatinumStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/platinum/train",
    local=f"{REMOTE_REPO_DIR}/data/ranked/platinum/train",
    proportion=1.0,
)


RankedDiamondStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/diamond/train",
    local=f"{REMOTE_REPO_DIR}/data/ranked/diamond/train",
    proportion=1.0,
)


RankedMasterStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/ranked/master/train",
    local=f"{REMOTE_REPO_DIR}/data/ranked/master/train",
    proportion=1.0,
)


StreamRegistry.register("ranked-platinum", [RankedPlatinumStream])
StreamRegistry.register("ranked-diamond", [RankedDiamondStream])
StreamRegistry.register("ranked-master", [RankedMasterStream])


### Top players


AkloStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Aklo/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Aklo/train",
    proportion=1.0,
)

AmsaStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/aMSa/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/aMSa/train",
    proportion=1.0,
)

CodyStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Cody/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Cody/train",
    proportion=1.0,
)

FranzStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Franz/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Franz/train",
    proportion=1.0,
)

FrenzyStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Frenzy/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Frenzy/train",
    proportion=1.0,
)

KodorinStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Kodorin/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Kodorin/train",
    proportion=1.0,
)

Mang0Stream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/mang0/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/mang0/train",
    proportion=2.0,
)

MorsecodeStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Morsecode/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Morsecode/train",
    proportion=1.0,
)

SFATStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/SFAT/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/SFAT/train",
    proportion=1.0,
)

SolobattleStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Solobattle/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Solobattle/train",
    proportion=1.0,
)

YCZStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/YCZ/train",
    local=f"{REMOTE_REPO_DIR}/data/top_players/YCZ/train",
    proportion=1.0,
)

StreamRegistry.register("cody", [CodyStream])
StreamRegistry.register("mang0", [Mang0Stream])
StreamRegistry.register("mang0-master", [Mang0Stream, RankedMasterStream])
StreamRegistry.register("cody-mang0", [CodyStream, Mang0Stream])
StreamRegistry.register("cody-mang0-master", [CodyStream, Mang0Stream, RankedMasterStream])
StreamRegistry.register("cody-master", [CodyStream, RankedMasterStream])
StreamRegistry.register(
    "cody-mang0-master-diamond", [CodyStream, Mang0Stream, RankedMasterStream, RankedDiamondStream]
)


### Validation


CodyValidationStream = Stream(
    remote=f"s3://{AWS_BUCKET}/hal/top_players/Cody/val",
    local=f"{REMOTE_REPO_DIR}/data/top_players/Cody/val",
    proportion=1.0,
)


StreamRegistry.register("cody-val", [CodyValidationStream])
