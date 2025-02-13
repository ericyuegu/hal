import os

from streaming.base.stream import Stream
from streaming.base.stream import streams_registry

AWS_BUCKET = os.getenv("AWS_BUCKET")
assert AWS_BUCKET is not None, "AWS_BUCKET environment variable is not set"


# Ranked
class Ranked1Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-1"
    local = "/tmp/hal/ranked-1"
    proportion = 1.0


class Ranked2Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-2"
    local = "/tmp/hal/ranked-2"
    proportion = 1.0


class Ranked3Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-3"
    local = "/tmp/hal/ranked-3"
    proportion = 1.0


class Ranked4Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-4"
    local = "/tmp/hal/ranked-4"
    proportion = 1.0


class Ranked5Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-5"
    local = "/tmp/hal/ranked-5"
    proportion = 1.0


class Ranked6Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/ranked-6"
    local = "/tmp/hal/ranked-6"
    proportion = 1.0


streams_registry.register("ranked-1", func=Ranked1Stream)
streams_registry.register("ranked-2", func=Ranked2Stream)
streams_registry.register("ranked-3", func=Ranked3Stream)
streams_registry.register("ranked-4", func=Ranked4Stream)
streams_registry.register("ranked-5", func=Ranked5Stream)
streams_registry.register("ranked-6", func=Ranked6Stream)


# Top players


class AkloStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Aklo"
    local = "/tmp/hal/aklo"
    proportion = 1.0


class AmsaStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/aMSa"
    local = "/tmp/hal/aMSa"
    proportion = 1.0


class CodyStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Cody"
    local = "/tmp/hal/cody"
    proportion = 1.0


class FranzStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Franz"
    local = "/tmp/hal/franz"
    proportion = 0.5


class FrenzyStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Frenzy"
    local = "/tmp/hal/frenzy"
    proportion = 1.0


class KodorinStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Kodorin"
    local = "/tmp/hal/kodorin"
    proportion = 1.0


class Mang0Stream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/mang0"
    local = "/tmp/hal/mang0"
    proportion = 1.0


class MorsecodeStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Morsecode"
    local = "/tmp/hal/morsecode"
    proportion = 1.0


class SFATStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/SFAT"
    local = "/tmp/hal/sfat"
    proportion = 1.0


class SolobattleStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/Solobattle"
    local = "/tmp/hal/solobattle"
    proportion = 1.0


class YCZStream(Stream):
    remote = f"s3://{AWS_BUCKET}/hal/top_players/YCZ"
    local = "/tmp/hal/ycz"
    proportion = 0.5


streams_registry.register("aklo", func=AkloStream)
streams_registry.register("amsa", func=AmsaStream)
streams_registry.register("cody", func=CodyStream)
streams_registry.register("frenzy", func=FrenzyStream)
streams_registry.register("kodorin", func=KodorinStream)
streams_registry.register("franz", func=FranzStream)
streams_registry.register("mang0", func=Mang0Stream)
streams_registry.register("morsecode", func=MorsecodeStream)
streams_registry.register("sfat", func=SFATStream)
streams_registry.register("solobattle", func=SolobattleStream)
streams_registry.register("ycz", func=YCZStream)
