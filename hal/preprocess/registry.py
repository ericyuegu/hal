from hal.preprocess.input_config import InputConfig
from hal.preprocess.postprocess_config import PostprocessConfig
from hal.preprocess.target_config import TargetConfig


class InputConfigRegistry:
    CONFIGS: dict[str, InputConfig] = {}

    @classmethod
    def get(cls, name: str) -> InputConfig:
        if name in cls.CONFIGS:
            return cls.CONFIGS[name]
        raise NotImplementedError(f"Config {name} not found. Valid configs: {sorted(cls.CONFIGS.keys())}.")

    @classmethod
    def register(cls, name: str, config: InputConfig) -> None:
        cls.CONFIGS[name] = config


class TargetConfigRegistry:
    CONFIGS: dict[str, TargetConfig] = {}

    @classmethod
    def get(cls, name: str) -> TargetConfig:
        if name in cls.CONFIGS:
            return cls.CONFIGS[name]
        raise NotImplementedError(f"Config {name} not found. Valid configs: {sorted(cls.CONFIGS.keys())}.")

    @classmethod
    def register(cls, name: str, config: TargetConfig) -> None:
        cls.CONFIGS[name] = config


class PostprocessConfigRegistry:
    CONFIGS: dict[str, PostprocessConfig] = {}

    @classmethod
    def get(cls, name: str) -> PostprocessConfig:
        if name in cls.CONFIGS:
            return cls.CONFIGS[name]
        raise NotImplementedError(f"Config {name} not found. Valid configs: {sorted(cls.CONFIGS.keys())}.")

    @classmethod
    def register(cls, name: str, config: PostprocessConfig) -> None:
        cls.CONFIGS[name] = config
