import attr


@attr.s(auto_attribs=True, frozen=True)
class LRStage:
    # Fraction of training progress where this stage ends; must be between 0-1
    end: float
    # Scalar multiple of initial LR to be reached by end of stage
    scalar: float
