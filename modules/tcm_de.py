from modules.explaination.effect_explainer import EffectExplainer
from modules.decomposition.model_decomposer import ModelDecomposer


class TCMDE():
    __effect_explainer: EffectExplainer = None
    __model_decomposer: ModelDecomposer = None

    def __init__(self) -> None:
        pass