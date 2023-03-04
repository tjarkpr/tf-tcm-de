from collections.abc import Callable
from modules.explaination.effect_explainer import EffectExplainer
from modules.decomposition.model_decomposer import ModelDecomposer


class TCMDE():
    __effect_explainer: EffectExplainer
    __model_decomposer: ModelDecomposer
    __seperate: Callable[[str], list[str]]
    __convert: Callable[[str], list]

    def __init__(
            self,
            separate: Callable[[str], list[str]], 
            convert: Callable[[str], list]
            ) -> None:
        self.__seperate = separate
        self.__convert = convert