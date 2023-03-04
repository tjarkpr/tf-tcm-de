import tensorflow as tf

from modules.datasets.bow_cosine_dataset import BoWCosineDataset
from modules.explaination.effect_explaination import EffectExplaination


class EffectExplainer():
    __meta_dataset: BoWCosineDataset
    __significant_effect_dataset: tf.data.Dataset

    def __init__(self, significant_effect_dataset: tf.data.Dataset) -> None:
        self.__significant_effect_dataset = significant_effect_dataset
    
    def explain(self) -> EffectExplaination:
        return EffectExplaination(self.__meta_dataset)