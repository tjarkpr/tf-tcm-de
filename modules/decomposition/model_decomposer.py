import tensorflow as tf

from collections.abc import Callable
from modules.datasets.effect_dataset import EffectDataset
from modules.datasets.permutation_dataset import PermutationDataset
from modules.decomposition.model_decomposition import ModelDecomposition


class ModelDecomposer():
    __base_dataset: tf.data.Dataset
    __base_model: tf.keras.Model
    __effect_dataset: EffectDataset
    __permutation_dataset: tf.data.Dataset

    def __init__(
            self, 
            base_dataset: tf.data.Dataset, 
            base_model: tf.keras.Model,
            separate: Callable[[str], list[str]], 
            convert: Callable[[str], list]
            ) -> None:
        self.__base_dataset = base_dataset
        self.__base_model = base_model
        self.__permutation_dataset = PermutationDataset.from_dataset(
            base_dataset,
            separate,
            convert
            )
    
    def decompose(self) -> ModelDecomposition:
        return ModelDecomposition(self.__permutation_dataset, self.__effect_dataset)