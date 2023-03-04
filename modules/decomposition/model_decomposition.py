import tensorflow as tf

from modules.datasets.effect_dataset import EffectDataset
from modules.models.local_model import LocalModel


class ModelDecomposition():
    __local_models: list[LocalModel] = []
    __effect_dataset: EffectDataset
    __permutation_dataset: tf.data.Dataset

    def __init__(
            self, 
            permutation_dataset: tf.data.Dataset, 
            effect_dataset: EffectDataset
            ) -> None:
        self.__permutation_dataset = permutation_dataset
        self.__effect_dataset = effect_dataset