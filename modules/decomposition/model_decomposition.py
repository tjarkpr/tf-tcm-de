from modules.datasets.effect_dataset import EffectDataset
from modules.datasets.permutation_dataset import PermutationDataset
from modules.models.local_model import LocalModel


class ModelDecomposition():
    __local_models: list[LocalModel] = []
    __effect_dataset: EffectDataset = None
    __permutation_dataset: PermutationDataset = None

    def __init__(
            self, 
            permutation_dataset: PermutationDataset, 
            effect_dataset: EffectDataset
            ) -> None:
        self.__permutation_dataset = permutation_dataset
        self.__effect_dataset = effect_dataset