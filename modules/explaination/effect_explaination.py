from modules.datasets.bow_cosine_dataset import BoWCosineDataset
from modules.models.decision_model import DecisionModel


class EffectExplaination():
    __meta_dataset: BoWCosineDataset
    __decision_model: DecisionModel

    def __init__(self, meta_dataset: BoWCosineDataset) -> None:
        self.__meta_dataset = meta_dataset