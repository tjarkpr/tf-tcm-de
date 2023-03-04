import tensorflow as tf

from collections.abc import Callable


class PermutationDataset(tf.data.Dataset):
    """
    Dataset which contains the logic for initializing
    permutation of all input textual sequences in the
    base dataset.

    Attributes
    ----------
    separate: Callable[[str], list[str]]
        Separation function for textual sequences.
    convert: Callable[[str], list]
        Conversion function for textual sequences into vector representation.
    
    Methods
    -------
    from_dataset(base_dataset: tf.data.Dataset)
        Creates a permutation tf.data.Dataset from the base dataset 
        based on the provided separation and conversion functions.
    """

    __seperate: Callable[[str], list[str]] = None
    __convert: Callable[[str], list] = None

    def __init__(
            self, 
            separate: Callable[[str], list[str]], 
            convert: Callable[[str], list]
            ) -> None:
        """
        Dataset which contains the logic for initializing
        permutation of all input textual sequences in the
        base dataset.

        Parameters
        ----------
        separate: Callable[[str], list[str]]
            Separation function for textual sequences.
        convert: Callable[[str], list]
            Conversion function for textual sequences into vector representation.
        """

        self.__seperate = separate
        self.__convert = convert

    def from_dataset(self, base_dataset: tf.data.Dataset) -> None:
        """
        Creates a permutation tf.data.Dataset from the base dataset 
        based on the provided separation and conversion functions.

        Parameters
        ----------
        base_dataset: tf.data.Dataset
            Base dataset which needs to be explained.
        """
        
        seperated_dataset = base_dataset.map(self.__seperate, num_parallel_calls=tf.data.AUTOTUNE)
        converted_dataset = base_dataset.map(self.__convert, num_parallel_calls=tf.data.AUTOTUNE)