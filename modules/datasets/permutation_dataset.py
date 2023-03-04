import tensorflow as tf
import itertools as it

from collections.abc import Callable


class PermutationDataset():
    """
    Dataset which contains the logic for initializing
    permutation of all input textual sequences in the
    base text dataset.
    
    Methods
    -------
    from_dataset(base_text_dataset: tf.data.Dataset, separate: Callable[[str], list[str]], convert: Callable[[str], list])
        Creates a permutation tf.data.Dataset from the base text dataset 
        based on the provided separation and conversion functions.
    """

    @staticmethod
    def from_dataset(
            base_text_dataset: tf.data.Dataset,
            separate: Callable[[str], list[str]],
            convert: Callable[[str], list]
        ) -> tf.data.Dataset:
        """
        Creates a permutation tf.data.Dataset from the base dataset 
        based on the provided separation and conversion functions.

        Parameters
        ----------
        base_text_dataset: tf.data.Dataset
            Base text dataset which needs to be explained.
        separate: Callable[[str], list[str]]
            Separation function for textual sequences.
        convert: Callable[[str], list]
            Conversion function for textual sequences into vector representation.
        """
        
        separated_dataset = base_text_dataset.map(separate, num_parallel_calls=tf.data.AUTOTUNE)
        converted_dataset = base_text_dataset.map(convert, num_parallel_calls=tf.data.AUTOTUNE)
        permutation_dataset = tf.data.Dataset.zip((separated_dataset, converted_dataset))

        def __permute_entry__(separated_entry: tf.Tensor, converted_entry: tf.Tensor):
            return separated_entry, converted_entry, it.product(range(2), repeat=len(separated_entry))

        return permutation_dataset.map(__permute_entry__, num_parallel_calls=tf.data.AUTOTUNE)