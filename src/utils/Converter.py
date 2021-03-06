import numpy as np
from utils.typing import assert_type
from tensorflow.keras.utils import to_categorical  # type: ignore
from utils.Window import Window




class Converter():

    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def sonar_convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=self.n_classes,
        )

        return np.array(sensor_arrays), np.array(activity_vectors)


