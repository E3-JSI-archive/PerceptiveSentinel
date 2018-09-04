import json
from datetime import date

import numpy as np


# Taken from
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        # Numpy stuff
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # General numpy object
        if isinstance(obj, np.generic):
            return obj.item()
        # Date format
        if isinstance(obj, date):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def encode(data, **kwargs):
    return json.dumps(data, cls=NumpyEncoder, **kwargs)
