import string
from sklearn.preprocessing import OneHotEncoder
import numpy as np
valid = np.array(list(string.ascii_letters) + list("""@,.!?’' \"""")).reshape(-1,1)
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(valid)