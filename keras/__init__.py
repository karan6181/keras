from __future__ import absolute_import

from . import utils
from . import activations
from . import applications
from . import backend
from . import datasets
from . import engine
from . import layers
from . import preprocessing
from . import wrappers
from . import callbacks
from . import constraints
from . import initializers
from . import metrics
from . import models
from . import losses
from . import optimizers
from . import regularizers

# Also importable from root
from .layers import Input
from .models import Model
from .models import Sequential

import warnings
warnings.warn("MXNet support in Keras is going to be discontinued and v2.2.4.3 is the last "
              "release as multi-backend Keras has been discontinued . It is recommended to "
              "consider switching to MXNet Gluon. More information can be found here: "
              "https://github.com/awslabs/keras-apache-mxnet", DeprecationWarning)

__version__ = '2.2.4.3'
