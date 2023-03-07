import os
import pathlib

import gin

from .models import AudioAutoEncoder

gin.add_config_file_search_path(
    os.path.join(os.path.dirname(__file__), "configs"))
