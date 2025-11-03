from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import debugpy

config = _config.get_config("pi0_aloha")

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "state": np.random.rand(14),
    "prompt": "do something",
    'images':{
        'cam_high': np.random.randint(256, size=(3,224, 224), dtype=np.uint8),
        #'cam_low': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        'cam_left_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        'cam_right_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
    }
}
print("start inference")
action_chunk = policy.infer(example)["actions"]
print(action_chunk)
