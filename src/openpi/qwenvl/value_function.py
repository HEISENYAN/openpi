import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
#from openpi.qwen_vl.qwen_eval import qwen_eval
from transformers import AutoProcessor
from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer
from PIL import Image
from torchvision import transforms
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
MIN_VALUE = -1.0
MAX_VALUE = 0.0
BINS = 201
IMAGE_SIZE = (224,224)
def load_model_and_processor(model_name_or_path, attn_implementation=None):
    """Load model and processor from checkpoint."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.config.use_cache = True
    value_tokenizer = ValueTokenizer(
            llm_path=model_name_or_path,  # Use same model path to ensure consistency
            bins=BINS,
            min_value=MIN_VALUE,
            max_value=MAX_VALUE,
    )
    return model, processor, value_tokenizer

class ValueFunction(nn.Module):
    def __init__(self, config):
        #model_path = "/project/peilab/junhao/Value_Function/qwen-vl-finetune/output/gpus_8/checkpoint-3000"
        super().__init__()
        self.model, self.processor, self.value_tokenizer = load_model_and_processor(config.value_function_path)
        self.dataset = lerobot_dataset.LeRobotDataset(config.data.repo_id)
    def get_value(self, obs):
        return self.model.get_value(obs)
    def forward(self, obs_batch):
        pass
    def preprocess_obs(self, obs_batch):
        if obs_batch[0].dtype == torch.float32:
            obs_batch = []
        if isinstance(obs, torch.Tensor):
            obs = obs.to(self.device)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        return obs_batch
    @torch.inference_mode()
    def predict_value(self, obs_batch, episode_index):
        device = obs_batch.device
        to_pil = transforms.ToPILImage()
        obs_batch = [to_pil(obs.detach().permute(2, 1, 0)) for obs in obs_batch]
        instructions = [self.dataset[index.item()]['task'] for index in episode_index]
        messages = process_obs(obs_batch, instructions)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Add generation prompt to get model to generate
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 1,  # We only need one value token
            "do_sample": False,   # Greedy decoding for deterministic prediction
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        generated_outputs = self.model.generate(
            **inputs,
            **generation_config,
            return_dict_in_generate=True,
            output_logits=True
        )
        generated_token_ids = [sequence[len(inputs['input_ids'][0]):] for sequence in generated_outputs.sequences]
        generated_token_id = [token_id.item() if len(token_id) > 0 else None for token_id in generated_token_ids]
        predicted_values = [decode_value_token(self.value_tokenizer, token_id) for token_id in generated_token_id if token_id and token_id in self.value_tokenizer.extra_id_token_ids]
        return predicted_values

def process_obs(obs_batch, instructions):
    messages = []
    for obs,instruction in zip(obs_batch, instructions):
        message_template = [{
            'role' : 'user',
            'content' : [
                {
                    'type' : 'text',
                    'text' : f"""You are estimating task progress for robotic manipulation.\n\nGiven a task instruction and a single image, estimate the current progress toward completing the task.\n\nObservation: '<image>'\n\nInstruction: {instruction}"""
                },
                {
                    'type' : 'image',
                    'image' : obs
                }
            ]
        }]
        messages.append(message_template)
    return messages
        
def decode_value_token(value_tokenizer, generated_token_id):
        """Decode a single token ID to continuous value using ValueTokenizer."""
        # Convert token ID to numpy array
        token_id_array = np.array([generated_token_id])
        # Decode using value_tokenizer
        value = value_tokenizer.decode_token_ids_to_values(token_id_array)
        return value[0] if len(value) > 0 else 0.0