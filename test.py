import debugpy
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
#from openpi.qwen_vl.qwen_eval import qwen_eval
import debugpy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from transformers import AutoProcessor
from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer
from PIL import Image
import numpy as np
def load_model_and_processor(model_name_or_path, attn_implementation=None):

    """Load model and processor from checkpoint."""

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.config.use_cache = True
    value_tokenizer = ValueTokenizer(
            llm_path=model_name_or_path,  # Use same model path to ensure consistency
            bins=201,
            min_value=-1.0,
            max_value=0.0,
    )
    return model, processor, value_tokenizer

def decode_value_token(value_tokenizer, generated_token_id):
    """Decode a single token ID to continuous value using ValueTokenizer."""
    # Convert token ID to numpy array
    token_id_array = np.array([generated_token_id])
    # Decode using value_tokenizer
    value = value_tokenizer.decode_token_ids_to_values(token_id_array)
    return value[0] if len(value) > 0 else 0.0

if __name__ == "__main__":
    print("start Listening")
    debugpy.listen(('0.0.0.0',5678))
    debugpy.wait_for_client()
    print("start Listening")

    data = LeRobotDataset(repo_id="beat_block_hammer_rollout")
    dataset_meta = LeRobotDatasetMetadata(repo_id = "beat_block_hammer_rollout")
    #dataset_meta.episodes[0]['length']
    #print(data[0])
    model, processor, value_tokenizer = load_model_and_processor("/project/peilab/junhao/Value_Function/qwen-vl-finetune/output/gpus_8/checkpoint-3000")
    messages = []
    for i in range(5):
        episode = data[i]
        image =  episode['observation.images.cam_high']
        image = image.detach().cpu().numpy() # (3, 224, 224)
        image = image.transpose(1, 2, 0) # (224, 224, 3)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        messages.append([{
                'role' : 'user',
                'content' : [
                    {
                        'type' : 'text',
                        'text' : "What's the main object in this picture?"
                    },
                    {
                        'type' : 'image',
                        'image' : image
                    }
                ]
            }])
    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Add generation prompt to get model to generate
            return_dict=True,
            return_tensors="pt"
        )
    with torch.no_grad():
        generation_config = {
            "max_new_tokens": 1,  # We only need one value token
            "do_sample": False,   # Greedy decoding for deterministic prediction
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        generated_outputs = model.generate(
            **inputs,
            **generation_config,
            return_dict_in_generate=True,
            output_logits=True
        )
        generated_token_ids = generated_outputs.sequences[0][len(inputs['input_ids'][0]):]
        generated_token_id = generated_token_ids[0].item() if len(generated_token_ids) > 0 else None
        if generated_token_id and generated_token_id in value_tokenizer.extra_id_token_ids:
            predicted_value = decode_value_token(value_tokenizer, generated_token_id)
        print(predicted_value)