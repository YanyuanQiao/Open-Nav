from openai import OpenAI
import torch
import numpy as np

import sys
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from transformers import AutoConfig, AutoModelForCausalLM
from SpatialBot3B.configuration_bunny_phi import *
from SpatialBot3B.modeling_bunny_phi import *

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

from recognize_anything.ram.models import ram
from recognize_anything.ram import inference_ram
from recognize_anything.ram import get_transform



class llmClient:
    def __init__(self, model_type = '', api_key=None, base_url=None):
        '''
        Initialize LLM client based on model type and API key.
        
        Args:
            model_type (str): Either "gpt" or "opensource"
            api_key (str): API key for OpenAI (if using GPT)
        '''
        # Configure based on model type
        if model_type == "gpt-4o-2024-08-06":
            self.model = model_type
            self.client = OpenAI(api_key=api_key)
            
        elif model_type == "Qwen/Qwen2-72B":
            self.model = model_type
            self.client = OpenAI(
                api_key="not-needed",  # This value doesn't matter for local deployment
                base_url="http://0.0.0.0:23333/v1"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gpt' or 'opensource'.")
        
        print(f"Initialized LLM client with model: {self.model}")

    def set_model(self, model):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def gpt_infer(self, system_prompt, user_prompt, num_output=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }
        
        if num_output == 1:
            chat_response = self._completion_with_backoff(**request_params)
            answer = chat_response.choices[0].message.content
            return answer
        else:
            responses = []
            for _ in range(num_output):
                chat_response = self._completion_with_backoff(**request_params)
                responses.append(chat_response.choices[0].message.content)
            return responses

    
class spatialClient:
    def __init__(self, device):
        self.device = device
        self.ram_path = "./recognize_anything/pretrained/ram_swin_large_14m.pth"
        self.spatialbot_path = "./SpatialBot3B"
        view_record_path = "cache_files/view_cache.json"
        try:
            self.spatialbot_model = AutoModelForCausalLM.from_pretrained(
                self.spatialbot_path,
                torch_dtype=torch.float16, # float32 for cpu
                device_map='auto',
                trust_remote_code=True)
            self.spatialbot_tokenizer = AutoTokenizer.from_pretrained(
                self.spatialbot_path,
                trust_remote_code=True)
            
            self.ram_transform = get_transform(image_size=224) 
            self.ram_model = ram(pretrained=self.ram_path, image_size=224, vit='swin_l').eval().to(self.device)
        except Exception as e:
            print(f"Error in loading RAM or SpatialBot: {e}")
            
    def ram_img_tagging(self, image):
        ram_img = self.ram_transform(image).unsqueeze(0).to(self.device)
        img_tags = inference_ram(ram_img, self.ram_model)[0]
        return img_tags
    
    def spatialbot_description(self, image_dict, prompt):
        offset_bos = 0
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [self.spatialbot_tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(self.device)
        image1 = image_dict['rgb']
        image2 = image_dict['depth']
        channels = len(image2.getbands())
        if channels == 1:
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            image2 = Image.fromarray(three_channel_array, 'RGB')
        image_tensor = self.spatialbot_model.process_images([image1,image2], self.spatialbot_model.config).to(dtype=self.spatialbot_model.dtype, device=self.device)
        self.spatialbot_model.get_vision_tower().to('cuda')
        output_ids = self.spatialbot_model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=200, 
            use_cache=True,
            repetition_penalty=1.0 
        )[0]
        return self.spatialbot_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    def observe_view(self, logger, current_step, direction_idx, direction_image):
        img_tags = self.ram_img_tagging(direction_image['rgb'])
        spatial_scene_description_prompt = "What objects are in the image, and how far are these objects from the camera, calculate the result in meter."
        spatial_scene_description = self.spatialbot_description(direction_image, spatial_scene_description_prompt)
        view_observation = f"Scene Description: {spatial_scene_description} Scene Objects: {img_tags}; "
        observe_result = f"Direction {direction_idx} Direction Viewpoint ID: {direction_idx} in Step ID: {current_step} Elevation: Eye Level "  + view_observation
        return observe_result