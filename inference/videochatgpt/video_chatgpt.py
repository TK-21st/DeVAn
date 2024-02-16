import os
import io
import torch
from typing import List
from PIL import Image
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, LlamaTokenizer
import logging

from ..base_model import VideoToTextModel
from ._video_chatgpt import (
    VideoChatGPTLlamaForCausalLM,
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
)
from ._video_chatgpt_conversation import conv_templates, SeparatorStyle
from .utils import disable_torch_init, get_spatio_temporal_features_torch, KeywordsStoppingCriteria, load_video

DEFAULT_CAPTION_PROMPT = "Please write a sentence caption of the video."
DEFAULT_SUMMARY_PROMPT = "Please write 3 to 5 sentence summary of the video."

logger = logging.getLogger(__name__)

class VideoChatGPT(VideoToTextModel):
    def __init__(self, model_name: str, projection_path: str = None):
        super().__init__()

        (
            self.model,
            self.vision_tower,
            self.tokenizer,
            self.image_processor,
            self.video_token_len
        ) = self.initialize_model(model_name, projection_path)


    def generate_text(
        self, 
        video: bytes, 
        prompt: str = DEFAULT_CAPTION_PROMPT, 
        conv_mode: str="video-chatgpt_v1"
    ) -> str:
        video_frames = load_video(io.BytesIO(video))
        # Run inference on the video and add the output to the list
        output = self.video_chatgpt_infer(
            video_frames,
            prompt,
            conv_mode,
            self.model,
            self.vision_tower,
            self.tokenizer,
            self.image_processor,
            self.video_token_len
        )
        return output

    @classmethod
    def video_chatgpt_infer(
        cls,
        video_frames: List[Image.Image],
        question: str,
        conv_mode: str,
        model: VideoChatGPTLlamaForCausalLM,
        vision_tower: CLIPVisionModel,
        tokenizer: AutoTokenizer,
        image_processor: CLIPImageProcessor,
        video_token_len: int
    ) -> str:
        """
        Run inference using the Video-ChatGPT model.

        Parameters:
        sample : Initial sample
        video_frames (torch.Tensor): Video frames to process.
        question (str): The question string.
        conv_mode: Conversation mode.
        model: The pretrained Video-ChatGPT model.
        vision_tower: Vision model to extract video features.
        tokenizer: Tokenizer for the model.
        image_processor: Image processor to preprocess video frames.
        video_token_len (int): The length of video tokens.

        Returns:
        dict: Dictionary containing the model's output.
        """

        # Prepare question string for the model
        if model.get_model().vision_config.use_vid_start_end:
            qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
        else:
            qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

        # Prepare conversation prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize the prompt
        inputs = tokenizer([prompt])

        # Preprocess video frames and get image tensor
        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

        # Move image tensor to GPU and reduce precision to half
        image_tensor = image_tensor.half().cuda()

        # Generate video spatio-temporal features
        with torch.no_grad():
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

        # Move inputs to GPU
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # Define stopping criteria for generation
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # Run model inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        # Check if output is the same as input
        n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        # Decode output tokens
        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Clean output string
        outputs = outputs.strip().rstrip(stop_str).strip()

        return outputs


    @classmethod
    def initialize_model(cls, model_name, projection_path=None):
        """
        Initializes the model with given parameters.

        Parameters:
        model_name (str): Name of the model to initialize.
        projection_path (str, optional): Path to the projection weights. Defaults to None.

        Returns:
        tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
        """
        logger.info(f"Initializing VideoChatGPT Model using Model {model_name} and Projection {projection_path}")

        # Disable initial torch operations
        disable_torch_init()

        # Convert model name to user path
        model_name = os.path.expanduser(model_name)

        # Load tokenizer
        logger.info(f"Loading tokenizer for model - {model_name}")
        # loading LlamaTokenizer from autotokenizer is incredibly slow. 
        # to save time, we first try loading with LlamaTokenizer.
        # NOTE: this may lead to undesirable behaviors when the tokenizer is not LlamaTokenizer, but oh well.
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        logger.info(f"Loading VideoChatGPTLlamaForCausalLM Model")
        model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                            use_cache=True)

        # Load image processor
        logger.info(f"Loading Image Processor Model")
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        # Set to use start and end tokens for video
        mm_use_vid_start_end = True

        # Add tokens to tokenizer
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_vid_start_end:
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

        # Resize token embeddings of the model
        model.resize_token_embeddings(len(tokenizer))

        # Load the weights from projection_path after resizing the token_embeddings
        if projection_path:
            logger.info(f"Loading projection weights from {projection_path}")
            status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
            if status.unexpected_keys:
                logger.warn(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
            logger.info(f"Projection weights loaded from {projection_path}")

        # Set model to evaluation mode and move to GPU
        model = model.eval()
        model = model.cuda()

        vision_tower_name = "openai/clip-vit-large-patch14"

        # Load vision tower and move to GPU
        logger.info("Loading vision tower")
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True).cuda()
        vision_tower = vision_tower.eval()

        # Configure vision model
        vision_config = model.get_model().vision_config
        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
        vision_config.use_vid_start_end = mm_use_vid_start_end
        if mm_use_vid_start_end:
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

        # Set video token length
        video_token_len = 356

        logger.info("Finished loading VideoChatGPT Model!")
        return model, vision_tower, tokenizer, image_processor, video_token_len