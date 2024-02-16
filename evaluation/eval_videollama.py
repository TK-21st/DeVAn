"""
Evaluation Script for Video-LLaMA2

Following instruction in https://github.com/DAMO-NLP-SG/Video-LLaMA for environment setup.
Place this script in the Video-LLaMA root path.
"""
import argparse
from pathlib import Path
from tqdm.auto import tqdm

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

# LM Decoder Config
MAX_NEW_TOKENS = 300
MAX_LENGTH = 2000

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_only_vl.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='LLama2', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


class ChatBot:

    def __init__(self, args):
        self.chat = self._init_model(args)
        if args.model_type == 'vicuna':
            self.chat_state = default_conversation.copy()
        else:
            self.chat_state = conv_llava_llama_2.copy()
        self.img_list = list()
        self.set_para()

    def _init_model(self, args):
        print('Initializing Chat')
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print('Initialization Finished')
        return chat

    def set_para(self, num_beams=1, temperature=1.0):
        self.num_beams = num_beams
        self.temperature = temperature
        print('set num_beams: {}'.format(num_beams))
        print('set temperature: {}'.format(temperature))

    def upload(self, up_img=False, up_video=False, audio_flag=False):
        if up_img and not up_video:
            self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            self.chat.upload_img(up_img, self.chat_state, self.img_list)
        elif not up_img and up_video:
            self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            if audio_flag:
                self.chat.upload_video(up_video, self.chat_state, self.img_list)
            else:
                self.chat.upload_video_without_audio(up_video, self.chat_state, self.img_list)

    def ask_answer(self, user_message):
        self.chat.ask(user_message, self.chat_state)
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       num_beams=self.num_beams,
                                       temperature=self.temperature,
                                       max_new_tokens=MAX_NEW_TOKENS,
                                       max_length=MAX_LENGTH)[0]

        return llm_message

    def reset(self):
        if self.chat_state is not None:
            self.chat_state.messages = list()
        if self.img_list is not None:
            self.img_list = list()
        self.set_para()

if __name__ == "__main__":
    args = parse_args()
    chatbot = ChatBot(args)
    input_paths = "/path/to/all/videos"
    video_paths = [p for p in Path(input_paths).iterdir() if p.suffix == ".mp4"]
    chatbot.set_para(num_beams=1, temperature=1.0) # default parameters

    for file_path in tqdm(video_paths):
        try:
            chatbot.upload(up_video=file_path, audio_flag=True)
        except Exception as e:
            print(f"Failed to Parse Video {file_path}, skipped: {e}")
            continue
        caption = chatbot.ask_answer(user_message="Summarize this video in one sentence.")
        print("CAPTION", caption)
        chatbot.reset()
        summary = chatbot.ask_answer(user_message="Summarize this video in three to ten sentences.")
        print("SUMMARY", caption)
        chatbot.reset()
        break