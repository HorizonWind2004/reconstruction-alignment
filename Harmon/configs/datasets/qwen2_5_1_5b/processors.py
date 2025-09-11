from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoTokenizer


llm_name_or_path = 'Qwen/Qwen2.5-1.5B-Instruct'
prompt_template = PROMPT_TEMPLATE.qwen_chat
pad_index = 151645
image_length = 1024 + 64
image_size = 512

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

