import torch
import transformers

from llava import conversation as conversation_lib
from llava.model import *
from llava.train.train import (
    TrainingArguments, ModelArguments, DataArguments, make_supervised_data_module,
    smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN
)
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPConfig


ANSWER_TOKEN = '<answer>'

# This one has already been pre-trained and finetuned with multimodal data
model_name_or_path = 'liuhaotian/LLaVA-Lightning-MPT-7B-preview'
# model_name_or_path = 'mosaicml/mpt-7b'
model_args = ModelArguments(
    model_name_or_path=model_name_or_path,
    version='v1',
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    vision_tower='openai/clip-vit-large-patch14',
    mm_vision_select_layer=-2,
    # TODO: figure out how to get this
    pretrain_mm_mlp_adapter='./checkpoints/mm_projector/llava-lightning-mpt-7b-pretrain.bin',
    mm_use_im_start_end=True
)

data_args = DataArguments(
    data_path='/Users/arnaudstiegler/Desktop/VQA/raw_datasets/DocVQA/train/train_llava.json',
    lazy_preprocess=True,
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=0,
    image_folder='/Users/arnaudstiegler/Desktop/VQA/raw_datasets/DocVQA/train/',
    image_aspect_ratio='keep'  # Can be 'square' or 'keep' or 'pad'
)

training_args = TrainingArguments(output_dir='/Users/arnaudstiegler/Desktop/test/')


if model_args.vision_tower is not None:
    if 'mpt' in model_args.model_name_or_path.lower():
        model = LlavaMPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
else:
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
model.config.use_cache = False

if model_args.freeze_backbone:
    model.model.requires_grad_(False)

if 'mpt' in model_args.model_name_or_path.lower():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

if model_args.version == "v0":
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
else:
    tokenizer.pad_token = tokenizer.unk_token
    # TODO: will have to resize the embedding
    tokenizer.add_tokens(ANSWER_TOKEN)
    if "mpt" in model_args.model_name_or_path.lower():
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]


if model_args.vision_tower is not None:
    # model_vision_dict = model.get_model().initialize_vision_modules(
    #     vision_tower=model_args.vision_tower,
    #     mm_vision_select_layer=model_args.mm_vision_select_layer,
    #     pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    # )
    image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)
    config_model_vision = CLIPConfig.from_pretrained(model_args.vision_tower)
    config_model_vision.vision_config.image_size = 1024
    config_model_vision.vision_config.use_im_start_end = True
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    import ipdb; ipdb.set_trace()
    # self.resize_token_embeddings(len(tokenizer))
    model_vision = CLIPVisionModel.from_pretrained(model_args.vision_tower, config=config_model_vision.vision_config)
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # model.get_model().vision_tower[0].to(dtype=dtype, device=training_args.device)
    vision_config = model_vision.config

    # data_args.image_token_len = model_vision_dict['image_token_len']
    # TODO: increasing the image size here:
    vision_config.image_size = 1024
    data_args.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    # data_args.image_processor = model_vision_dict['image_processor']
    data_args.image_processor = image_processor
    data_args.is_multimodal = True

    # model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # if model_args.tune_mm_mlp_adapter:
    #     model.requires_grad_(False)
    #     for p in model.get_model().mm_projector.parameters():
    #         p.requires_grad = True

    # model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    # if training_args.freeze_mm_mlp_adapter:
    #     for p in model.get_model().mm_projector.parameters():
    #         p.requires_grad = False

    # model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    # vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
    # model.config.sep_image_conv_front = data_args.sep_image_conv_front
    # model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
    #                                   tokenizer=tokenizer, device=training_args.device,
    #                                   tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
    #                                   pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)
    #
    # params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # if len(params_no_grad) > 0:
    #     if training_args.fsdp is not None and len(training_args.fsdp) > 0:
    #         if len(params_no_grad) < 10:
    #             print(
    #                 '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
    #                     len(params_no_grad), params_no_grad))
    #         else:
    #             print(
    #                 '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
    #                     len(params_no_grad), ', '.join(params_no_grad[:10])))
    #         print(
    #             "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
    #         print(
    #             "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")
    #
    #         from torch.distributed.fsdp.fully_sharded_data_parallel import \
    #             FullyShardedDataParallel as FSDP
    #
    #
    #         def patch_FSDP_use_orig_params(func):
    #             def wrap_func(*args, **kwargs):
    #                 use_orig_params = kwargs.pop('use_orig_params', True)
    #                 return func(*args, **kwargs, use_orig_params=use_orig_params)
    #
    #             return wrap_func
    #
    #
    #         FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

data_dicts = make_supervised_data_module(
    tokenizer=tokenizer,
    data_args=data_args
)

dataset = data_dicts['train_dataset']
print(dataset[0])
import ipdb; ipdb.set_trace()