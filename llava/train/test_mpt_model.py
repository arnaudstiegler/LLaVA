from llava.model.llava_mpt import LlavaMPTForCausalLM

model = LlavaMPTForCausalLM.from_pretrained(
                'liuhaotian/LLaVA-Lightning-MPT-7B-preview',
            )

import ipdb; ipdb.set_trace()