from paddlenlp.transformers import AutoModelForCausalLM
from paddlenlp.transformers import AutoTokenizer
# from paddlenlp.generation import GenerationConfig
# from paddlenlp.trl import llm_utils
# pip install -U   paddlepaddle   -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

model_id = "paddlenlp/PP-UIE-0.5B"


def test():
    model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # generation_config = GenerationConfig.from_pretrained(model_id)


    template = """
    你是一个阅读理解专家，请提取所给句子与问题，提取实体。请注意，如果存在实体，则一定在原句中逐字出现，请输出对应实体的原文，不要进行额外修改；如果无法提取，请输出“无相应实体”。
    **句子开始**
    {sentence}
    **句子结束**
    **问题开始**
    {prompt}
    **问题结束**
    **回答开始**
    """

    sentences = [
        "如有单位或个人对公示人员申请廉租住房保障资格有异议的，可以信件和电话的形式向市住建局举报，监督电话：5641079",
        "姓名：张三，年龄：30岁，手机：13854488452，性别：男，家庭住址：北京市海淀区西北旺",
        "张三,30岁,13854488452,男,北京市海淀区西北旺",
    ]

    prompts = [
        "电话号码",
        "姓名，年龄，手机号码，性别，地址",
        "姓名",
    ]

    inputs = [template.format(sentence=sentence, prompt=prompt) for sentence, prompt in zip(sentences, prompts)]
    inputs = [tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in inputs]
    input_features = tokenizer(
        inputs,
        max_length=512,
        return_position_ids=False,
        truncation=True,
        truncation_side="left",
        padding=True,
        return_tensors="pd",
        add_special_tokens=False,
    )

    outpus = model(**input_features)

    # outputs = model.generate(
    #     **input_features,
    #     max_new_tokens=200,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=llm_utils.get_eos_token_id(tokenizer, generation_config),
    #     pad_token_id=tokenizer.pad_token_id,
    #     decode_strategy="greedy_search",
    #     temperature=1.0,
    #     top_k=1,
    #     top_p=1.0,
    #     repetition_penalty=1.0,
    # )


    def get_clean_entity(text):
        ind1 = text.find("\n**回答结束**\n\n")
        if ind1 != -1:
            pred = text[:ind1]
        else:
            pred = text
        return pred


    results = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    results = [get_clean_entity(result) for result in results]

    for sentence, prompt, result in zip(sentences, prompts, results):
        print("-" * 50)
        print(f"Sentence: {sentence}")
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        

test()
        
# from pprint import pprint
# from paddlenlp import Taskflow

# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
# ie = Taskflow('information_extraction',
#               schema= ['时间', '选手', '赛事名称'],
#               schema_lang="zh",
#               batch_size=1,
#               model='paddlenlp/PP-UIE-0.5B')
# pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
# # 输出
# [{'时间': [{'text': '2月8日上午'}],
#   '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
#   '选手': [{'text': '谷爱凌'}]}]