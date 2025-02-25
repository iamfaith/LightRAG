from collections import defaultdict
import time
from utils import *
from prompt import PROMPTS
import re
language = PROMPTS["DEFAULT_LANGUAGE"]
print(language)
language = "Chinese"
entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]

example_context_base = dict(
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    entity_types=",".join(entity_types),
    language=language,
)

example_number = None
if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
    examples = "\n".join(
        PROMPTS["entity_extraction_examples"][: int(example_number)]
    )
else:
    examples = "\n".join(PROMPTS["entity_extraction_examples"])

# add example's format
examples = examples.format(**example_context_base)

entity_extract_prompt = PROMPTS["entity_extraction"]
context_base = dict(
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    entity_types=",".join(entity_types),
    examples=examples,
    language=language,
)



async def _handle_single_entity_extraction(
    record_attributes,
    chunk_key: str,
):
    # “entity”
    if len(record_attributes) < 4 or "entity" not in record_attributes[0]:
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes,
    chunk_key: str,
):
    # '“relationship”'
    if len(record_attributes) < 5 or "relationship" not in record_attributes[0]:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]
              ) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


# print(records)


async def _user_llm_func_with_cache(
    input_text: str, history_messages=None, enable_llm_cache_for_entity_extract=False, llm_response_cache=None
) -> str:
    if enable_llm_cache_for_entity_extract and llm_response_cache:
        if history_messages:
            history = json.dumps(history_messages, ensure_ascii=False)
            _prompt = history + "\n" + input_text
        else:
            _prompt = input_text

        arg_hash = compute_args_hash(_prompt)
        cached_return, _1, _2, _3 = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type="extract",
            force_llm_cache=True,
        )
        if cached_return:
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1
            return cached_return
        statistic_data["llm_call"] += 1
        if history_messages:
            res: str = await use_llm_func(
                input_text, history_messages=history_messages
            )
        else:
            res: str = await use_llm_func(input_text)
        await save_to_cache(
            llm_response_cache,
            CacheData(
                args_hash=arg_hash,
                content=res,
                prompt=_prompt,
                cache_type="extract",
            ),
        )
        return res

    from rag import RAG
    use_llm_func = RAG.llm_model_func
    if history_messages:
        return await use_llm_func(input_text, history_messages=history_messages)
    else:
        return await use_llm_func(input_text)


async def _process_single_content(processed_chunks, total_chunks, chunk_key, content, save_prompt=True):
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    
    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(**context_base, input_text=content)

    if save_prompt:
        with open("prompt.txt", "w") as f:
            f.write(hint_prompt)
    
    final_result = await _user_llm_func_with_cache(hint_prompt)

    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue

        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )

        if_entities = await _handle_single_entity_extraction(
            record_attributes, chunk_key
        )

        if if_entities is not None:
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            continue

        if_relation = await _handle_single_relationship_extraction(
            record_attributes, chunk_key
        )
        if if_relation is not None:
            maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                if_relation
            )
    entities_count = len(maybe_nodes)
    relations_count = len(maybe_edges)
    
    logger.info(
        f"Chunk {processed_chunks}/{total_chunks}: extracted {entities_count} entities and {relations_count} relationships (deduplicated)"
    )
    # print(maybe_edges)
    # print(maybe_nodes)
    return dict(maybe_nodes), dict(maybe_edges)


async def extract_entities(ordered_chunks):
    processed_chunks = 0
    total_chunks = len(ordered_chunks)
    
    # tasks = [_process_single_content(c) for c in ordered_chunks]
    tasks = []
    for c in ordered_chunks:
        tasks.append(_process_single_content(processed_chunks, total_chunks, chunk_key, c))
        processed_chunks += 1
    
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    print(maybe_nodes)
    print(maybe_edges)
    print(len(maybe_edges), len(maybe_nodes))
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    

if __name__ == "__main__":
    content = '''正方云曦山地址是(香洲区)珠海大道157号, 建筑类型是塔楼, 房屋总数是366户, 楼栋总数是4栋, 绿化率是35%, 容积率是3.3, 交易权属是商品房, 建成年代是暂无信息, 供暖类型是无供暖, 用水类型是民水, 用电类型是民电, 初中学区是香洲区 容国团中学, 小学学区是香洲区 容国团小学'''
    # content = '城市内交通费7月5日金额114广州至佛山'
    chunk_key = "xuequ"    
    ordered_chunks = []
    import json
    with open("/home/faith/community_dict/data/贝壳/珠海-upload.json", 'r') as f:
        for k, v in json.load(f).items():
            content = k
            for _k, _v in v.items():
                content += _k + "是" + _v + ","
            content = content[:-1] + '。'
            content = content.replace(' ', '')
            ordered_chunks.append(content)
    
    
    ordered_chunks = ordered_chunks[:5]
    print(len(ordered_chunks))

    import time
    start = time.time()
    loop = always_get_an_event_loop()
    loop.run_until_complete(extract_entities(ordered_chunks))
    print(time.time() - start)
    # loop.run_until_complete(_process_single_content(chunk_key, content))
    
    
# defaultdict(<class 'list'>, {'"天地源上唐府"': [{'entity_name': '"天地源上唐府"', 'entity_type': '"ORGANIZATION"', 'description': '"天地源上唐 府是一家位于高新区的房地产开发企业，拥有板楼建筑类型，共有976户住宅，5栋楼栋，绿化率为25.6%，容积率为2.5，交易权属为商品房。"', 'source_id': 'xuequ'}], '"高新区"': [{'entity_name': '"高新区"', 'entity_type': '"GEO"', 'description': '"高新区是天地源上唐府所在的地理位置，是行政区划名称。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区"', 'entity_type': '"GEO"', 'description': '"高新区是正圆花园所在的地理区域，具有明确的地理 位置和行政划分。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区"', 'entity_type': '"GEO"', 'description': '"高新区是正方世和苑所在的地域， 属于行政区域。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区"', 'entity_type': '"GEO"', 'description': '"高新区是金泰园所在的地理区域，具 有特定的行政和规划功能。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区"', 'entity_type': '"GEO"', 'description': '"高新区是唐家中学教师楼 所在地的地理区域。"', 'source_id': 'xuequ'}], '"金业北路"': [{'entity_name': '"金业北路"', 'entity_type': '"GEO"', 'description': '"金业北路是 天地源上唐府的具体地址所在道路。"', 'source_id': 'xuequ'}], '"板楼"': [{'entity_name': '"板楼"', 'entity_type': '"CATEGORY"', 'description': '"板楼是天地源上唐府的建筑类型，指的是多层住宅建筑。"', 'source_id': 'xuequ'}, {'entity_name': '"板楼"', 'entity_type': '"CATEGORY"', 'description': '"板楼是正圆花园的建筑类型，通常指多层住宅建筑。"', 'source_id': 'xuequ'}, {'entity_name': '"板楼"', 'entity_type': '"CATEGORY"', 'description': '"板楼是金泰园的建筑类型，通常指多层住宅建筑。"', 'source_id': 'xuequ'}], '"高新区礼和小学"': [{'entity_name': '"高新区礼和小学"', 'entity_type': '"ORGANIZATION"', 'description': '"高新区礼和小学是天地源上唐府所在的小学学区。"', 'source_id': 'xuequ'}], '"正圆花园"': [{'entity_name': '"正圆花园"', 'entity_type': '"ORGANIZATION"', 'description': '"正圆花园是一个位于高新区的住宅小区，拥有板楼建筑类型，房屋总数209户，楼栋总数11栋，绿化率为25%，容积率为3。"', 'source_id': 'xuequ'}], '"唐淇路1208号"': [{'entity_name': '"唐淇路1208号"', 'entity_type': '"GEO"', 'description': '"唐淇路1208号是正圆花园的具体地址，位于高新区内。"', 'source_id': 'xuequ'}], '"209户"': [{'entity_name': '"209户"', 'entity_type': '"ORGANIZATION"', 'description': '"正圆花园共有209户住宅。"', 'source_id': 'xuequ'}], '"11栋"': [{'entity_name': '"11栋"', 'entity_type': '"ORGANIZATION"', 'description': '"正圆花园共有11栋楼栋。"', 'source_id': 'xuequ'}], '"25%"': [{'entity_name': '"25%"', 'entity_type': '"CATEGORY"', 'description': '"正圆花园的绿化率为25%，表示小区内绿化覆盖面积的比例。"', 'source_id': 'xuequ'}], '"3"': [{'entity_name': '"3"', 'entity_type': '"CATEGORY"', 'description': '"正圆花园的容积率为3，表示建筑密度与土地使用效率。"', 'source_id': 'xuequ'}, {'entity_name': '"3"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的容积率为3，表示建筑总体积与用地面积的比值。"', 'source_id': 'xuequ'}], '"商品房/使用权"': [{'entity_name': '"商品房/使用权"', 'entity_type': '"ORGANIZATION"', 'description': '"正圆花园的交易权属为商品房/使用权，意味着房屋可以买卖或租赁。"', 'source_id': 'xuequ'}], '"暂无信息"': [{'entity_name': '"暂无信息"', 'entity_type': '"CATEGORY"', 'description': '"暂无信息表示关于正圆花园的供暖类型和建成年代目前没有具体信息。"', 'source_id': 'xuequ'}, {'entity_name': '"暂无信息"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的建成年代信息暂无。"', 'source_id': 'xuequ'}, {'entity_name': '"暂无信息"', 'entity_type': '"CATEGORY"', 'description': '"暂无信息表示某些关于金泰园的信息目前尚未公布或不可用。"', 'source_id': 'xuequ'}, {'entity_name': '"暂无信息"', 'entity_type': '"CATEGORY"', 'description': '"关于唐家中学教师楼的建成年代和供暖类型暂无信息。"', 'source_id': 'xuequ'}], '"民水"': [{'entity_name': '"民水"', 'entity_type': '"CATEGORY"', 'description': '"正圆花园的用水类型为民水，表示使用的是居民用水。"', 'source_id': 'xuequ'}, {'entity_name': '"民水"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的用水类型是民水，表示使用的是居民用水。"', 'source_id': 'xuequ'}, {'entity_name': '"民水"', 'entity_type': '"CATEGORY"', 'description': '"民水是金泰园的用水类型，通常指居民生活用水。"', 'source_id': 'xuequ'}, {'entity_name': '"民水"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼使用民水。"', 'source_id': 'xuequ'}], '"民电"': [{'entity_name': '"民电"', 'entity_type': '"CATEGORY"', 'description': '"正圆花园的用电类型为民电，表示使用的是居民用电。"', 'source_id': 'xuequ'}, {'entity_name': '"民电"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的用电类型是民电，表示使用的是居民用电。"', 'source_id': 'xuequ'}, {'entity_name': '"民电"', 'entity_type': '"CATEGORY"', 'description': '"民电是金泰园的用电类型，通常指居民生活用电。"', 'source_id': 'xuequ'}, {'entity_name': '"民电"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼使用民电。"', 'source_id': 'xuequ'}], '"高新区中大附中"': [{'entity_name': '"高新区中大附中"', 'entity_type': '"GEO"', 'description': '"高新区中大附中是正圆花园的初中学区，为学生提供教育资源。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区中 大附中"', 'entity_type': '"GEO"', 'description': '"高新区中大附中是金泰园的初中学区，反映了小区的教育资源。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区中大附中"', 'entity_type': '"GEO"', 'description': '"高新区中大附中是唐家中学教师楼的初中学区。"', 'source_id': 'xuequ'}], '"高 新区中大附小后环校区"': [{'entity_name': '"高新区中大附小后环校区"', 'entity_type': '"GEO"', 'description': '"高新区中大附小后环校区是正圆花园 的小学区，为学生提供教育资源。"', 'source_id': 'xuequ'}, {'entity_name': '"高新区中大附小后环校区"', 'entity_type': '"GEO"', 'description': '" 高新区中大附小后环校区是金泰园的小学区，反映了小区的教育资源。"', 'source_id': 'xuequ'}], '"正方世和苑"': [{'entity_name': '"正方世和苑"', 'entity_type': '"ORGANIZATION"', 'description': '"正方世和苑是一个位于高新区的住宅小区，具有塔楼建筑类型，共有196户住宅，3栋楼栋，绿化率为35.19%， 容积率为3，交易权属为商品房/房改房。"', 'source_id': 'xuequ'}], '"泉星路81号"': [{'entity_name': '"泉星路81号"', 'entity_type': '"GEO"', 'description': '"泉星路81号是正方世和苑的具体地址。"', 'source_id': 'xuequ'}], '"塔楼"': [{'entity_name': '"塔楼"', 'entity_type': '"CATEGORY"', 'description': '"塔楼是正方世和苑的建筑类型，通常指高层住宅建筑。"', 'source_id': 'xuequ'}], '"196户"': [{'entity_name': '"196户"', 'entity_type': '"PERSON"', 'description': '"正方世和苑共有196户住宅。"', 'source_id': 'xuequ'}], '"3栋"': [{'entity_name': '"3栋"', 'entity_type': '"GEO"', 'description': '"正方世和苑共有3栋楼栋。"', 'source_id': 'xuequ'}], '"35.19%"': [{'entity_name': '"35.19%"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的绿化率为35.19%，表示绿化覆盖面积占总面积的百分比。"', 'source_id': 'xuequ'}], '"商品房/房改房"': [{'entity_name': '"商品 房/房改房"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的交易权属为商品房/房改房，表示房屋性质。"', 'source_id': 'xuequ'}], '"无 供暖"': [{'entity_name': '"无供暖"', 'entity_type': '"CATEGORY"', 'description': '"正方世和苑的供暖类型是无供暖，表示该小区没有集中供暖设施。"', 'source_id': 'xuequ'}], '"金泰园"': [{'entity_name': '"金泰园"', 'entity_type': '"ORGANIZATION"', 'description': '"金泰园是一个位于高新区的住宅小区，拥有板楼建筑类型，共有90户住宅和7栋楼栋。"', 'source_id': 'xuequ'}], '"唐淇路1228号"': [{'entity_name': '"唐淇路1228号"', 'entity_type': '"GEO"', 'description': '"唐淇路1228号是金泰园的具体地址，位于高新区内。"', 'source_id': 'xuequ'}], '"90户"': [{'entity_name': '"90户"', 'entity_type': '"CATEGORY"', 'description': '"90户是金泰园的住宅数量，反映了小区的规模。"', 'source_id': 'xuequ'}], '"7栋"': [{'entity_name': '"7栋"', 'entity_type': '"CATEGORY"', 'description': '"7栋是金泰园的楼栋总数，反映了小区的建筑密度。"', 'source_id': 'xuequ'}], '"34%"': [{'entity_name': '"34%"', 'entity_type': '"CATEGORY"', 'description': '"34%是金泰园的绿化率，反映了小区的生态环境。"', 'source_id': 'xuequ'}], '"2.3"': [{'entity_name': '"2.3"', 'entity_type': '"CATEGORY"', 'description': '"2.3是金泰园的容积率，反映了小区的建筑密度。"', 'source_id': 'xuequ'}], '"商品房"': [{'entity_name': '"商品房"', 'entity_type': '"CATEGORY"', 'description': '"商品房是金泰园的交易权属类型，通常指个人或家庭拥有的住宅。"', 'source_id': 'xuequ'}, {'entity_name': '"商品房"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼的交易权属为商品房。"', 'source_id': 'xuequ'}], '"唐家中学"': [{'entity_name': '"唐家中学"', 'entity_type': '"ORGANIZATION"', 'description': '"唐家中学是一所中学，其教师楼位于高新区白埔西路。"', 'source_id': 'xuequ'}], '"白埔西路"': [{'entity_name': '"白埔西路"', 'entity_type': '"GEO"', 'description': '"白埔西路 是唐家中学教师楼的具体地址。"', 'source_id': 'xuequ'}], '"平房"': [{'entity_name': '"平房"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼建筑类型为平房。"', 'source_id': 'xuequ'}], '"96户"': [{'entity_name': '"96户"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼共有96户房屋。"', 'source_id': 'xuequ'}], '"4栋"': [{'entity_name': '"4栋"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼共有4栋楼栋。"', 'source_id': 'xuequ'}], '"30%"': [{'entity_name': '"30%"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼的绿 化率为30%。"', 'source_id': 'xuequ'}], '"1.5"': [{'entity_name': '"1.5"', 'entity_type': '"CATEGORY"', 'description': '"唐家中学教师楼的容积率 为1.5。"', 'source_id': 'xuequ'}], '"高新区唐家小学"': [{'entity_name': '"高新区唐家小学"', 'entity_type': '"GEO"', 'description': '"高新区唐家小学是唐家中学教师楼的小学区。"', 'source_id': 'xuequ'}]})
# defaultdict(<class 'list'>, {('"天地源上唐府"', '"高新区"'): [{'src_id': '"天地源上唐府"', 'tgt_id': '"高新区"', 'weight': 8.0, 'description': '"天地源上唐府位于高新区的金业北路。"', 'keywords': '"地理位置关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472128.6251717}}], ('"天地源上唐府"', '"板楼"'): [{'src_id': '"天地源上唐府"', 'tgt_id': '"板楼"', 'weight': 7.0, 'description': '"天地源上唐府的建筑类型是板楼。"', 'keywords': '"建筑类型关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472128.6252031}}], ('"天地源上唐府"', '"高新区礼和小学"'): [{'src_id': '"天地源上唐府"', 'tgt_id': '"高新区礼和小学"', 'weight': 6.0, 'description': '"天地源上唐府属于高新区礼和小学的学区范围。"', 'keywords': '"学区关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472128.625225}}], ('"正圆花园"', '"高新区"'): [{'src_id': '"正圆花园"', 'tgt_id': '"高新区"', 'weight': 8.0, 'description': '"正圆花园位于高新区的地理区域内，两者之间存在地理位置关系。"', 'keywords': '"地理位置"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.230076}}], ('"唐淇路1208号"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"唐淇路1208号"', 'weight': 7.0, 'description': '"正圆花园的具体地址是唐淇路1208号，两者之间存在地址关系。"', 'keywords': '"地址关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301066}}], ('"板楼"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"板楼"', 'weight': 6.0, 'description': '"正圆花园的建筑类型是板楼，两者之间存在建筑类型关系。"', 'keywords': '"建筑类型"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301147}}], ('"209户"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"209户"', 'weight': 5.0, 'description': '"正圆花园共有209户住宅，两者之间存在数量关系。"', 'keywords': '"数量关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.230122}}], ('"11栋"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"11栋"', 'weight': 4.0, 'description': '"正圆花园共有11栋楼栋，两者之间存在 数量关系。"', 'keywords': '"数量关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301292}}], ('"25%"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"25%"', 'weight': 3.0, 'description': '"正圆花园的绿化率为25%，两者之间存在属性关系。"', 'keywords': '"属性关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301362}}], ('"3"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"3"', 'weight': 2.0, 'description': '"正圆花园的容积率为3，两者之间存在属性关系。"', 'keywords': '"属性关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301435}}], ('"商品房/使用权"', '"正圆花园"'): [{'src_id': '"正圆花园"', 'tgt_id': '"商品房/使用权"', 'weight': 1.0, 'description': '"正圆花园的交易权属为商品房/使用权，两者之间存在权属关系。"', 'keywords': '"权属关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301505}}], ('"正圆花园"', '"民水"'): [{'src_id': '"正圆花园"', 'tgt_id': '"民水"', 'weight': 9.0, 'description': '"正圆花园的用水类型为民水，两者之间存在用水类型关系。"', 'keywords': '"用水类型关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301574}}], ('"正圆花园"', '"民电"'): [{'src_id': '"正圆花园"', 'tgt_id': '"民电"', 'weight': 8.0, 'description': '"正圆花园的用电类型为民电，两者之间存在用电类型关系。"', 'keywords': '"用电类型关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301643}}], ('"正圆花园"', '"高新区中大附中"'): [{'src_id': '"正圆花园"', 'tgt_id': '"高新区中大附中"', 'weight': 7.0, 'description': '"高新区中大附中是正圆花园的初中学区，两者之间存在教育服务关系。"', 'keywords': '"教育服务关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.2301702}}], ('"正圆花园"', '"高新区中大附小后环校区"'): [{'src_id': '"正圆花园"', 'tgt_id': '"高新区中大附小后环校区"', 'weight': 6.0, 'description': '"高新区中大附小后环校区是正圆花园的小学区，两者之间存在教育服务关系。"', 'keywords': '"教育服务关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472163.230178}}], ('"正方世和苑"', '"高新区"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"高新区"', 'weight': 8.0, 'description': '"正方世和苑位于高新 区的泉星路81号。"', 'keywords': '"地理位置"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1066868}}], ('"塔楼"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"塔楼"', 'weight': 7.0, 'description': '"正方世和苑的建筑类型是塔楼。"', 'keywords': '"建筑类型"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067007}}], ('"196户"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"196 户"', 'weight': 6.0, 'description': '"正方世和苑共有196户住宅。"', 'keywords': '"住宅数量"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067088}}], ('"3栋"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"3栋"', 'weight': 5.0, 'description': '"正方世和苑共有3栋楼栋。"', 'keywords': '"楼栋数量"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067157}}], ('"35.19%"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"35.19%"', 'weight': 4.0, 'description': '"正方世和苑的绿化率为35.19%。"', 'keywords': '"绿化率"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067245}}], ('"3"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"3"', 'weight': 3.0, 'description': '"正方世和苑的容积率为3。"', 'keywords': '"容积率"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.106731}}], ('"商品房/房改房"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"商品房/房改房"', 'weight': 2.0, 'description': '"正方世和苑的交 易权属为商品房/房改房。"', 'keywords': '"房屋性质"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067376}}], ('"暂无信息"', '" 正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"暂无信息"', 'weight': 1.0, 'description': '"正方世和苑的建成年代信息暂无。"', 'keywords': '"建成年代"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.106744}}], ('"无供暖"', '"正方世和苑"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"无供暖"', 'weight': 1.0, 'description': '"正方世和苑的供暖类型是无供暖。"', 'keywords': '"供暖类型"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067502}}], ('"正方世和苑"', '"民水"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"民水"', 'weight': 1.0, 'description': '"正方世和苑的用水类型是民水。"', 'keywords': '"用水类型"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067557}}], ('"正方世和苑"', '"民电"'): [{'src_id': '"正方世和苑"', 'tgt_id': '"民电"', 'weight': 1.0, 'description': '"正方世和苑的用电类型是民电。"', 'keywords': '"用电类型"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472195.1067615}}], ('"金泰园"', '"高新区"'): [{'src_id': '"金泰园"', 'tgt_id': '"高新区"', 'weight': 8.0, 'description': '"金泰园位于高新区的地理区域内，反映了小区的地理位置。《"地理区域，地理位置"', 'keywords': '8', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8906322}}], ('"唐淇路1228号"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"唐淇路1228号"', 'weight': 7.0, 'description': '"金泰园的具体地址是唐淇路1228号，反映了小区的详细地址。《"详细地址，地理位置"', 'keywords': '7', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.890667}}], ('"板楼"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"板楼"', 'weight': 6.0, 'description': '"金泰园的建筑类型是板楼，反映了小区的建筑风格。《"建筑风格，建筑类型"', 'keywords': '6', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8906882}}], ('"90户"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"90户"', 'weight': 5.0, 'description': '"金泰园共有90户住宅，反映了小区的住宅数量。《"住宅数量，规模"', 'keywords': '5', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.890707}}], ('"7栋"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"7栋"', 'weight': 4.0, 'description': '"金泰园共有7栋楼栋，反映了小区的建筑密度。《"建筑密度，楼栋数量"', 'keywords': '4', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8907225}}], ('"34%"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"34%"', 'weight': 3.0, 'description': '"金泰园的绿化率是34%，反映了小区的生态环境。《"生态环境，绿化率"', 'keywords': '3', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8907378}}], ('"2.3"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"2.3"', 'weight': 2.0, 'description': '"金泰园的容积率是2.3，反映了小区的建筑密度。《"建筑密度，容积率"', 'keywords': '2', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8907535}}], ('"商品房"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"商品房"', 'weight': 1.0, 'description': '"金泰园的交易权属是商品房，反映了小区的产权性质。《"产权性质，交易权属"', 'keywords': '1', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8907692}}], ('"民水"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"民水"', 'weight': 10.0, 'description': '"金泰园的用水类型是民水，反映了小区的供水情况。《"供水情况，用水类型"', 'keywords': '10', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8907855}}], ('"民电"', '"金泰园"'): [{'src_id': '"金泰园"', 'tgt_id': '"民电"', 'weight': 9.0, 'description': '"金泰园的用电类型是民电，反映了 小区的供电情况。《"供电情况，用电类型"', 'keywords': '9', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8908014}}], ('"金泰园"', '"高新区中大附中"'): [{'src_id': '"金泰园"', 'tgt_id': '"高新区中大附中"', 'weight': 8.0, 'description': '"金泰园的初中学区是高新区中大附中，反映了小区的教育资源。《"教育资源，学区"', 'keywords': '8', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.890816}}], ('"金泰园"', '"高新区中大附小后环校区"'): [{'src_id': '"金泰园"', 'tgt_id': '"高新区中大附小后环校区"', 'weight': 7.0, 'description': '"金泰园的小学区是高新 区中大附小后环校区，反映了小区的教育资源。《"教育资源，学区"', 'keywords': '7', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472229.8908315}}], ('"唐家中学"', '"高新区"'): [{'src_id': '"唐家中学"', 'tgt_id': '"高新区"', 'weight': 8.0, 'description': '"唐家中学位于高新区。"', 'keywords': '"地理位置"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472248.3402247}}], ('"唐家中学教师楼"', '"高新区中大附中"'): [{'src_id': '"唐家中学教师楼"', 'tgt_id': '"高新区中大附中"', 'weight': 7.0, 'description': '"唐家中学教师楼的初中学区是高新区中大附中。"', 'keywords': '"学区关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472248.3402336}}], ('"唐家中学教师楼"', '"高新区唐家小学"'): [{'src_id': '"唐家中学教师楼"', 'tgt_id': '"高新区唐家小学"', 'weight': 7.0, 'description': '"唐家中学教师楼的小学区是高新区唐家小学。"', 'keywords': '"学区关系"', 'source_id': 'xuequ', 'metadata': {'created_at': 1740472248.3402407}}]})
# 41 40
# 131.6445770263672