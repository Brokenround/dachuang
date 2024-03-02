# -*- coding: utf-8 -*-

import math
import random
from collections import OrderedDict, namedtuple, defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from dee import transformer
from dee.ner_model import NERModel
from dee.dee_helper import merge_multi_events
from dre.utils import generate_label_info, generate_span_sent_id_list, DREProcessor

DocSpanInfo = namedtuple( #命名元组，可通过字段的名称来访问字段的值
    'DocSpanInfo', (
        'span_token_tup_list',  # [(span_token_id, ...), ...], num_spans 跨度的标记元组列表，每个元组表示一个跨度的标记
        'span_dranges_list',  # [[(sent_idx, char_s, char_e), ...], ...], num_spans 跨度的字符范围列表，每个列表表示一个跨度的字符范围
        'span_mention_range_list',  # [(mention_idx_s, mention_idx_e), ...], num_spans 跨度的提及范围列表，每个元组表示一个跨度的提及范围
        'mention_drange_list',  # [(sent_idx, char_s, char_e), ...], num_mentions 提及的字符范围列表，每个元组表示一个提及的字符范围
        'mention_type_list',  # [mention_type_id, ...], num_mentions 提及的类型列表，每个元素表示一个提及的类型
        'event_dag_info',  # event_idx -> field_idx -> pre_path -> cur_span_idx_set 事件DAG信息，映射事件索引到字段索引到前向路径到当前跨度索引集合
        'missed_sent_idx_list',  # index list of sentences where gold spans are not extracted 未提取到金标跨度的句子的索引列表
    )
)


# 定义一个空列表，用于存储实体标签和事件类型标签
# TOTAL_ENTITY_LABEL_LST = []
# TOTAL_EVENT_TYPE_LST = []
# 读取词汇表文件，将每一行的内容去掉首尾空白字符，并存储到VOCAB_LST列表中
# with open("/Users/yuan88/Documents/work/models/pretrained_models/torch_ckpt/bert_base_chinese/vocab.txt", "r") as f:
#     VOCAB_LST = [l.strip() for l in f]
#
# 读取标签文件，将每一行的内容去掉首尾空白字符，并存储到LABEL_LST列表中
# with open("/Users/yuan88/Documents/work/event_extraction/Doc2EDAG/ner_res/labels.txt", "r") as f:
#     LABEL_LST = [l.strip() for l in f]

# 定义一个函数，用于获取文档跨度信息列表
def get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=False):
    # 断言确保输入的文档标记类型列表和文档特征列表长度相等
    assert len(doc_token_types_list) == len(doc_fea_list) #若condition条件为假，则抛出AssertionError异常
    doc_span_info_list = []
    # 遍历文档标记类型列表和文档特征列表
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        # 将文档标记类型转换为二维列表
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # 如果使用金标跨度，则使用金标跨度。‘gold_span’通常是指在NLP中预先标注好的真实或理想的实体跨度信息。
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list    #存储跨度标记的列表。此处做引用，并未实际复制列表
            span_dranges_list = doc_fea.span_dranges_list        #存储跨度区间的列表。
        else:
            # BIEOS格式
            # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
            # BIEOS format
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info_for_bieos(doc_token_type_mat, doc_fea)
            if len(span_token_tup_list) == 0:
                # 如果未获得有效实体跨度结果，使用金标跨度以避免在较早迭代时崩溃
                # TODO: 考虑生成随机的负跨度
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list
                doc_token_type_mat = doc_fea.doc_token_labels.tolist()
        # 一个跨度可能有多个提及
        span_mention_range_list, mention_drange_list, mention_type_list = get_span_mention_info(
            span_dranges_list, doc_token_type_mat
        )

        # curr_idx = 0
        # entity_label_map = {}
        # for i, entity in enumerate(span_token_tup_list):
        #     entity = "".join([VOCAB_LST[i] for i in entity])
        #     label = LABEL_LST[mention_type_list[curr_idx]].split("-")[1]
        #     entity_label_map[entity] = label
        #     curr_idx += len(span_dranges_list[i])
        # TOTAL_ENTITY_LABEL_LST.append(entity_label_map)

        # 生成用于模型训练的事件解码DAG图
        # 有向无环图DAG是一个图结构，其中节点表示事件，边表示事件之间的依赖关系。
        event_dag_info, _, missed_sent_idx_list = doc_fea.generate_dag_info_for(span_token_tup_list, return_miss=True)

        # doc_span_info将包含事件提取所需的所有跨度级别信息
        doc_span_info = DocSpanInfo(
            span_token_tup_list, span_dranges_list, span_mention_range_list,
            mention_drange_list, mention_type_list,
            event_dag_info, missed_sent_idx_list,
        )

        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list


# 定义一个名为ReDEEModel的PyTorch模型类，用于文档级事件提取
class ReDEEModel(nn.Module):
    """文档级事件提取模型"""

    def __init__(self, config, event_type_fields_pairs, id_2_vocab=None, ner_model=None,
                 re_model=None, re_processor=None):
        super(ReDEEModel, self).__init__()
        # 注意，在分布式训练中，必须确保对于任何批次，所有参数都需要被使用

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        self.vocab_list = [val for _, val in id_2_vocab.items()]

        if ner_model is None:
            self.ner_model = NERModel(config)    # 根据给定的config参数创建一个NERModel类的实例
        else:
            self.ner_model = ner_model

        self.re_model = re_model
        self.re_processor = re_processor

        # 所有事件表
        self.event_tables = nn.ModuleList([
            EventTable(event_type, field_types, config.hidden_size)
            for event_type, field_types in self.event_type_fields_pairs
        ])

        # 句子位置指示器
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        if self.config.use_token_role:
            self.ment_type_encoder = MentionTypeEncoder(
                config.hidden_size, config.num_entity_labels, dropout=config.dropout
            )

        # 不同的注意力约减器
        if self.config.seq_reduce_type == 'AWA':    # 注意力权重方法（Attentive Weighted Attention）
            # 创建三个具有相同配置的 AttentiveReducer 对象。
            self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_mention_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        else:    # 另一种，池化pooling
            assert self.config.seq_reduce_type in {'MaxPooling', 'MeanPooling'}

        if self.config.use_doc_enc:
            # 获取每个提及和句子的文档级上下文信息
            self.doc_context_encoder = transformer.make_transformer_encoder(    # 编码器.Transformer 编码器非常适合处理序列数据，通过自注意力层来捕获序列中各个元素之间的复杂关系.
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout,
                raat=config.raat, entity_structure=config.entity_structure,
                num_structural_dependencies=config.num_relation + 3
            )

        if self.config.use_path_mem:
            # 获取每个跨度的字段特定和具有历史感知的信息
            self.field_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout,
                raat=config.raat_path_mem, entity_structure=config.entity_structure,
                num_structural_dependencies=config.num_relation + 3
            )
        # 10.2 优化（替代策略：使用一个通用的'None'表示）
        # self.none_emb_for_fields = nn.Parameter(torch.Tensor(1, config.hidden_size))

    def get_doc_span_mention_emb(self, doc_token_emb, doc_span_info):
        # 该函数目的从文档级的token嵌入中为每个mention生成一个嵌入表示，最终返回所有提及嵌入堆叠得到的张量 doc_token_emb .
        if len(doc_span_info.mention_drange_list) == 0:
            doc_mention_emb = None
            # 若mention信息长度为0，设置成none
        else:
            # 获取提及上下文嵌入
            # doc_mention_emb = torch.cat([
            #     # doc_token_emb[sent_idx, char_s:char_e, :].sum(dim=0, keepdim=True)
            #     doc_token_emb[sent_idx, char_s:char_e, :].max(dim=0, keepdim=True)[0]
            #     for sent_idx, char_s, char_e in doc_span_info.mention_drange_list
            # ])
            mention_emb_list = []
            for sent_idx, char_s, char_e in doc_span_info.mention_drange_list:
                mention_token_emb = doc_token_emb[sent_idx, char_s: char_e, :]  # [num_mention_tokens, hidden_size]
               # 根据叙序列约减类型对token嵌入进行约减，获得提及的单个向量表示.
                if self.config.seq_reduce_type == 'AWA':
                    mention_emb = self.span_token_reducer(mention_token_emb)  # [hidden_size]
                elif self.config.seq_reduce_type == 'MaxPooling':
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == 'MeanPooling':
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                mention_emb_list.append(mention_emb)
            doc_mention_emb = torch.stack(mention_emb_list, dim=0)

            # 添加句子位置嵌入
            mention_sent_id_list = [drange[0] for drange in doc_span_info.mention_drange_list]
            doc_mention_emb = self.sent_pos_encoder(doc_mention_emb, sent_pos_ids=mention_sent_id_list)

            if self.config.use_token_role:
                # 获取提及类型嵌入
                doc_mention_emb = self.ment_type_encoder(doc_mention_emb, doc_span_info.mention_type_list)

        return doc_mention_emb

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # 该函数旨在从批量的 NER 的token嵌入中生成句子级别的嵌入.
        # ner_token_masks 是与 ner_token_emb 对应的掩码张量，指示哪些token是有效的，哪些应被忽略.
        # 从[ner_batch_size, sent_len, hidden_size]变成[ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=total_sent_pos_ids)    # 将句子位置信息编码到每个句子的嵌入中

        return total_sent_emb

    def get_doc_span_sent_context(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info, re_span_rep,
                                  structure_mask=None):
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_span_info)

        # 仅考虑实际的句子
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[:doc_fea.valid_sent_num, :]

        span_context_list = []

        if doc_mention_emb is None:
            if self.config.use_doc_enc:
                # 考虑RAAT
                doc_sent_context = self.doc_context_encoder(
                    doc_sent_emb.unsqueeze(0), None, structure_mask=None).squeeze(0)
            else:
                doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)

            if self.config.use_doc_enc:
                # Size([1, num_mentions + num_valid_sents, hidden_size])
                total_ment_sent_emb = torch.cat([doc_mention_emb, doc_sent_emb], dim=0).unsqueeze(0)

                # size = [num_mentions+num_valid_sents, hidden_size]
                # 这里我们不需要掩码
                # 考虑 RAAT
                total_ment_sent_context = self.doc_context_encoder(total_ment_sent_emb, None,
                                                                   structure_mask=structure_mask).squeeze(0)
                # 收集跨度上下文
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_context = total_ment_sent_context[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_context, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_context.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

                    span_context_list.append(span_context)

                # 收集句子上下文
                doc_sent_context = total_ment_sent_context[num_mentions:, :]
            else:
                # 收集跨度上下文
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_emb = doc_mention_emb[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size is [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_emb, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_emb.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_emb.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                    span_context_list.append(span_context)

                # 收集句子上下文
                doc_sent_context = doc_sent_emb

        # 考虑合并span_context_rep和re_span_rep
        if len(span_context_list) > 0 and re_span_rep is not None:
            for i in range(len(span_context_list)):
                if i >= self.config.max_ent_cnt:
                    continue
                # TODO: consider more fusion method.
                span_context_list[i] += re_span_rep[i]

        return span_context_list, doc_sent_context

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(sent_context_emb=sent_context_emb)  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]

        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels, device=device, dtype=torch.long, requires_grad=False
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(doc_event_logps, doc_event_labels, reduction='sum')
            return doc_event_cls_loss
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_field_cls_info(self, event_idx, field_idx, batch_span_emb,
                           batch_span_label=None, train_flag=True):
        batch_span_logp = self.get_field_pred_logp(event_idx, field_idx, batch_span_emb)

        if train_flag:
            assert batch_span_label is not None
            device = batch_span_logp.device
            data_type = batch_span_logp.dtype
            # 防止过多的FP
            class_weight = torch.tensor(
                [self.config.neg_field_loss_scaling, 1.0], device=device, dtype=data_type, requires_grad=False
            )
            field_cls_loss = F.nll_loss(batch_span_logp, batch_span_label, weight=class_weight, reduction='sum')
            return field_cls_loss, batch_span_logp
        else:
            span_pred_list = batch_span_logp.argmax(dim=-1).tolist()
            return span_pred_list, batch_span_logp

    def get_field_pred_logp(self, event_idx, field_idx, batch_span_emb, include_prob=False):
        event_table = self.event_tables[event_idx]
        batch_span_logp = event_table(batch_span_emb=batch_span_emb, field_idx=field_idx)

        if include_prob:
            # 用于决策采样，不在计算图内
            batch_span_prob = batch_span_logp.detach().exp()
            return batch_span_logp, batch_span_prob
        else:
            return batch_span_logp

    def get_none_span_context(self, init_tensor):
        none_span_context = torch.zeros(
            1, self.config.hidden_size,
            device=init_tensor.device, dtype=init_tensor.dtype, requires_grad=False
        )
        return none_span_context

    def conduct_field_level_reasoning(self, event_idx, field_idx, prev_decode_context, batch_span_context,
                                      structure_mask_2=None, re_label=None, expand_span_id_list=None,
                                      span_sent_id_list=None):
        event_table = self.event_tables[event_idx]
        field_query = event_table.field_queries[field_idx]
        num_spans = batch_span_context.size(0)
        # 使模型知道哪个字段
        batch_cand_emb = batch_span_context + field_query
        batch_cand_emb = torch.cat([batch_cand_emb, event_table.none_emb_for_fields[field_idx]])
        num_spans += 1
        if self.config.use_path_mem:
            # [1, num_spans + valid_sent_num, hidden_size]
            total_cand_emb = torch.cat([batch_cand_emb, prev_decode_context], dim=0).unsqueeze(0)
            # 使用transformer进行推理
            if structure_mask_2 is not None:
                structure_mask_2 = DREProcessor.expand_structure_mask(structure_mask_2, expand_span_id_list, re_label,
                                                                      num_spans - 1, span_sent_id_list,
                                                                      head_center=self.config.head_center,
                                                                      num_relation=self.config.num_relation)
            total_cand_emb = self.field_context_encoder(total_cand_emb, None, structure_mask=structure_mask_2).squeeze(
                0)
            batch_cand_emb = total_cand_emb[:num_spans, :]
        # TODO: what if reasoning over reasoning context
        return batch_cand_emb, prev_decode_context

    def get_field_mle_loss_list(self, doc_sent_context, batch_span_context, event_idx,
                                field_idx2pre_path2cur_span_idx_set, structure_mask_2=None,
                                re_label=None, span_sent_id_list=None):
        field_mle_loss_list = []
        num_fields = self.event_tables[event_idx].num_fields
        num_spans = batch_span_context.size(0)
        prev_path2prev_decode_context = {
            (): doc_sent_context
        }

        for field_idx in range(num_fields):
            prev_path2cur_span_idx_set = field_idx2pre_path2cur_span_idx_set[field_idx]
            for prev_path, cur_span_idx_set in prev_path2cur_span_idx_set.items():
                if prev_path not in prev_path2prev_decode_context:
                    # 注意，当None和valid_span共存时，训练过程中忽略None路径
                    continue
                # 获取解码上下文
                prev_decode_context = prev_path2prev_decode_context[prev_path]
                # 在这个字段上进行推理
                batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                    event_idx, field_idx, prev_decode_context, batch_span_context, structure_mask_2, re_label,
                    prev_path, span_sent_id_list
                )
                num_spans += 1
                # 为候选跨度准备标签
                batch_span_label = get_batch_span_label(
                    num_spans, cur_span_idx_set, batch_span_context.device
                )
                # 计算损失
                cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                    event_idx, field_idx, batch_cand_emb,
                    batch_span_label=batch_span_label, train_flag=True
                )

                field_mle_loss_list.append(cur_field_cls_loss)
                num_spans -= 1

                # cur_span_idx_set需要确保至少有一个元素，即None
                for span_idx in cur_span_idx_set:
                    # Teacher-forcing Style Training
                    if span_idx is None:
                        span_context = self.event_tables[event_idx].field_queries[field_idx]
                    else:
                        # TODO: add either batch_cand_emb or batch_span_context to the memory tensor
                        span_context = batch_cand_emb[span_idx].unsqueeze(0)

                    cur_path = prev_path + (span_idx,)
                    if self.config.use_path_mem:
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context
                    else:
                        prev_path2prev_decode_context[cur_path] = prev_decode_context

        return field_mle_loss_list

    def get_loss_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info, re_span_rep,
                        structure_mask=None, structure_mask_2=None, re_label=None, span_sent_id_list=None):
        # 获取文档的跨度和句子上下文
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info, re_span_rep, structure_mask=structure_mask
        )
        # 如果没有有效的跨度，抛出异常
        if len(span_context_list) == 0:
            raise Exception('Error: doc_fea.ex_idx {} does not have valid span'.format(doc_fea.ex_idx))

        # 将跨度上下文拼接成一个张量
        batch_span_context = torch.cat(span_context_list, dim=0)
        num_spans = len(span_context_list)
        # 获取事件与字段的关系信息
        event_idx2field_idx2pre_path2cur_span_idx_set = doc_span_info.event_dag_info

        # 1. 获取事件类型分类损失
        event_cls_loss = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=True)

        # 2. 对于每种事件类型，获取字段分类损失
        # 注意，包括内存张量在计算图中可以提高性能（F1>1）
        all_field_loss_list = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            if event_label == 0:
                # 将所有跨度都视为该事件的无效参数，
                # 因为我们需要使用所有参数来支持分布式训练
                prev_decode_context = doc_sent_context
                num_fields = self.event_tables[event_idx].num_fields
                expand_span_id_list = []
                for field_idx in range(num_fields):
                    # 对该字段进行推理
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, batch_span_context, structure_mask_2, re_label,
                        expand_span_id_list
                    )
                    num_spans += 1
                    # 为候选跨度准备标签
                    batch_span_label = get_batch_span_label(
                        num_spans, (None,), batch_span_context.device
                    )
                    # 计算字段损失
                    cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb,
                        batch_span_label=batch_span_label, train_flag=True
                    )
                    # 更新内存张量
                    span_context = self.event_tables[event_idx].field_queries[field_idx]
                    if self.config.use_path_mem:
                        prev_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        expand_span_id_list.append(None)

                    all_field_loss_list.append(cur_field_cls_loss)
                    num_spans -= 1
            else:
                field_idx2pre_path2cur_span_idx_set = event_idx2field_idx2pre_path2cur_span_idx_set[event_idx]
                field_loss_list = self.get_field_mle_loss_list(
                    doc_sent_context, batch_span_context, event_idx, field_idx2pre_path2cur_span_idx_set,
                    structure_mask_2, re_label, span_sent_id_list
                )
                all_field_loss_list += field_loss_list

        total_event_loss = event_cls_loss + sum(all_field_loss_list)
        return total_event_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list, re_loss):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list)) + \
            re_loss * self.config.re_loss_ratio

    def get_eval_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info, re_span_rep, re_pair_cands=None,
                        structure_mask=None, structure_mask_2=None, re_label=None, span_sent_id_list=None):
        # 获取文档的跨度上下文和句子上下文
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info, re_span_rep, structure_mask=structure_mask)
        # 如果没有有效的跨度，返回默认结果
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            event_idx2event_decode_paths = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2event_decode_paths

        batch_span_context = torch.cat(span_context_list, dim=0)

        # 1. 获取事件类型预测
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)
        # TOTAL_EVENT_TYPE_LST.append(event_pred_list)
        # 2. 对于每种事件类型，获取字段预测
        # 以下映射都使用列表索引实现
        event_idx2event_decode_paths = []
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2event_decode_paths.append(None)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields

            prev_path2prev_decode_context = {(): doc_sent_context}
            last_field_paths = [()]  # 仅记录最后一个字段的路径
            for field_idx in range(num_fields):
                cur_paths = []
                for prev_path in last_field_paths:  # 遍历所有先前的解码路径
                    # 获取解码上下文
                    prev_decode_context = prev_path2prev_decode_context[prev_path]
                    # 在这个字段上进行推理
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, batch_span_context, structure_mask_2,
                        re_label, prev_path, span_sent_id_list
                    )

                    # 获取所有跨度的字段预测
                    span_pred_list, _ = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb, train_flag=False
                    )

                    # 准备下一个字段使用的跨度索引
                    cur_span_idx_list = []
                    for span_idx, span_pred in enumerate(span_pred_list):
                        if span_idx == len(span_pred_list) - 1 and span_pred == 1:
                            cur_span_idx_list.append(None)
                        elif span_pred == 1:
                            cur_span_idx_list.append(span_idx)
                    if len(cur_span_idx_list) == 0:
                        # 这个字段的所有跨度对于此字段来说都是无效的，只需选择'Unknown'标记
                        cur_span_idx_list.append(None)

                    for span_idx in cur_span_idx_list:
                        if span_idx is None:
                            span_context = self.event_tables[event_idx].field_queries[field_idx]
                            # span_context = none_span_context
                        else:
                            span_context = batch_cand_emb[span_idx].unsqueeze(0)

                        cur_path = prev_path + (span_idx,)
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        cur_paths.append(cur_path)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context

                # 更新解码路径
                last_field_paths = cur_paths

            obj_idx2field_idx2token_tup = []
            for decode_path in last_field_paths:
                assert len(decode_path) == num_fields
                field_idx2token_tup = []
                for span_idx in decode_path:
                    if span_idx is None:
                        token_tup = None
                    else:
                        token_tup = doc_span_info.span_token_tup_list[span_idx]

                    field_idx2token_tup.append(token_tup)
                obj_idx2field_idx2token_tup.append(field_idx2token_tup)

            event_idx2event_decode_paths.append(last_field_paths)
            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        if self.config.with_post_process:
            event_idx2obj_idx2field_idx2token_tup = merge_multi_events(event_idx2obj_idx2field_idx2token_tup,
                                                                       self.vocab_list)
        # 输出关系提取对信息
        if re_pair_cands is not None:
            span_token_tup_list = doc_span_info.span_token_tup_list
            for pair_cand in re_pair_cands:
                h_info, t_info = pair_cand[0], pair_cand[1]
                h_info[0] = "".join([self.vocab_list[i] for i in list(span_token_tup_list[h_info[0]])])
                t_info[0] = "".join([self.vocab_list[i] for i in list(span_token_tup_list[t_info[0]])])

        # 前三项是用于度量计算，后两项是用于案例研究
        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2event_decode_paths, re_pair_cands

    def adjust_token_label(self, doc_token_labels_list):
        if self.config.use_token_role:  # 不使用详细的标记
            return doc_token_labels_list
        else:
            adj_doc_token_labels_list = []
            for doc_token_labels in doc_token_labels_list:
                entity_begin_mask = doc_token_labels % 2 == 1
                entity_inside_mask = (doc_token_labels != 0) & (doc_token_labels % 2 == 0)
                adj_doc_token_labels = doc_token_labels.masked_fill(entity_begin_mask, 1)
                adj_doc_token_labels = adj_doc_token_labels.masked_fill(entity_inside_mask, 2)

                adj_doc_token_labels_list.append(adj_doc_token_labels)
            return adj_doc_token_labels_list

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = self.adjust_token_label(doc_batch_dict[label_key])
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']

        # 将doc_batch转换为sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # 获取命名实体识别（NER）输出
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )

        if use_gold_span:  # 绝对使用金标记信息
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # 获取句子嵌入（考虑在下一步中的位置编码）
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # 以下都是张量列表
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx + 1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def forward(self, doc_batch_dict, doc_features,
                train_flag=True, use_gold_span=False, teacher_prob=1,
                event_idx2entity_idx2field_idx=None, heuristic_type=None):
        """
        模型的前向传播函数，负责处理文档级别的事件提取任务。该函数接受输入文档信息，包括词元、特征等，
        并根据训练标志、使用金标跨度标志、计划采样概率等进行前向传播。在训练时，根据计划采样的概率
        使用金标跨度进行训练，否则使用模型预测的跨度。该函数还包括关系提取模块，根据配置使用RAAT
        (Relation-Aware Attentive Transition)模块。在推理阶段，可以选择使用启发式方法对事件和字段进行解码。

        参数：
        - doc_batch_dict: 包含文档信息的字典，例如词元、特征等。
        - doc_features: 包含文档特征的对象。
        - train_flag: 是否处于训练模式。
        - use_gold_span: 是否使用金标跨度进行训练。
        - teacher_prob: 计划采样的概率，逐渐减小。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。
        - heuristic_type: 启发式解码的类型。

        返回：
        - mix_loss: 训练时的混合损失（事件分类损失和事件结构损失）。
        或
        - eval_results: 推理时的结果列表。
        """

        # 使用计划采样逐渐过渡到预测的实体跨度
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob将在外部逐渐减小
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False

        # 获取文档token-level局部上下文信息
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # 获取文档特征对象
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # 获取用于事件提取的文档跨度级别的信息
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)

        # 添加关系提取模块
        re_pair_cands_list = None
        device = doc_batch_dict["doc_token_ids"][0].device
        structure_mask_list = [None for _ in range(len(doc_span_info_list))]
        structure_mask_2_list = [None for _ in range(len(doc_span_info_list))]
        span_sent_id_lists = [None for _ in range(len(doc_span_info_list))]
        re_label_list = [None for _ in range(len(doc_span_info_list))]

        if self.re_model is not None:
            re_features = self.re_processor.convert_ner_result_to_dre_feature(doc_fea_list, doc_span_info_list,
                                                                              self.config.use_bert,
                                                                              label_map=self.re_processor.label_map)
            re_inputs = self.re_processor.prepare_input_tensor_for_re(re_features, device)
            re_outputs = self.re_model(**re_inputs)
            re_loss, re_logits, re_span_rep = re_outputs
            re_span_rep_list = [re_span_rep[i] for i in range(len(ex_idx_list))]

            re_pair_cands_list = self.re_processor.output_relation_pair_candidates(re_inputs["label_mask"],
                                                                                   re_logits,
                                                                                   doc_span_info_list)

            # 类似于NER模块，使用计划采样逐渐过渡到预测的关系对。
            # 向RAAT模块提供structure_mask。
            if self.config.raat:
                structure_mask_list = []
                if train_flag:
                    for i, (doc_span_info, doc_token_emb) in enumerate(zip(doc_span_info_list, doc_token_emb_list)):
                        label_list = generate_label_info(doc_span_info.event_dag_info, max_ent_cnt=42,
                                                         label_map=self.re_processor.label_map)
                        if not use_gold_span:
                            label_list = re_pair_cands_list[i]
                        structure_mask = DREProcessor.generate_structural_attentive_feature(
                            doc_span_info.span_mention_range_list, doc_span_info.mention_drange_list, label_list,
                            len(doc_token_emb), device=device, raat_id=1, head_center=self.config.head_center,
                            num_relation=self.config.num_relation)
                        structure_mask_list.append(structure_mask)
                else:
                    for doc_span_info, doc_token_emb, re_pair_cands in \
                            zip(doc_span_info_list, doc_token_emb_list, re_pair_cands_list):
                        structure_mask = DREProcessor.generate_structural_attentive_feature(
                            doc_span_info.span_mention_range_list, doc_span_info.mention_drange_list, re_pair_cands,
                            len(doc_token_emb), device=device, raat_id=1, head_center=self.config.head_center,
                            num_relation=self.config.num_relation)
                        structure_mask_list.append(structure_mask)
            # else:
            #     structure_mask_list = [None for _ in range(len(doc_span_info_list))]

            if self.config.raat_path_mem:
                structure_mask_2_list = []
                span_sent_id_lists = []
                # re_label_list用于根据EDAG的增长扩展现有structure_mask_2。
                re_label_list = []
                if train_flag:
                    for i, (doc_span_info, doc_token_emb) in enumerate(zip(doc_span_info_list, doc_token_emb_list)):
                        label_list = generate_label_info(doc_span_info.event_dag_info, max_ent_cnt=42,
                                                         label_map=self.re_processor.label_map)
                        if not use_gold_span:
                            label_list = re_pair_cands_list[i]
                        span_sent_id_list = generate_span_sent_id_list(doc_span_info.mention_drange_list,
                                                                       doc_span_info.span_mention_range_list)
                        re_label_list.append(label_list)
                        dummy_span_mention_range_list = [(i, i + 1) for i in
                                                         range(len(doc_span_info.span_mention_range_list))]
                        # 考虑在EDAG部分添加一个“None”实体，因此structure_mask的大小应该
                        # 要大一点。
                        structure_mask = DREProcessor.generate_structural_attentive_feature(
                            dummy_span_mention_range_list, span_sent_id_list, label_list, len(doc_token_emb) + 1,
                            device=device, raat_id=2, head_center=self.config.head_center,
                            num_relation=self.config.num_relation)
                        structure_mask_2_list.append(structure_mask)
                        span_sent_id_lists.append(span_sent_id_list)
                else:
                    re_label_list = re_pair_cands_list
                    for doc_span_info, doc_token_emb, re_pair_cands in \
                            zip(doc_span_info_list, doc_token_emb_list, re_pair_cands_list):
                        # 构建structure_mask_2
                        span_sent_id_list = generate_span_sent_id_list(doc_span_info.mention_drange_list,
                                                                       doc_span_info.span_mention_range_list)
                        dummy_span_mention_range_list = [(i, i + 1) for i in
                                                         range(len(doc_span_info.span_mention_range_list))]
                        # 考虑在EDAG部分添加一个“None”实体，因此structure_mask的大小应该
                        # 要大一点。
                        structure_mask = DREProcessor.generate_structural_attentive_feature(
                            dummy_span_mention_range_list, span_sent_id_list, re_pair_cands, len(doc_token_emb) + 1,
                            device=device, raat_id=2, head_center=self.config.head_center,
                            num_relation=self.config.num_relation)
                        structure_mask_2_list.append(structure_mask)
                        span_sent_id_lists.append(span_sent_id_list)
            # else:
            #     structure_mask_2_list = [None for _ in range(len(doc_span_info_list))]
            #     span_sent_id_lists = [None for _ in range(len(doc_span_info_list))]
            #     re_label_list = [None for _ in range(len(doc_span_info_list))]
        else:
            re_loss = None
            re_span_rep_list = [None for _ in range(len(ex_idx_list))]

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        re_span_rep_list[batch_idx],
                        structure_mask_list[batch_idx],
                        structure_mask_2_list[batch_idx],
                        re_label_list[batch_idx],
                        span_sent_id_lists[batch_idx]
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list, re_loss)

            return mix_loss
        else:
            # 返回一个列表对象可能不受torch.nn.parallel.DataParallel的支持
            # 确保在单GPU模式下运行
            eval_results = []

            if heuristic_type is None:
                for batch_idx, ex_idx in enumerate(ex_idx_list):
                    if re_pair_cands_list is None:
                        re_pair_cands = None
                    else:
                        re_pair_cands = re_pair_cands_list[batch_idx]
                    eval_results.append(
                        self.get_eval_on_doc(
                            doc_token_emb_list[batch_idx],
                            doc_sent_emb_list[batch_idx],
                            doc_fea_list[batch_idx],
                            doc_span_info_list[batch_idx],
                            re_span_rep_list[batch_idx],
                            re_pair_cands,
                            structure_mask_list[batch_idx],
                            structure_mask_2_list[batch_idx],
                            re_label_list[batch_idx],
                            span_sent_id_lists[batch_idx]
                        )
                    )
            else:
                assert event_idx2entity_idx2field_idx is not None
                for batch_idx, ex_idx in enumerate(ex_idx_list):
                    eval_results.append(
                        self.heuristic_decode_on_doc(
                            doc_token_emb_list[batch_idx],
                            doc_sent_emb_list[batch_idx],
                            doc_fea_list[batch_idx],
                            doc_span_info_list[batch_idx],
                            event_idx2entity_idx2field_idx,
                            heuristic_type=heuristic_type,
                        )
                    )
            # with open("/Users/yuan88/Documents/work/datasets/event_extraction/ee_test/optim_v1/entities.json", "w") as f:
            #     json.dump(TOTAL_ENTITY_LABEL_LST, f, ensure_ascii=False, indent=2)
            # with open("/Users/yuan88/Documents/work/datasets/event_extraction/ee_test/optim_v1/event_type_pred.json", "w") as f:
            #     json.dump(TOTAL_EVENT_TYPE_LST, f, ensure_ascii=False, indent=2)

            return eval_results

    def heuristic_decode_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info,
                                event_idx2entity_idx2field_idx, heuristic_type='GreedyDec'):
        """
        启发式解码函数，用于推理时基于模型输出结果的事件和字段解码。支持不同的启发式类型，包括
        GreedyDec（贪婪解码）和ProductDec（联合概率最大化解码）。

        参数：
        - doc_token_emb: 文档词元嵌入。
        - doc_sent_emb: 文档句子嵌入。
        - doc_fea: 文档特征。
        - doc_span_info: 文档跨度信息。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。
        - heuristic_type: 启发式解码的类型。

        返回：
        - 解码结果的一系列元组，包括事件预测、字段预测和跨度信息。
        """

        support_heuristic_types = ['GreedyDec', 'ProductDec']
        if heuristic_type not in support_heuristic_types:
            raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                heuristic_type, str(support_heuristic_types)
            ))

        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
        )

        span_token_tup_list = doc_span_info.span_token_tup_list
        span_mention_range_list = doc_span_info.span_mention_range_list
        mention_drange_list = doc_span_info.mention_drange_list
        mention_type_list = doc_span_info.mention_type_list
        # 启发式解码策略将在这些跨度候选上起作用
        event_idx2field_idx2span_token_tup2dranges = self.get_event_field_span_candidates(
            span_token_tup_list, span_mention_range_list, mention_drange_list,
            mention_type_list, event_idx2entity_idx2field_idx,
        )

        # 如果没有提取的跨度，直接返回
        if len(span_token_tup_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []  # 此项将与地面真实表内容进行比较
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2field_idx2span_token_tup2dranges

        # 1. 获取事件类型预测作为基于模型的方法
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)

        # 2. 对于每个事件类型，获取字段预测
        # 从现在开始，使用启发式推理获取字段的标记
        # 以下映射都是使用列表索引实现的
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields
            field_idx2span_token_tup2dranges = event_idx2field_idx2span_token_tup2dranges[event_idx]

            obj_idx2field_idx2token_tup = [[]]  # 至少会追加一条解码路径
            for field_idx in range(num_fields):
                if heuristic_type == support_heuristic_types[0]:
                    obj_idx2field_idx2token_tup = append_top_span_only(
                        obj_idx2field_idx2token_tup, field_idx, field_idx2span_token_tup2dranges
                    )
                elif heuristic_type == support_heuristic_types[1]:
                    obj_idx2field_idx2token_tup = append_all_spans(
                        obj_idx2field_idx2token_tup, field_idx, field_idx2span_token_tup2dranges
                    )
                else:
                    raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                        heuristic_type, str(support_heuristic_types)
                    ))

            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2field_idx2span_token_tup2dranges

    def get_event_field_span_candidates(self, span_token_tup_list, span_mention_range_list,
                                        mention_drange_list, mention_type_list, event_idx2entity_idx2field_idx):
        """
        获取事件字段的候选跨度信息。根据提供的跨度信息、实体-字段映射，整理出每个事件字段的候选跨度。

        参数：
        - span_token_tup_list: 词元跨度的元组列表。
        - span_mention_range_list: 提及跨度的起止索引列表。
        - mention_drange_list: 提及的文档级跨度列表。
        - mention_type_list: 提及的类型列表。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。

        返回：
        - event_idx2field_idx2span_token_tup2dranges: 事件索引到字段索引到跨度元组到文档级跨度列表的映射。
        """

        # 获取提及索引到词元跨度索引的映射
        mention_span_idx_list = []
        for span_idx, (ment_idx_s, ment_idx_e) in enumerate(span_mention_range_list):
            mention_span_idx_list.extend([span_idx] * (ment_idx_e - ment_idx_s))
        assert len(mention_span_idx_list) == len(mention_drange_list)

        event_idx2field_idx2span_token_tup2dranges = {}
        for event_idx, (event_type, field_types) in enumerate(self.event_type_fields_pairs):
            # 获取预定义实体索引到字段索引的映射
            gold_entity_idx2field_idx = event_idx2entity_idx2field_idx[event_idx]

            # 存储此文档的字段候选跨度信息
            field_idx2span_token_tup2dranges = {}
            for field_idx, _ in enumerate(field_types):
                field_idx2span_token_tup2dranges[field_idx] = {}

            # 根据提及类型，聚合字段候选跨度信息
            for ment_idx, (ment_drange, ment_entity_idx) in enumerate(zip(mention_drange_list, mention_type_list)):
                if ment_entity_idx not in gold_entity_idx2field_idx:
                    continue
                ment_field_idx = gold_entity_idx2field_idx[ment_entity_idx]
                if ment_field_idx is None:
                    continue

                ment_span_idx = mention_span_idx_list[ment_idx]
                span_token_tup = span_token_tup_list[ment_span_idx]

                # 由于它是字典，因此对键的所有修改将在原始字典中生效
                cur_span_token_tup2dranges = field_idx2span_token_tup2dranges[ment_field_idx]
                if span_token_tup not in cur_span_token_tup2dranges:
                    cur_span_token_tup2dranges[span_token_tup] = []
                cur_span_token_tup2dranges[span_token_tup].append(ment_drange)

            event_idx2field_idx2span_token_tup2dranges[event_idx] = field_idx2span_token_tup2dranges

        return event_idx2field_idx2span_token_tup2dranges


def append_top_span_only(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    """
    仅追加每个字段的最佳跨度。根据字段候选跨度字典，将每个路径的最佳跨度追加到路径末尾。

    参数：
    - last_token_path_list: 上一个词元路径的列表。
    - field_idx: 字段索引。
    - field_idx2span_token_tup2dranges: 字段索引到词元跨度元组到文档级跨度列表的映射。

    返回：
    - new_token_path_list: 更新后的词元路径列表。
    """

    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]
    token_min_drange_list = [
        (token_tup, dranges[0]) for token_tup, dranges in span_token_tup2dranges.items()
    ]
    token_min_drange_list.sort(key=lambda x: x[1])

    for last_token_path in last_token_path_list:
        new_token_path = list(last_token_path)
        if len(token_min_drange_list) == 0:
            new_token_path.append(None)
        else:
            token_tup = token_min_drange_list[0][0]
            new_token_path.append(token_tup)

        new_token_path_list.append(new_token_path)

    return new_token_path_list


def append_all_spans(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    """
    追加字段的所有候选跨度。根据字段候选跨度字典，将每个路径追加字段的所有候选跨度。

    参数：
    - last_token_path_list: 上一个词元路径的列表。
    - field_idx: 字段索引。
    - field_idx2span_token_tup2dranges: 字段索引到词元跨度元组到文档级跨度列表的映射。

    返回：
    - new_token_path_list: 更新后的词元路径列表。
    """

    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

    for last_token_path in last_token_path_list:
        for token_tup in span_token_tup2dranges.keys():
            new_token_path = list(last_token_path)
            new_token_path.append(token_tup)
            new_token_path_list.append(new_token_path)

        if len(span_token_tup2dranges) == 0:  # ensure every last path will be extended
            new_token_path = list(last_token_path)
            new_token_path.append(None)
            new_token_path_list.append(new_token_path)

    return new_token_path_list


class AttentiveReducer(nn.Module):
    """
    注意力机制约减器模块。该模块接受词元嵌入，通过自注意力机制约减序列信息。

    参数：
    - hidden_size: 隐藏层大小。
    - dropout: 随机丢弃的概率。

    返回：
    - 无返回值。
    """

    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        """
        前向传播函数。根据输入的词元嵌入，通过自注意力机制约减序列信息。

        参数：
        - batch_token_emb: 大小为[*, seq_len, hidden_size]的词元嵌入。
        - masks: 大小为[*, seq_len]的掩码，1表示正常，0表示填充。

        返回：
        - batch_att_emb: 大小为[*, 1, hidden_size]的注意力约减嵌入。
        或
        - batch_att_emb.squeeze(-2): 大小为[*, hidden_size]的注意力约减嵌入，如果keepdim为False。
        """
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        """
        返回模块的额外字符串表示，用于打印模型参数。

        返回：
        - 'hidden_size={}, att_norm={}'：隐藏层大小和注意力规范化参数。
        """
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)


class SentencePosEncoder(nn.Module):
    """
    句子位置编码器模块。该模块通过对句子位置进行编码，将句子位置信息融合到输入的元素嵌入中。

    参数：
    - hidden_size: 隐藏层大小。
    - max_sent_num: 最大句子数，默认为100。
    - dropout: 随机丢弃的概率。

    返回：
    - 无返回值。
    """

    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        """
        前向传播函数。根据输入的元素嵌入和句子位置编码，返回融合了位置信息的新嵌入。

        参数：
        - batch_elem_emb: 元素嵌入，大小为[*, seq_len, hidden_size]。
        - sent_pos_ids: 句子位置标识，大小为[*, seq_len]，默认为None。

        返回：
        - out: 融合了位置信息的新嵌入，大小为[*, seq_len, hidden_size]。
        """
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypeEncoder(nn.Module):
    """
    提及类型编码器模块。该模块通过对提及类型进行编码，将提及类型信息融合到输入的提及嵌入中。

    参数：
    - hidden_size: 隐藏层大小。
    - num_ment_types: 提及类型数。
    - dropout: 随机丢弃的概率。

    返回：
    - 无返回值。
    """

    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        """
        前向传播函数。根据输入的提及嵌入和提及类型编码，返回融合了类型信息的新嵌入。

        参数：
        - batch_mention_emb: 提及嵌入，大小为[*, seq_len, hidden_size]。
        - mention_type_ids: 提及类型标识，大小为[*, seq_len]。

        返回：
        - out: 融合了类型信息的新嵌入，大小为[*, seq_len, hidden_size]。
        """
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out


class EventTable(nn.Module):
    """
    事件表模块。该模块包含事件类型和相关字段，用于对文档进行事件触发预测和字段触发预测。

    参数：
    - event_type: 事件类型。
    - field_types: 字段类型列表。
    - hidden_size: 隐藏层大小。

    返回：
    - 无返回值。
    """

    def __init__(self, event_type, field_types, hidden_size):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
        self.field_cls_list = nn.ModuleList(
            # 0: NA, 1: 触发此字段
            [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        )

        # 用于聚合句子和跨度嵌入
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # 用于不包含任何有效跨度的字段
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        # strategy 10.2 to represent None
        self.none_emb_for_fields = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)
        for fq in self.none_emb_for_fields:
            fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        """
        前向传播函数。根据输入的句子或跨度嵌入，进行事件触发预测或字段触发预测。

        参数：
        - sent_context_emb: 句子上下文嵌入，大小为[num_spans+num_sents, hidden_size]。
        - batch_span_emb: 跨度嵌入，大小为[batch_size, hidden_size]或[hidden_size]。
        - field_idx: 字段索引。

        返回：
        - doc_pred_logp: 事件触发预测对数概率，大小为[1, 2]。
        或
        - span_pred_logp: 字段触发预测对数概率，大小为[batch_size, 2]。
        """
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp

    def extra_repr(self):
        """
        返回模块的额外字符串表示，用于打印模型参数。

        返回：
        - 'event_type={}, num_fields={}, hidden_size={}'：事件类型、字段数和隐藏层大小。
        """
        return 'event_type={}, num_fields={}, hidden_size={}'.format(
            self.event_type, self.num_fields, self.hidden_size
        )


class MLP(nn.Module):
    """
    多层感知机模块。该模块实现了多层感知机，用于进行非线性映射。

    参数：
    - input_size: 输入大小。
    - output_size: 输出大小。
    - mid_size: 中间层大小，默认为输入大小。
    - num_mid_layer: 中间层数，默认为1。
    - dropout: 随机丢弃的概率。

    返回：
    - 无返回值。
    """

    def __init__(self, input_size, output_size, mid_size=None, num_mid_layer=1, dropout=0.1):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer - 1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数。根据输入的张量，通过多层感知机进行非线性映射。

        参数：
        - x: 输入张量。

        返回：
        - x: 输出张量。
        """
        x = self.dropout(F.relu(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(F.relu(mid_fc(x)))
        x = self.out_fc(x)
        return x


def get_span_mention_info(span_dranges_list, doc_token_type_list):
    """
    获取跨度和提及信息的辅助函数。将跨度和提及信息从列表中提取出来。

    参数：
    - span_dranges_list: 跨度范围列表。
    - doc_token_type_list: 文档词元类型列表。

    返回：
    - span_mention_range_list: 提及范围列表。
    - mention_drange_list: 提及的跨度列表。
    - mention_type_list: 提及类型列表。
    """
    span_mention_range_list = []
    mention_drange_list = []
    mention_type_list = []
    for span_dranges in span_dranges_list:
        ment_idx_s = len(mention_drange_list)
        for drange in span_dranges:
            mention_drange_list.append(drange)
            sent_idx, char_s, char_e = drange
            mention_type_list.append(doc_token_type_list[sent_idx][char_s])
        ment_idx_e = len(mention_drange_list)
        span_mention_range_list.append((ment_idx_s, ment_idx_e))
    # Flatten all entity idx.
    return span_mention_range_list, mention_drange_list, mention_type_list


def extract_doc_valid_span_info(doc_token_type_mat, doc_fea):
    """
    从文档特征中提取有效的跨度信息。该函数通过解析文档的词元类型矩阵和特征对象，提取出有效的跨度信息，
    并以列表的形式返回跨度的词元元组列表和跨度范围列表。

    参数：
    - doc_token_type_mat: 文档词元类型矩阵。
    - doc_fea: 文档特征对象。

    返回：
    - span_token_tup_list: 跨度的词元元组列表。
    - span_dranges_list: 跨度范围列表。
    """
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]
            # Get the valid token span. the division is 2 since the encoding format is BIO, otherwise 3 for BIEOS.
            if entity_idx % 2 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1

                token_tup = tuple(seq_token_id_list[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                span_token_drange_list.append((token_tup, drange))

                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # 范围是独占的并且已排序
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list


def extract_doc_valid_span_info_for_bieos(doc_token_type_mat, doc_fea):
    """
    从文档特征中提取有效的BIEOS编码跨度信息。该函数通过解析文档的词元类型矩阵和特征对象，
    提取出有效的BIEOS编码跨度信息，并以列表的形式返回跨度的词元元组列表和跨度范围列表。

    参数：
    - doc_token_type_mat: 文档词元类型矩阵。
    - doc_fea: 文档特征对象。

    返回：
    - span_token_tup_list: 跨度的词元元组列表。
    - span_dranges_list: 跨度范围列表。
    """
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]
            # Get the valid token span. the division is 2 since the encoding format is BIO, otherwise 4 for BIEOS.
            if entity_idx % 4 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1
                if char_e == seq_len:
                    break
                elif char_e >= char_s + 1 and seq_token_type_list[char_e] == entity_idx + 2:
                    token_tup = tuple(seq_token_id_list[char_s: char_e + 1])
                    drange = (sent_idx, char_s, char_e + 1)
                    char_e += 1
                    span_token_drange_list.append((token_tup, drange))
                char_s = char_e
                #     char_s = char_e
                # else:
                #     char_s = char_e

            elif entity_idx % 4 == 0 and entity_idx != 0:
                char_e = char_s + 1
                token_tup = tuple(seq_token_id_list[char_s: char_e])
                drange = (sent_idx, char_s, char_e)
                span_token_drange_list.append((token_tup, drange))
                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # 范围是独占的并且已排序
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list


def get_batch_span_label(num_spans, cur_span_idx_set, device):
    """
    生成用于字段的批量跨度标签。该函数根据给定的跨度索引集，生成用于字段的二元标签列表。

    参数：
    - num_spans: 总跨度数量。
    - cur_span_idx_set: 当前字段的跨度索引集。
    - device: 计算设备。

    返回：
    - batch_field_label: 用于字段的跨度标签张量，取值为0或1。
    """

    # 为该字段和该路径准备跨度标签
    span_field_labels = [
        1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)
    ]
    if None in cur_span_idx_set:
        span_field_labels[-1] = 1

    batch_field_label = torch.tensor(
        span_field_labels, dtype=torch.long, device=device, requires_grad=False
    )  # [num_spans], val \in {0, 1}

    return batch_field_label


class DCFEEModel(nn.Module):
    """
    This module implements the baseline model described in http://www.aclweb.org/anthology/P18-4009:
        "DCFEE: A Document-level Chinese Financial Event Extraction System
        based on Automatically Labeled Training Data"
    该模块实现了http://www.aclweb.org/anthology/P18-4009中描述的基线模型
    """

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(DCFEEModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # 将词元嵌入聚合成句子嵌入的注意力规约器
        self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        # 将句子嵌入映射到事件预测对数的线性层
        self.event_cls_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, 2) for _ in self.event_type_fields_pairs
        ])

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        """
        从命名实体识别模型的词元嵌入中提取句子嵌入。

        参数：
        - ner_token_emb: 命名实体识别模型的词元嵌入。
        - ner_token_masks: 命名实体识别模型的词元掩码。
        - valid_sent_num_list: 有效句子数量的列表。

        返回：
        - total_sent_emb: 句子嵌入的总和。
        """
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        total_sent_emb = self.doc_token_reducer(ner_token_emb, ner_token_masks)
        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))

        return total_sent_emb

    def get_loss_on_doc(self, doc_sent_emb, doc_fea):
        """
        计算文档级别的损失。

        参数：
        - doc_sent_emb: 文档的句子嵌入。
        - doc_fea: 文档特征对象。

        返回：
        - final_loss: 文档级别的损失。
        """
        doc_sent_label_mat = torch.tensor(
            doc_fea.doc_sent_labels, dtype=torch.long, device=doc_sent_emb.device, requires_grad=False
        )
        event_cls_loss_list = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_labels = doc_sent_label_mat[:, event_idx]  # [sent_num]
            event_cls_loss = F.cross_entropy(doc_sent_logits, doc_sent_labels, reduction='sum')
            event_cls_loss_list.append(event_cls_loss)

        final_loss = sum(event_cls_loss_list)
        return final_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        """
        获取混合损失，用于字段级别的事件提取。

        参数：
        - doc_sent_loss_list: 文档中每个句子的损失列表。
        - doc_event_loss_list: 文档中每个事件的损失列表。
        - doc_span_info_list: 文档中跨度信息的列表。

        返回：
        - mix_loss: 混合损失，包括事件分类损失和事件结构损失。
        """
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            sent_loss_scaling = doc_sent_loss.new_full(
                doc_sent_loss.size(), 1, requires_grad=False
            )
            sent_loss_scaling[doc_span_info.missed_sent_idx_list] = self.config.loss_gamma
            doc_ner_loss = (doc_sent_loss * sent_loss_scaling).sum()
            doc_ner_loss_list.append(doc_ner_loss)

        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list))

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        """
        从文档批次字典中提取局部上下文信息。

        参数：
        - doc_batch_dict: 包含文档信息的字典，例如词元、特征等。
        - train_flag: 是否处于训练模式。
        - use_gold_span: 是否使用金标跨度。

        返回：
        - doc_token_emb_list: 文档词元嵌入列表。
        - doc_token_masks_list: 文档词元掩码列表。
        - doc_token_types_list: 文档词元类型列表。
        - doc_sent_emb_list: 文档句子嵌入列表。
        - doc_sent_loss_list: 文档中每个句子的损失列表。
        """
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = doc_batch_dict[label_key]
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx + 1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def forward(self, doc_batch_dict, doc_features,
                use_gold_span=False, train_flag=True, heuristic_type='DCFEE-O',
                event_idx2entity_idx2field_idx=None, **kwargs):
        """
        模型的前向传播函数，负责处理文档级别的事件提取任务。该函数接受输入文档信息，包括词元、特征等，
        并根据训练标志、使用金标跨度标志、计划采样概率等进行前向传播。在训练时，根据计划采样的概率
        使用金标跨度进行训练，否则使用模型预测的跨度。该函数还包括关系提取模块，根据配置使用RAAT
        (Relation-Aware Attentive Transition)模块。在推理阶段，可以选择使用启发式方法对事件和字段进行解码。

        参数：
        - doc_batch_dict: 包含文档信息的字典，例如词元、特征等。
        - doc_features: 包含文档特征的对象。
        - use_gold_span: 是否使用金标跨度进行训练。
        - train_flag: 是否处于训练模式。
        - heuristic_type: 启发式解码的类型。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。

        返回：
        - mix_loss: 训练时的混合损失（事件分类损失和事件结构损失）。
        或
        - eval_results: 推理时的结果列表。
        """
        # DCFEE does not need scheduled sampling
        # get doc token-level local context
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # get doc feature objects
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)

        if train_flag:  # 训练时
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list)

            return mix_loss
        else:  # 推理时
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []

            assert event_idx2entity_idx2field_idx is not None
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.heuristic_decode_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        event_idx2entity_idx2field_idx,
                        heuristic_type=heuristic_type,
                    )
                )

            return eval_results

    def heuristic_decode_on_doc(self, doc_sent_emb, doc_fea, doc_span_info,
                                event_idx2entity_idx2field_idx, heuristic_type='DCFEE-O'):
        """
        启发式解码，处理文档级别的事件解码。

        参数：
        - doc_sent_emb: 文档的句子嵌入。
        - doc_fea: 文档特征对象。
        - doc_span_info: 文档的跨度信息。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。
        - heuristic_type: 启发式解码的类型。

        返回：
        - doc_fea.ex_idx: 文档示例索引。
        - event_pred_list: 事件预测列表。
        - event_idx2obj_idx2field_idx2token_tup: 事件索引到对象索引到字段索引到词元元组的映射。
        - doc_span_info: 文档的跨度信息。
        - event_idx2field_idx2span_token_tup2dranges: 事件索引到字段索引到词元元组到跨度范围的映射。
        - event_idx2key_sent_idx_list: 事件索引到关键句子索引列表的映射。
        """
        # DCFEE-O: 只生成每个触发句子的一个事件
        # DCFEE-M: 对每个触发句子生成多个潜在事件
        support_heuristic_types = ['DCFEE-O', 'DCFEE-M']
        if heuristic_type not in support_heuristic_types:
            raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                heuristic_type, str(support_heuristic_types)
            ))

        span_token_tup_list = doc_span_info.span_token_tup_list
        span_mention_range_list = doc_span_info.span_mention_range_list
        mention_drange_list = doc_span_info.mention_drange_list
        mention_type_list = doc_span_info.mention_type_list
        # 启发式解码策略将在这些跨度候选上执行
        event_idx2field_idx2span_token_tup2dranges = self.get_event_field_span_candidates(
            span_token_tup_list, span_mention_range_list, mention_drange_list,
            mention_type_list, event_idx2entity_idx2field_idx,
        )

        # 如果没有提取的跨度，直接返回
        if len(span_token_tup_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []  # this term will be compared with ground-truth table contents
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2field_idx2span_token_tup2dranges

        event_idx2key_sent_idx_list = []
        event_pred_list = []
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            event_type, field_types = self.event_type_fields_pairs[event_idx]
            num_fields = len(field_types)
            field_idx2span_token_tup2dranges = event_idx2field_idx2span_token_tup2dranges[event_idx]

            # 获取关键事件句子预测
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_logp = F.log_softmax(doc_sent_logits, dim=-1)  # [sent_num, 2]
            doc_sent_pred_list = doc_sent_logp.argmax(dim=-1).tolist()
            key_sent_idx_list = [
                sent_idx for sent_idx, sent_pred in enumerate(doc_sent_pred_list) if sent_pred == 1
            ]
            event_idx2key_sent_idx_list.append(key_sent_idx_list)

            if len(key_sent_idx_list) == 0:
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
            else:
                obj_idx2field_idx2token_tup = []
                for key_sent_idx in key_sent_idx_list:
                    if heuristic_type == support_heuristic_types[0]:
                        field_idx2token_tup = get_one_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                    elif heuristic_type == support_heuristic_types[1]:
                        field_idx2token_tup_list = get_many_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.extend(field_idx2token_tup_list)
                    else:
                        raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                            heuristic_type, str(support_heuristic_types)
                        ))
                event_pred_list.append(1)
                event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2field_idx2span_token_tup2dranges, event_idx2key_sent_idx_list

    def get_event_field_span_candidates(self, span_token_tup_list, span_mention_range_list,
                                        mention_drange_list, mention_type_list, event_idx2entity_idx2field_idx):
        """
        获取事件字段跨度候选者。此函数将根据提供的跨度信息和实体到字段映射，为每个事件生成字段到跨度的映射。

        参数：
        - span_token_tup_list: 跨度的词元元组列表。
        - span_mention_range_list: 跨度对应的提及范围列表。
        - mention_drange_list: 提及的范围列表。
        - mention_type_list: 提及的类型列表。
        - event_idx2entity_idx2field_idx: 事件索引到实体索引到字段索引的映射。

        返回：
        - event_idx2field_idx2span_token_tup2dranges: 事件索引到字段索引到词元元组到跨度范围的映射。
        """
        # 获取提及索引 -> 跨度索引
        mention_span_idx_list = []
        for span_idx, (ment_idx_s, ment_idx_e) in enumerate(span_mention_range_list):
            mention_span_idx_list.extend([span_idx] * (ment_idx_e - ment_idx_s))
        assert len(mention_span_idx_list) == len(mention_drange_list)

        event_idx2field_idx2span_token_tup2dranges = {}
        for event_idx, (event_type, field_types) in enumerate(self.event_type_fields_pairs):
            # 获取预定义实体索引到字段索引映射
            gold_entity_idx2field_idx = event_idx2entity_idx2field_idx[event_idx]

            # 为该文档存储字段候选者
            field_idx2span_token_tup2dranges = {}
            for field_idx, _ in enumerate(field_types):
                field_idx2span_token_tup2dranges[field_idx] = {}

            # 根据提及类型聚合字段候选者
            for ment_idx, (ment_drange, ment_entity_idx) in enumerate(zip(mention_drange_list, mention_type_list)):
                if ment_entity_idx not in gold_entity_idx2field_idx:
                    continue
                ment_field_idx = gold_entity_idx2field_idx[ment_entity_idx]
                if ment_field_idx is None:
                    continue

                ment_span_idx = mention_span_idx_list[ment_idx]
                span_token_tup = span_token_tup_list[ment_span_idx]

                # 由于是字典，因此对键的所有修改将在原始字典中生效
                cur_span_token_tup2dranges = field_idx2span_token_tup2dranges[ment_field_idx]
                if span_token_tup not in cur_span_token_tup2dranges:
                    cur_span_token_tup2dranges[span_token_tup] = []
                cur_span_token_tup2dranges[span_token_tup].append(ment_drange)

            event_idx2field_idx2span_token_tup2dranges[event_idx] = field_idx2span_token_tup2dranges

        return event_idx2field_idx2span_token_tup2dranges


def get_one_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    """
    获取仅包含一个关键句子的事件。

    参数：
    - key_sent_idx: 关键句子的索引。
    - num_fields: 字段的数量。
    - field_idx2span_token_tup2dranges: 字段索引到词元元组到跨度范围的映射。

    返回：
    - field_idx2token_tup: 字段索引到词元元组的映射。
    """
    field_idx2token_tup = []
    for field_idx in range(num_fields):
        token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

        # 找到距离关键句子最近的词元元组
        best_token_tup = None
        best_dist = 10000
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                cur_dist = abs(sent_idx - key_sent_idx)
                if cur_dist < best_dist:
                    best_token_tup = token_tup
                    best_dist = cur_dist

        field_idx2token_tup.append(best_token_tup)
    return field_idx2token_tup


def get_many_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    """
    获取包含多个关键句子的事件。

    参数：
    - key_sent_idx: 关键句子的索引。
    - num_fields: 字段的数量。
    - field_idx2span_token_tup2dranges: 字段索引到词元元组到跨度范围的映射。

    返回：
    - field_idx2token_tup_list: 事件中的字段索引到词元元组的列表。
    """
    # 获取包含在关键事件句子中的关键字段索引
    key_field_idx2token_tup_set = defaultdict(lambda: set())
    for field_idx, token_tup2dranges in field_idx2span_token_tup2dranges.items():
        assert field_idx < num_fields
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                if sent_idx == key_sent_idx:
                    key_field_idx2token_tup_set[field_idx].add(token_tup)

    field_idx2token_tup_list = []
    while len(key_field_idx2token_tup_set) > 0:
        # 获取关键字段索引到词元元组的候选者，根据句子中的距离
        prev_field_idx = None
        prev_token_cand = None
        key_field_idx2token_cand = {}
        for key_field_idx, token_tup_set in key_field_idx2token_tup_set.items():
            assert len(token_tup_set) > 0

            if prev_token_cand is None:
                best_token_tup = token_tup_set.pop()
            else:
                prev_char_range = field_idx2span_token_tup2dranges[prev_field_idx][prev_token_cand][0][1:]
                best_dist = 10000
                best_token_tup = None
                for token_tup in token_tup_set:
                    cur_char_range = field_idx2span_token_tup2dranges[key_field_idx][token_tup][0][1:]
                    cur_dist = min(
                        abs(cur_char_range[1] - prev_char_range[0]),
                        abs(cur_char_range[0] - prev_char_range[1])
                    )
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_token_tup = token_tup
                token_tup_set.remove(best_token_tup)

            key_field_idx2token_cand[key_field_idx] = best_token_tup
            prev_field_idx = key_field_idx
            prev_token_cand = best_token_tup

        field_idx2token_tup = []
        for field_idx in range(num_fields):
            token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

            if field_idx in key_field_idx2token_tup_set:
                token_tup_set = key_field_idx2token_tup_set[field_idx]
                if len(token_tup_set) == 0:
                    del key_field_idx2token_tup_set[field_idx]
                token_tup = key_field_idx2token_cand[field_idx]
                field_idx2token_tup.append(token_tup)
            else:
                # 找到距离关键句子最近的词元元组
                best_token_tup = None
                best_dist = 10000
                for token_tup, dranges in token_tup2dranges.items():
                    for sent_idx, _, _ in dranges:
                        cur_dist = abs(sent_idx - key_sent_idx)
                        if cur_dist < best_dist:
                            best_token_tup = token_tup
                            best_dist = cur_dist

                field_idx2token_tup.append(best_token_tup)

        field_idx2token_tup_list.append(field_idx2token_tup)

    return field_idx2token_tup_list
