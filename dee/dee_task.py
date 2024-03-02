# -*- coding: utf-8 -*-

import logging
import json
import os
import torch.optim as optim
import torch.distributed as dist
from itertools import product

from transformers import AutoConfig

from dee.dee_helper import logger, DEEExample, DEEExampleLoader, DEEFeatureConverter, \
    convert_dee_features_to_dataset, prepare_doc_batch_dict, measure_dee_prediction, \
    measure_dee_prediction_with_pickle_input, decode_dump_template, eval_dump_template, \
    merge_multi_events
from dee.utils import BERTChineseCharacterTokenizer, default_dump_json, default_load_pkl
from dee.ner_model import BertForBasicNER
from dee.base_task import TaskSetting, BasePytorchTask
from dee.event_type import event_type_fields_list
from dee.dee_model import ReDEEModel, DCFEEModel
from dre.modeling_bert import BertForDocRED
from dre.utils import DREProcessor


class DEETaskSetting(TaskSetting):
    # 基础键属性，继承自TaskSetting的基础键属性
    base_key_attrs = TaskSetting.base_key_attrs
    # 基础属性默认键值对列表
    base_attr_default_pairs = [
        ('train_file_name', 'train.json'),            # 训练文件名
        ('dev_file_name', 'dev.json'),                # 开发集文件名
        ('test_file_name', 'test.json'),              # 测试集文件名
        ('dev_pkl_name', 'dev.pkl'),                  # 开发集pickle文件名
        ('test_pkl_name', 'test.pkl'),                # 测试集pickle文件名
        ('summary_dir_name', '/tmp/Summary'),         # 摘要目录名
        ('max_sent_len', 128),                        # 最大句子长度
        ('max_sent_num', 64),                         # 最大句子数量
        ('max_span_freq', 10),                        # 最大跨度频率
        ('train_batch_size', 64),                     # 训练批量大小
        ('gradient_accumulation_steps', 8),           # 梯度累积步数
        ('eval_batch_size', 2),                       # 评估批量大小
        ('learning_rate', 5e-5),                      # 学习率
        ('num_train_epochs', 2),                      # 训练轮数
        ('no_cuda', False),                           # 是否禁用CUDA
        ('local_rank', -1),                           # 本地排名
        ('seed', 99),                                 # 随机种子
        ('optimize_on_cpu', False),                   # 是否在CPU上优化
        ('fp16', False),                              # 是否使用混合精度训练
        ('use_bert', False),                          # 是否使用BERT作为编码器
        ('bert_model', ''),                           # 使用的预训练BERT模型
        ('only_master_logging', True),                # 是否只打印来自多个进程的日志的主节点
        ('resume_latest_cpt', True),                  # 是否在训练时恢复最新的检查点以实现容错性
        ('cpt_file_name', 'ReDEE'),                   # 检查点、评估结果等的标识符
        ('model_type', 'ReDEE'),                      # 使用的模型类
        ('rearrange_sent', False),                    # 是否重新排列句子
        ('use_crf_layer', True),                      # 是否使用CRF层
        ('min_teacher_prob', 0.1),                    # 使用金标跨度的最小概率
        ('schedule_epoch_start', 10),                 # 计划抽样开始的轮数
        ('schedule_epoch_length', 10),                # 线性过渡到min_teacher_prob的轮数
        ('loss_lambda', 0.05),                        # ner loss的比例
        ('loss_gamma', 1.0),                          # 错过跨度句子ner loss的缩放比例
        ('add_greedy_dec', False),                    # 是否添加额外的greedy解码
        ('use_token_role', True),                     # 是否使用详细的token role
        ('seq_reduce_type', 'MaxPooling'),            # 使用'MaxPooling'、'MeanPooling'或'AWA'来减少张量序列
        # 网络参数（遵循Bert Base）
        ('hidden_size', 768),                         # 隐藏层大小
        ('dropout', 0.1),                             # 丢弃率
        ('ff_size', 1024),                            # 前馈中间层大小
        ('num_tf_layers', 4),                         # transformer层的数量
        # 切除研究参数
        ('use_path_mem', True),                       # 是否在扩展路径时使用内存模块
        ('use_scheduled_sampling', True),             # 是否使用计划的抽样
        ('use_doc_enc', True),                        # 是否使用文档级实体编码
        ('neg_field_loss_scaling', 3.0),              # FN优于FP的比例
        ('with_post_process', False),                 # 是否将多个事件融合成一个
        ('with_pkl_file', False),                     # 是否加载dev和test文件的pkl文件
        ('use_re', True),                             # 是否使用正则表达式
        ('re_label_map_path', "label_map.json"),      # 正则表达式标签映射路径
        ('max_ent_cnt', 42),                          # 最大实体计数
        ('doc_max_length', 512),                      # 文档最大长度
        ('with_naive_feature', True),                 # 是否使用简单特征
        ("entity_structure", 'biaffine'),             # 实体结构
        ("logging_steps", 1),                         # 日志步骤
        ("re_loss_ratio", 1.0),                       # 正则表达式损失比率
        ("raat", False),                              # 是否用RAAT模块替代vanilla transformer-2
        ("raat_path_mem", False),                     # 是否用RAAT模块替代vanilla transformer-3
        ("head_center", True),                        # 是否使用关系内的头实体或尾实体作为RAAT聚类中心
        ("num_relation", 18),                         # 考虑关系的数量
    ]

    # 构造函数，初始化DEETaskSetting类的实例
    def __init__(self, **kwargs):
        # 调用父类TaskSetting的构造函数
        super(DEETaskSetting, self).__init__(
            self.base_key_attrs, self.base_attr_default_pairs, **kwargs
        )


class DEETask(BasePytorchTask):
    # 创建一个名为DEETask的类，继承自BasePytorchTask
    """Doc-level Event Extraction Task"""

    # 初始化方法，接受一些参数用于配置任务
    def __init__(self, dee_setting, load_train=True, load_dev=True, load_test=True,
                 parallel_decorate=True):
        # 调用父类的初始化方法
        super(DEETask, self).__init__(dee_setting, only_master_logging=dee_setting.only_master_logging)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging('Initializing {}'.format(self.__class__.__name__))

        # 使用BERT中文字符分词器
        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.bert_model)
        self.setting.vocab_size = len(self.tokenizer.vocab.items())
        self.id_2_vocab = {pair[1]: pair[0] for pair in list(self.tokenizer.vocab.items())}

        # 获取实体和事件标签名（BIO格式）
        self.entity_label_list = DEEExample.get_entity_label_list()

        # 获取实体和事件标签名（BIEOS格式）
        self.entity_label_list = DEEExample.get_entity_label_list_bieos()

        # 获取事件类型字段对
        self.event_type_fields_pairs = DEEExample.get_event_type_fields_pairs()

        # 构建示例加载器
        self.example_loader_func = DEEExampleLoader(self.setting.rearrange_sent, self.setting.max_sent_len)

        # 构建特征转换器
        if self.setting.use_bert:
            self.feature_converter_func = DEEFeatureConverter(
                self.entity_label_list, self.event_type_fields_pairs,
                self.setting.max_sent_len, self.setting.max_sent_num, self.tokenizer,
                include_cls=True, include_sep=True,
            )
        else:
            self.feature_converter_func = DEEFeatureConverter(
                self.entity_label_list, self.event_type_fields_pairs,
                self.setting.max_sent_len, self.setting.max_sent_num, self.tokenizer,
                include_cls=False, include_sep=False,
            )


        # 加载数据
        self._load_data(
            self.example_loader_func, self.feature_converter_func, convert_dee_features_to_dataset,
            load_train=load_train, load_dev=load_dev, load_test=load_test,
        )
        # 定制的小批量生成器
        self.custom_collate_fn = prepare_doc_batch_dict

        if not self.setting.use_token_role:
            # no token role conflicts with some settings
            assert self.setting.model_type == 'ReDEE'
            assert self.setting.add_greedy_dec is False
            self.setting.num_entity_labels = 3  # 0: 'O', 1: 'Begin', 2: 'Inside'
        else:
            self.setting.num_entity_labels = len(self.entity_label_list)

        if self.setting.use_bert:
            # 如果使用BERT，初始化BERT模型
            ner_model = BertForBasicNER.from_pretrained(
                self.setting.bert_model, num_entity_labels=self.setting.num_entity_labels
            )
            self.setting.update_by_dict(ner_model.config.__dict__)  # BertConfig dictionary

            # 替换BERT中的池化器以支持分布式训练
            # 因为未使用的参数在进行分布式all_reduce时会导致错误
            class PseudoPooler(object):
                def __init__(self):
                    pass

                def __call__(self, *x):
                    return x
            del ner_model.bert.pooler
            ner_model.bert.pooler = PseudoPooler()
        else:
            ner_model = None

        # 初始化关系抽取模块
        if self.setting.use_re:
            with open(self.setting.re_label_map_path, "r", encoding="utf-8") as f:
                re_label_map = json.load(f)
            config = AutoConfig.from_pretrained(
                self.setting.bert_model,
                cache_dir=None,
            )
            re_model = BertForDocRED.from_pretrained(self.setting.bert_model,
                                                     from_tf=False,
                                                     config=config,
                                                     cache_dir=None,
                                                     num_labels=len(re_label_map),
                                                     max_ent_cnt=self.setting.max_ent_cnt,
                                                     with_naive_feature=self.setting.with_naive_feature,
                                                     entity_structure=self.setting.entity_structure,
                                                     )
            re_processor = DREProcessor(doc_max_length=self.setting.doc_max_length,
                                        max_ent_cnt=self.setting.max_ent_cnt,
                                        label_map=re_label_map,
                                        token_ner_label_list=self.entity_label_list)

            class PseudoPooler(object):
                def __init__(self):
                    pass

                def __call__(self, *x):
                    return x

            del re_model.bert.pooler
            re_model.bert.pooler = PseudoPooler()
        else:
            re_model = None
            re_label_map = None
            re_processor = None

        if self.setting.model_type == 'ReDEE':
            # 如果模型类型是ReDEE，使用ReDEEModel
            self.model = ReDEEModel(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model, id_2_vocab=self.id_2_vocab,
                re_model=re_model, re_processor=re_processor
            )
        elif self.setting.model_type == 'DCFEE':
            # 如果模型类型是DCFEE，使用DCFEEModel
            self.model = DCFEEModel(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model
            )
        else:
            raise Exception('Unsupported model type {}'.format(self.setting.model_type))

        self._decorate_model(parallel_decorate=parallel_decorate)

        # 准备优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.setting.learning_rate)

        # # 恢复选项
        # if resume_model or resume_optimizer:
        #     self.resume_checkpoint(resume_model=resume_model, resume_optimizer=resume_optimizer)

        self.min_teacher_prob = None
        self.teacher_norm = None
        self.teacher_cnt = None
        self.teacher_base = None
        self.reset_teacher_prob()

        self.logging('Successfully initialize {}'.format(self.__class__.__name__))

    def reset_teacher_prob(self):
        self.min_teacher_prob = self.setting.min_teacher_prob
        if self.train_dataset is None:
            # avoid crashing when not loading training data
            num_step_per_epoch = 500
        else:
            num_step_per_epoch = int(len(self.train_dataset) / self.setting.train_batch_size)
        self.teacher_norm = num_step_per_epoch * self.setting.schedule_epoch_length
        self.teacher_base = num_step_per_epoch * self.setting.schedule_epoch_start
        self.teacher_cnt = 0

    def get_teacher_prob(self, batch_inc_flag=True):
        if self.teacher_cnt < self.teacher_base:
            prob = 1
        else:
            prob = max(
                self.min_teacher_prob, (self.teacher_norm - self.teacher_cnt + self.teacher_base) / self.teacher_norm
            )
            # prob = 1

        if batch_inc_flag:
            self.teacher_cnt += 1

        return prob

    def get_event_idx2entity_idx2field_idx(self):
        entity_idx2entity_type = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            if entity_label == 'O':
                entity_type = entity_label
            else:
                entity_type = entity_label[2:]

            entity_idx2entity_type[entity_idx] = entity_type

        event_idx2entity_idx2field_idx = {}
        for event_idx, (event_name, field_types) in enumerate(self.event_type_fields_pairs):
            field_type2field_idx = {}
            for field_idx, field_type in enumerate(field_types):
                field_type2field_idx[field_type] = field_idx

            entity_idx2field_idx = {}
            for entity_idx, entity_type in entity_idx2entity_type.items():
                if entity_type in field_type2field_idx:
                    entity_idx2field_idx[entity_idx] = field_type2field_idx[entity_type]
                else:
                    entity_idx2field_idx[entity_idx] = None

            event_idx2entity_idx2field_idx[event_idx] = entity_idx2field_idx

        return event_idx2entity_idx2field_idx

    def get_loss_on_batch(self, doc_batch_dict, features=None):
        if features is None:
            features = self.train_features

        # teacher_prob = 1
        # if use_gold_span, gold spans will be used every time
        # else, teacher_prob will ensure the proportion of using gold spans
        if self.setting.use_scheduled_sampling:
            use_gold_span = False
            teacher_prob = self.get_teacher_prob()
        else:
            use_gold_span = True
            teacher_prob = 1

        try:
            loss = self.model(
                doc_batch_dict, features, use_gold_span=use_gold_span, train_flag=True, teacher_prob=teacher_prob
            )
        except Exception as e:
            print('-'*30)
            print('Exception occurs when processing ' +
                  ','.join([features[ex_idx].guid for ex_idx in doc_batch_dict['ex_idx']]))
            raise Exception('Cannot get the loss')

        return loss

    def get_event_decode_result_on_batch(self, doc_batch_dict, features=None, use_gold_span=False, heuristic_type=None):
        if features is None:
            raise Exception('Features mush be provided')

        if heuristic_type is None:
            event_idx2entity_idx2field_idx = None
        else:
            # this mapping is used to get span candidates for each event field
            event_idx2entity_idx2field_idx = self.get_event_idx2entity_idx2field_idx()

        batch_eval_results = self.model(
            doc_batch_dict, features, use_gold_span=use_gold_span, train_flag=False,
            event_idx2entity_idx2field_idx=event_idx2entity_idx2field_idx, heuristic_type=heuristic_type,
        )

        return batch_eval_results

    def train(self, save_cpt_flag=True, resume_base_epoch=None):
        self.logging('=' * 20 + 'Start Training' + '=' * 20)
        self.reset_teacher_prob()

        # resume_base_epoch arguments have higher priority over settings
        if resume_base_epoch is None:
            # whether to resume latest cpt when restarting, very useful for preemptive scheduling clusters
            if self.setting.resume_latest_cpt:
                resume_base_epoch = self.get_latest_cpt_epoch()
            else:
                resume_base_epoch = 0

        # resume cpt if possible
        if resume_base_epoch > 0:
            self.logging('Training starts from epoch {}'.format(resume_base_epoch))
            for _ in range(resume_base_epoch):
                self.get_teacher_prob()
            self.resume_cpt_at(resume_base_epoch, resume_model=True, resume_optimizer=True)
        else:
            self.logging('Training starts from scratch')

        self.base_train(
            DEETask.get_loss_on_batch,
            kwargs_dict1={},
            epoch_eval_func=DEETask.resume_save_eval_at,
            kwargs_dict2={
                'save_cpt_flag': save_cpt_flag,
                'resume_cpt_flag': False,
            },
            base_epoch_idx=resume_base_epoch,
        )

    def resume_save_eval_at(self, epoch, resume_cpt_flag=False, save_cpt_flag=True):
        if self.is_master_node():
            print('\nPROGRESS: {:.2f}%\n'.format(epoch / self.setting.num_train_epochs * 100))
        self.logging('Current teacher prob {}'.format(self.get_teacher_prob(batch_inc_flag=False)))

        if resume_cpt_flag:
            self.resume_cpt_at(epoch)

        if self.is_master_node() and save_cpt_flag:
            self.save_cpt_at(epoch)

        if self.setting.model_type == 'DCFEE':
            eval_tasks = product(['dev', 'test'], [False, True], ['DCFEE-O', 'DCFEE-M'])
        else:
            if self.setting.add_greedy_dec:
                eval_tasks = product(['dev', 'test'], [False, True], ['GreedyDec', None])
            else:
                eval_tasks = product(['dev', 'test'], [False, True], [None])

        for task_idx, (data_type, gold_span_flag, heuristic_type) in enumerate(eval_tasks):
            if self.in_distributed_mode() and task_idx % dist.get_world_size() != dist.get_rank():
                continue

            if data_type == 'test':
                features = self.test_features
                dataset = self.test_dataset
            elif data_type == 'dev':
                features = self.dev_features
                dataset = self.dev_dataset
            else:
                raise Exception('Unsupported data type {}'.format(data_type))

            if gold_span_flag:
                span_str = 'gold_span'
            else:
                span_str = 'pred_span'

            if heuristic_type is None:
                # store user-provided name
                model_str = self.setting.cpt_file_name.replace('.', '~')
            else:
                model_str = heuristic_type

            decode_dump_name = decode_dump_template.format(data_type, span_str, model_str, epoch)
            eval_dump_name = eval_dump_template.format(data_type, span_str, model_str, epoch)
            self.eval(features, dataset, use_gold_span=gold_span_flag, heuristic_type=heuristic_type,
                      dump_decode_pkl_name=decode_dump_name, dump_eval_json_name=eval_dump_name, data_type=data_type)

    def save_cpt_at(self, epoch):
        self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch), epoch=epoch)

    def resume_cpt_at(self, epoch, resume_model=True, resume_optimizer=False):
        self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch),
                               resume_model=resume_model, resume_optimizer=resume_optimizer)

    def get_latest_cpt_epoch(self):
        prev_epochs = []
        for fn in os.listdir(self.setting.model_dir):
            if fn.startswith('{}.cpt'.format(self.setting.cpt_file_name)):
                try:
                    epoch = int(fn.split('.')[-1])
                    prev_epochs.append(epoch)
                except Exception as e:
                    continue
        prev_epochs.sort()

        if len(prev_epochs) > 0:
            latest_epoch = prev_epochs[-1]
            self.logging('Pick latest epoch {} from {}'.format(latest_epoch, str(prev_epochs)))
        else:
            latest_epoch = 0
            self.logging('No previous epoch checkpoints, just start from scratch')

        return latest_epoch

    def eval(self, features, dataset, use_gold_span=False, heuristic_type=None,
             dump_decode_pkl_name=None, dump_eval_json_name=None, data_type=None):
        self.logging('=' * 20 + 'Start Evaluation' + '=' * 20)

        if dump_decode_pkl_name is not None:
            dump_decode_pkl_path = os.path.join(self.setting.output_dir, dump_decode_pkl_name)
            self.logging('Dumping decode results into {}'.format(dump_decode_pkl_name))
        else:
            dump_decode_pkl_path = None

        total_event_decode_results = self.base_eval(
            dataset, DEETask.get_event_decode_result_on_batch,
            reduce_info_type='none', dump_pkl_path=dump_decode_pkl_path,
            features=features, use_gold_span=use_gold_span, heuristic_type=heuristic_type,
        )

        self.logging('Measure DEE Prediction')

        if dump_eval_json_name is not None:
            dump_eval_json_path = os.path.join(self.setting.output_dir, dump_eval_json_name)
            self.logging('Dumping eval results into {}'.format(dump_eval_json_name))
        else:
            dump_eval_json_path = None

        if self.setting.with_pkl_file:
            pkl_data = self.dev_pkl if data_type == "dev" else self.test_pkl
            total_eval_res = measure_dee_prediction_with_pickle_input(
                self.event_type_fields_pairs, pkl_data, total_event_decode_results,
                dump_json_path=dump_eval_json_path, with_post_process=self.setting.with_post_process,
                vocab_list=list(self.tokenizer.vocab.keys())
            )
        else:
            total_eval_res = measure_dee_prediction(
                self.event_type_fields_pairs, features, total_event_decode_results,
                dump_json_path=dump_eval_json_path, with_post_process=self.setting.with_post_process,
                vocab_list=list(self.tokenizer.vocab.keys())
            )

        return total_event_decode_results, total_eval_res

    def predict(self, sent_lst):
        """inference method, the input is a list of sentences, and the output is the prediction result."""
        origin_input = {"sentences": sent_lst}
        example_loader_func = DEEExampleLoader(self.setting.rearrange_sent,
                                               self.setting.max_sent_len,
                                               only_inference=True)
        examples = example_loader_func(None, origin_input)
        features = self.feature_converter_func(examples)
        dataset = convert_dee_features_to_dataset(features)

        total_event_decode_results = self.base_eval(
            dataset, DEETask.get_event_decode_result_on_batch,
            reduce_info_type='none', dump_pkl_path=None,
            features=features, use_gold_span=False, heuristic_type=None,
        )

        # transfer the decode result into corresponding keyword results.
        event_res_lst = {}

        # get the vocab_list for entity merge implementation.
        vocab_list = list(self.tokenizer.vocab.keys())
        unk_id = vocab_list.index("[UNK]")
        for result in total_event_decode_results:
            pred_event_type_labels, pred_record_mat = result[1: 3]
            pred_record_mat = merge_multi_events(pred_record_mat, vocab_list, process_for_display=True)

            for event_id, has_event in enumerate(pred_event_type_labels):
                if has_event:
                    event_type, event_role_names = event_type_fields_list[event_id]
                    events_per_type = pred_record_mat[event_id]
                    sample_res_lst = []
                    for i, event_mat in enumerate(events_per_type):
                        event_dict = {}
                        for event_role_span, role_name in zip(event_mat, event_role_names):
                            if event_role_span:
                                # For event having multiple people(with corresponding position),
                                # name them with 'PersonName_id'(with corresponding 'Position_id').
                                # if isinstance(event_role_span, list):
                                #     for i, span in enumerate(event_role_span):
                                #         if span is not None:
                                #             keyword = ""
                                #             for id in span:
                                #                 keyword += self.id_2_vocab[id]
                                #             event_dict[f"{role_name}_{i}"] = keyword
                                # else:
                                #     keyword = ""
                                #     for id in event_role_span:
                                #         keyword += self.id_2_vocab[id]
                                #     event_dict[role_name] = keyword
                                keyword = ""
                                for id in event_role_span:
                                    if id != unk_id:
                                        keyword += self.id_2_vocab[id]
                                event_dict[role_name] = keyword
                            # else:
                                # For event role which does not have entity, do not output it.
                                # event_dict[role_name] = "None"

                        # For event which does not have 'Event' entity, do not output it; Disaster event should have
                        # at least two valid entities; Other type event should have at least three valid entities.
                        # if 'Event' not in event_dict\
                        #         or event_type == "Disaster" and len(event_dict.values()) < 2\
                        #         or event_type != "Disaster" and len(event_dict.values()) < 3:
                        #     continue

                        sample_res_lst.append(event_dict)
                    if len(sample_res_lst) > 0:
                        event_res_lst[event_type] = sample_res_lst

        return event_res_lst

    def reevaluate_dee_prediction(self, target_file_pre='dee_eval', target_file_suffix='.pkl',
                                  dump_flag=False):
        """Enumerate the evaluation directory to collect all dumped evaluation results"""
        eval_dir_path = self.setting.output_dir
        logger.info('Re-evaluate dee predictions from {}'.format(eval_dir_path))
        data_span_type2model_str2epoch_res_list = {}
        for fn in os.listdir(eval_dir_path):
            fn_splits = fn.split('.')
            if fn.startswith(target_file_pre) and fn.endswith(target_file_suffix) and len(fn_splits) == 6:
                _, data_type, span_type, model_str, epoch, _ = fn_splits

                data_span_type = (data_type, span_type)
                if data_span_type not in data_span_type2model_str2epoch_res_list:
                    data_span_type2model_str2epoch_res_list[data_span_type] = {}
                model_str2epoch_res_list = data_span_type2model_str2epoch_res_list[data_span_type]

                if model_str not in model_str2epoch_res_list:
                    model_str2epoch_res_list[model_str] = []
                epoch_res_list = model_str2epoch_res_list[model_str]

                if data_type == 'dev':
                    features = self.dev_features
                elif data_type == 'test':
                    features = self.test_features
                else:
                    raise Exception('Unsupported data type {}'.format(data_type))

                epoch = int(epoch)
                fp = os.path.join(eval_dir_path, fn)
                self.logging('Re-evaluating {}'.format(fp))
                event_decode_results = default_load_pkl(fp)

                if self.setting.with_pkl_file:
                    pkl_data = self.dev_pkl if data_type == "dev" else self.test_pkl
                    total_eval_res = measure_dee_prediction_with_pickle_input(
                        event_type_fields_list, pkl_data, event_decode_results,
                        with_post_process=self.setting.with_post_process,
                        vocab_list=list(self.tokenizer.vocab.keys())
                    )
                else:
                    total_eval_res = measure_dee_prediction(
                        event_type_fields_list, features, event_decode_results,
                        with_post_process=self.setting.with_post_process,
                        vocab_list=list(self.tokenizer.vocab.keys())
                    )

                if dump_flag:
                    fp = fp.rstrip('.pkl') + '.json'
                    self.logging('Dumping {}'.format(fp))
                    default_dump_json(total_eval_res, fp)

                epoch_res_list.append((epoch, total_eval_res))

        for data_span_type, model_str2epoch_res_list in data_span_type2model_str2epoch_res_list.items():
            for model_str, epoch_res_list in model_str2epoch_res_list.items():
                epoch_res_list.sort(key=lambda x: x[0])

        return data_span_type2model_str2epoch_res_list
