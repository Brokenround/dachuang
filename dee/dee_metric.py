# -*- coding: utf-8 -*-

import numpy as np


def agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num, polysemy_mat):
    """
    聚合单个实例的事件预测的 TP、FP、FN 统计信息。
    pred_records 应该格式化为
    [(记录索引)
        ((角色索引)
            参数 1, ...
        ), ...
    ]，其中参数 1 应该支持 '=' 操作，空参数为 None。

    参数:
    - pred_records: 预测的事件记录。
    - gold_records: 实际的黄金事件记录。
    - role_num: 事件中的角色数量。
    - polysemy_mat: 多义性矩阵。

    返回:
    角色的 TP、FP、FN 统计信息列表。
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]
    polysemy_tpfpfn_stats = [0, 0, 0]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
                        if polysemy_mat and arg_tup in polysemy_mat:
                            polysemy_tpfpfn_stats[1] += 1
        else:  # 忽略 TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
                        if polysemy_mat and arg_tup in polysemy_mat:
                            polysemy_tpfpfn_stats[2] += 1
        else:  # 在事件级别的 True Positive
            # sort predicted event records by the non-empty count
            # to remove the impact of the record order on evaluation
            pred_records = sorted(pred_records,
                                  key=lambda x: sum(1 for a in x if a is not None),
                                  reverse=True)
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # pick the most similar gold record
                _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if pa == ga])
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                            if polysemy_mat and pred_arg in polysemy_mat:
                                polysemy_tpfpfn_stats[1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                            if polysemy_mat and gold_arg in polysemy_mat:
                                polysemy_tpfpfn_stats[2] += 1
                        else:
                            if pred_arg == gold_arg:  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                                if polysemy_mat and pred_arg in polysemy_mat:
                                    polysemy_tpfpfn_stats[0] += 1
                            else:
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1
                                if polysemy_mat and pred_arg in polysemy_mat:
                                    polysemy_tpfpfn_stats[1] += 1
                                if polysemy_mat and gold_arg in polysemy_mat:
                                    polysemy_tpfpfn_stats[2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # 剩余的 FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
                        if polysemy_mat and arg_tup in polysemy_mat:
                            polysemy_tpfpfn_stats[1] += 1
            # 剩余的 FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
                        if polysemy_mat and arg_tup in polysemy_mat:
                            polysemy_tpfpfn_stats[2] += 1
    if polysemy_mat:
        role_tpfpfn_stats.append(polysemy_tpfpfn_stats)
    return role_tpfpfn_stats


def agg_event_role_tpfpfn_stats_as_edag(pred_records, gold_records, role_num):
    """
    聚合预测 edag 和 gt edag 的 TP、FP、FN 统计信息。
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            pred_records = rearrange_event_entity_via_edag(pred_records, role_num)
            assert len(pred_records) == role_num
            for role_idx, pred_record in enumerate(pred_records):
                # 判断当前角色是否具有有效的实体提及。
                if pred_record[0] is not None:
                    role_tpfpfn_stats[role_idx][1] += len(pred_record)
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            gold_records = rearrange_event_entity_via_edag(gold_records, role_num)
            assert len(gold_records) == role_num
            for role_idx, gold_record in enumerate(gold_records):
                if gold_record[0] is not None:
                    role_tpfpfn_stats[role_idx][2] += len(gold_record)
        else:  # True Positive at the event level
            pred_records = rearrange_event_entity_via_edag(pred_records, role_num)
            gold_records = rearrange_event_entity_via_edag(gold_records, role_num)

            assert len(pred_records) == len(gold_records) == role_num
            for role_idx, (pred_record, gold_record) in enumerate(zip(pred_records, gold_records)):
                if pred_record[0] is None and gold_record[0] is None:
                    pass
                elif pred_record[0] is None:
                    role_tpfpfn_stats[role_idx][2] += len(gold_record)
                elif gold_record[0] is None:
                    role_tpfpfn_stats[role_idx][1] += len(pred_record)
                else:
                    tp = set(pred_record).intersection(set(gold_record))
                    fp = set(pred_record) - set(gold_record)
                    fn = set(gold_record) - set(pred_record)
                    role_tpfpfn_stats[role_idx][0] += len(tp)
                    role_tpfpfn_stats[role_idx][1] += len(fp)
                    role_tpfpfn_stats[role_idx][2] += len(fn)
    return role_tpfpfn_stats


def agg_event_role_tpfpfn_stats_with_synonym(pred_records, gold_records, role_num):
    """
    聚合带有同义词考虑的单个实例的事件预测的 TP、FP、FN 统计信息。
    """
    def overlap_for_two_synonyms_tuple(tup1, tup2, delimiter_id):
        # 考虑 tup1 和 tup2 中的同义词关键字，并用 "/" 分隔它们
        # delimiter_id 对应 "/"
        if tup1 is None and tup2 is None:
            return True

        elif tup1 is None or tup2 is None:
            return False

        split_tups = [[], []]
        for split_tup, tup in zip(split_tups, [tup1, tup2]):
            curr_arg_ids = []
            for id in tup:
                if id != delimiter_id:
                    curr_arg_ids.append(id)
                else:
                    split_tup.append(tuple(curr_arg_ids))
                    curr_arg_ids = []
            if len(curr_arg_ids) > 0:
                split_tup.append(tuple(curr_arg_ids))
        split_tup1, split_tup2 = split_tups

        if len(set(split_tup1).intersection(split_tup2)) > 0:
            return True
        return False

    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
        else:  # 在事件级别的 True Positive
            # 将预测的事件记录按非空计数排序，以消除记录顺序对评估的影响
            pred_records = sorted(pred_records,
                                  key=lambda x: sum(1 for a in x if a is not None),
                                  reverse=True)
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # 选择最相似的 gold record（考虑同义词）
                _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if
                                           overlap_for_two_synonyms_tuple(pa, ga, 120)])
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                        else:
                            # 120 is the index of "/"
                            if overlap_for_two_synonyms_tuple(pred_arg, gold_arg, 120):  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                            else:
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # remaining FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
            # remaining FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1

    return role_tpfpfn_stats


def rearrange_event_entity_via_edag(event_entity_mat, role_num):
    """将实体格式转换为edag格式。"""
    rearrage_mat = [[] for _ in range(role_num)]
    for event in event_entity_mat:
        for i, entity in enumerate(event):
            # 如果entity为None，应插入到列表中。
            if entity not in rearrage_mat[i]:
                rearrage_mat[i].append(entity)
    return rearrage_mat


def agg_event_level_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    获取事件级别的TP、FP、FN。
    """
    # 将角色级别的统计作为事件级别的统计
    role_tpfpfn_stats = agg_event_role_tpfpfn_stats(
        pred_records, gold_records, role_num
    )

    return list(np.sum(role_tpfpfn_stats, axis=0))


def agg_ins_event_role_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list, polysemy_mat=None):
    """
    聚合单个实例的TP、FP、FN统计信息。
    record_mat 应该格式化为
    [(事件索引)
        [(记录索引)
            ((角色索引)
                参数 1, ...
            ), ...
        ], ...
    ]，其中参数 1 应该支持 '=' 操作，空参数为 None。
    """
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_role_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records) in enumerate(zip(pred_record_mat, gold_record_mat)):
        role_num = event_role_num_list[event_idx]
        role_tpfpfn_stats = agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num, polysemy_mat)
        # role_tpfpfn_stats = agg_event_role_tpfpfn_stats_as_edag(pred_records, gold_records, role_num)
        # role_tpfpfn_stats = agg_event_role_tpfpfn_stats_with_synonym(pred_records, gold_records, role_num)
        event_role_tpfpfn_stats.append(role_tpfpfn_stats)

    return event_role_tpfpfn_stats


def agg_ins_event_level_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list):
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records, role_num) in enumerate(zip(
            pred_record_mat, gold_record_mat, event_role_num_list)):
        event_tpfpfn = agg_event_level_tpfpfn_stats(pred_records, gold_records, role_num)
        event_tpfpfn_stats.append(event_tpfpfn)

    return event_tpfpfn_stats


def get_prec_recall_f1(tp, fp, fn):
    a = tp + fp
    prec = tp / a if a > 0 else 0
    b = tp + fn
    rec = tp / b if b > 0 else 0
    if prec > 0 and rec > 0:
        f1 = 2.0 / (1 / prec + 1 / rec)
    else:
        f1 = 0
    return prec, rec, f1


def measure_event_table_filling(pred_record_mat_list, gold_record_mat_list, event_type_roles_list, avg_type='micro',
                                dict_return=False, polysemy_mat_list=None):
    """
    record_mat_list 格式为
    [(文档索引)
        [(事件索引)
            [(记录索引)
                ((角色索引)
                    参数 1, ...
                ), ...
            ], ...
        ], ...
    ]
    参数类型应支持 '==' 操作。
    空参数和记录设置为 None。
    """
    event_role_num_list = [len(roles) for _, roles in event_type_roles_list]
    # 存储TP、FP、FN的总统计信息
    total_event_role_stats = [
        [
            [0]*3 for _ in range(role_num)
        ] for event_idx, role_num in enumerate(event_role_num_list)
    ]
    polysemy_tpfpfn_stats = [0, 0, 0]
    if not polysemy_mat_list:
        polysemy_mat_list = [None] * len(pred_record_mat_list)

    # 确保输入列表的长度相同
    assert len(pred_record_mat_list) == len(gold_record_mat_list)
    for pred_record_mat, gold_record_mat, polysemy_mat in zip(pred_record_mat_list, gold_record_mat_list, polysemy_mat_list):
        # 聚合单个实例的TP、FP、FN统计信息
        event_role_tpfpfn_stats = agg_ins_event_role_tpfpfn_stats(
            pred_record_mat, gold_record_mat, event_role_num_list, polysemy_mat
        )
        for event_idx, role_num in enumerate(event_role_num_list):
            for role_idx in range(role_num):
                for sid in range(3):
                    total_event_role_stats[event_idx][role_idx][sid] += \
                        event_role_tpfpfn_stats[event_idx][role_idx][sid]
        if polysemy_mat:
            for stats in event_role_tpfpfn_stats:
                p_mat = stats[-1]
                for id in range(3):
                    polysemy_tpfpfn_stats[id] += p_mat[id]

    per_role_metric = []
    per_event_metric = []

    num_events = len(event_role_num_list)
    g_tpfpfn_stat = [0] * 3
    g_prf1_stat = [0] * 3
    event_role_eval_dicts = []
    for event_idx, role_num in enumerate(event_role_num_list):
        event_tpfpfn = [0] * 3  # tp, fp, fn
        event_prf1_stat = [0] * 3
        per_role_metric.append([])
        role_eval_dicts = []
        for role_idx in range(role_num):
            role_tpfpfn_stat = total_event_role_stats[event_idx][role_idx][:3]
            role_prf1_stat = get_prec_recall_f1(*role_tpfpfn_stat)
            per_role_metric[event_idx].append(role_prf1_stat)
            for mid in range(3):
                event_tpfpfn[mid] += role_tpfpfn_stat[mid]
                event_prf1_stat[mid] += role_prf1_stat[mid]

            role_eval_dict = {
                'RoleType': event_type_roles_list[event_idx][1][role_idx],
                'Precision': role_prf1_stat[0],
                'Recall': role_prf1_stat[1],
                'F1': role_prf1_stat[2],
                'TP': role_tpfpfn_stat[0],
                'FP': role_tpfpfn_stat[1],
                'FN': role_tpfpfn_stat[2]
            }
            role_eval_dicts.append(role_eval_dict)

        for mid in range(3):
            event_prf1_stat[mid] /= role_num
            g_tpfpfn_stat[mid] += event_tpfpfn[mid]
            g_prf1_stat[mid] += event_prf1_stat[mid]

        micro_event_prf1 = get_prec_recall_f1(*event_tpfpfn)
        macro_event_prf1 = tuple(event_prf1_stat)
        if avg_type.lower() == 'micro':
            event_prf1_stat = micro_event_prf1
        elif avg_type.lower() == 'macro':
            event_prf1_stat = macro_event_prf1
        else:
            raise Exception('Unsupported average type {}'.format(avg_type))

        per_event_metric.append(event_prf1_stat)

        event_eval_dict = {
            'EventType': event_type_roles_list[event_idx][0],
            'MacroPrecision': macro_event_prf1[0],
            'MacroRecall': macro_event_prf1[1],
            'MacroF1': macro_event_prf1[2],
            'MicroPrecision': micro_event_prf1[0],
            'MicroRecall': micro_event_prf1[1],
            'MicroF1': micro_event_prf1[2],
            'TP': event_tpfpfn[0],
            'FP': event_tpfpfn[1],
            'FN': event_tpfpfn[2],
        }
        event_role_eval_dicts.append((event_eval_dict, role_eval_dicts))

    micro_g_prf1 = get_prec_recall_f1(*g_tpfpfn_stat)
    macro_g_prf1 = tuple(s / num_events for s in g_prf1_stat)
    if avg_type.lower() == 'micro':
        g_metric = micro_g_prf1
    else:
        g_metric = macro_g_prf1

    g_eval_dict = {
        'MacroPrecision': macro_g_prf1[0],
        'MacroRecall': macro_g_prf1[1],
        'MacroF1': macro_g_prf1[2],
        'MicroPrecision': micro_g_prf1[0],
        'MicroRecall': micro_g_prf1[1],
        'MicroF1': micro_g_prf1[2],
        'TP': g_tpfpfn_stat[0],
        'FP': g_tpfpfn_stat[1],
        'FN': g_tpfpfn_stat[2],
    }
    if any(polysemy_mat_list):
        precision, recall, f1 = get_prec_recall_f1(*polysemy_tpfpfn_stats)
        g_eval_dict["PolyPrecision"] = precision
        g_eval_dict["PolyRecall"] = recall
        g_eval_dict["PolyF1"] = f1
    event_role_eval_dicts.append(g_eval_dict)

    if not dict_return:
        return g_metric, per_event_metric, per_role_metric
    else:
        return event_role_eval_dicts

