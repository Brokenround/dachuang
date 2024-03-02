# -*- coding: utf-8 -*-


class BaseEvent(object):
    def __init__(self, fields, event_name='Event', key_fields=(), recguid=None):
        self.recguid = recguid
        self.name = event_name
        self.fields = list(fields)
        self.field2content = {f: None for f in fields}
        self.nonempty_count = 0
        self.nonempty_ratio = self.nonempty_count / len(self.fields)

        self.key_fields = set(key_fields)
        for key_field in self.key_fields:
            assert key_field in self.field2content

    def __repr__(self):
        event_str = "\n{}[\n".format(self.name)
        event_str += "  {}={}\n".format("recguid", self.recguid)
        event_str += "  {}={}\n".format("nonempty_count", self.nonempty_count)
        event_str += "  {}={:.3f}\n".format("nonempty_ratio", self.nonempty_ratio)
        event_str += "] (\n"
        for field in self.fields:
            if field in self.key_fields:
                key_str = " (key)"
            else:
                key_str = ""
            event_str += "  " + field + "=" + str(self.field2content[field]) + ", {}\n".format(key_str)
        event_str += ")\n"
        return event_str

    def update_by_dict(self, field2text, recguid=None):
        self.nonempty_count = 0
        self.recguid = recguid

        for field in self.fields:
            if field in field2text and field2text[field] is not None:
                self.nonempty_count += 1
                self.field2content[field] = field2text[field]
            else:
                self.field2content[field] = None

        self.nonempty_ratio = self.nonempty_count / len(self.fields)

    def field_to_dict(self):
        return dict(self.field2content)

    def set_key_fields(self, key_fields):
        self.key_fields = set(key_fields)

    def is_key_complete(self):
        for key_field in self.key_fields:
            if self.field2content[key_field] is None:
                return False

        return True

    def is_good_candidate(self):
        raise NotImplementedError()

    def get_argument_tuple(self):
        args_tuple = tuple(self.field2content[field] for field in self.fields)
        return args_tuple


# 原金融事件
class EquityFreezeEvent(BaseEvent):
    NAME = 'EquityFreeze'  # 股权冻结
    FIELDS = [
        'EquityHolder',  # 股权持有者
        'FrozeShares',  # 冻结股票
        'LegalInstitution',  # 法律机构
        'TotalHoldingShares',  # 合计持股
        'TotalHoldingRatio',  # 合计持仓比率
        'StartDate',
        'EndDate',
        'UnfrozeDate',  # 未冻结日期
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityFreezeEvent.FIELDS, event_name=EquityFreezeEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'FrozeShares',
            'LegalInstitution',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityRepurchaseEvent(BaseEvent):
    NAME = 'EquityRepurchase'  # 股权回购
    FIELDS = [
        'CompanyName',  # 公司名
        'HighestTradingPrice',  # 最高交易价格
        'LowestTradingPrice',  # 最低交易价格
        'RepurchasedShares',  # 回购股票
        'ClosingDate',  # 关闭日期
        'RepurchaseAmount',  # 回购金额
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityRepurchaseEvent.FIELDS, event_name=EquityRepurchaseEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'CompanyName',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityUnderweightEvent(BaseEvent):
    NAME = 'EquityUnderweight'  # 股票减持
    FIELDS = [
        'EquityHolder',  # 股权持有者
        'TradedShares',  # 交易股票
        'StartDate',  # 开始日期
        'EndDate',  # 结束日期
        'LaterHoldingShares',  # 后来控股股份
        'AveragePrice',  # 平均价格
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityUnderweightEvent.FIELDS, event_name=EquityUnderweightEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'TradedShares',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityOverweightEvent(BaseEvent):
    NAME = 'EquityOverweight'  # 股票增持
    FIELDS = [
        'EquityHolder',
        'TradedShares',
        'StartDate',
        'EndDate',
        'LaterHoldingShares',
        'AveragePrice',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityOverweightEvent.FIELDS, event_name=EquityOverweightEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'TradedShares',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityPledgeEvent(BaseEvent):
    NAME = 'EquityPledge'  # 股权质押
    FIELDS = [
        'Pledger',  # 质押者
        'PledgedShares',  # 质押股份
        'Pledgee',  # 质押人
        'TotalHoldingShares',  # 合计持股
        'TotalHoldingRatio',  # 合计持仓比率
        'TotalPledgedShares',  # 合计质押股数
        'StartDate',
        'EndDate',
        'ReleasedDate',  # 发布日期
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            EquityPledgeEvent.FIELDS, event_name=EquityPledgeEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'Pledger',
            'PledgedShares',
            'Pledgee',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


common_fields = ['StockCode', 'StockAbbr', 'CompanyName']

event_type2event_class = {
    EquityFreezeEvent.NAME: EquityFreezeEvent,
    EquityRepurchaseEvent.NAME: EquityRepurchaseEvent,
    EquityUnderweightEvent.NAME: EquityUnderweightEvent,
    EquityOverweightEvent.NAME: EquityOverweightEvent,
    EquityPledgeEvent.NAME: EquityPledgeEvent,
}

event_type_fields_list = [
    (EquityFreezeEvent.NAME, EquityFreezeEvent.FIELDS),
    (EquityRepurchaseEvent.NAME, EquityRepurchaseEvent.FIELDS),
    (EquityUnderweightEvent.NAME, EquityUnderweightEvent.FIELDS),
    (EquityOverweightEvent.NAME, EquityOverweightEvent.FIELDS),
    (EquityPledgeEvent.NAME, EquityPledgeEvent.FIELDS),
]
