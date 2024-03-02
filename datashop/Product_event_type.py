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


class PurchaseProcessEvent(BaseEvent):
    NAME = 'PurchaseProcess'    # 购买过程
    FIELDS = [
        'ProductName',      # 产品名称
        'EaseOfPurchase',   # 易于购买
        'PaymentOptions',   # 付款选项
        'CustomerSupport',  # 顾客支持
    ]

    def __init__(self, recguid=None):
        super().__init__(
            PurchaseProcessEvent.FIELDS, event_name=PurchaseProcessEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EaseOfPurchase',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 物流体验事件类
class LogisticsExperienceEvent(BaseEvent):
    NAME = 'LogisticsExperience'    # 物流
    FIELDS = [
        'DeliverySpeed',     # 运输速度
        'PackageCondition',  # 包裹情况
        'DeliveryService',   # 运输服务
    ]

    def __init__(self, recguid=None):
        super().__init__(
            LogisticsExperienceEvent.FIELDS, event_name=LogisticsExperienceEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'DeliverySpeed',
            'PackageCondition',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 产品体验事件类
class ProductExperienceEvent(BaseEvent):
    NAME = 'ProductExperience'  # 产品
    FIELDS = [
        'ProductName',      # 产品名称
        'ProductQuality',   # 质量
        'SatisfactionLevel',    # 满意度
        'MeetsExpectations',    # 符合期望
        'ProductDurability',    # 产品耐用性
    ]

    def __init__(self, recguid=None):
        super().__init__(
            ProductExperienceEvent.FIELDS, event_name=ProductExperienceEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'ProductName',
            'SatisfactionLevel',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 售后服务事件类
class AfterSalesServiceEvent(BaseEvent):
    NAME = 'AfterSalesService'  # 售后服务
    FIELDS = [
        'CustomerServiceResponse',  # 客服反应
        'ProblemResolution',    # 问题解决
        'ReturnPolicy',  # 退货策略
        'WarrantyService',  # 保修服务
        'TechnicalSupport',  # 技术支持
    ]

    def __init__(self, recguid=None):
        super().__init__(
            AfterSalesServiceEvent.FIELDS, event_name=AfterSalesServiceEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'CustomerServiceResponse',
        ])

    def is_good_candidate(self, min_match_count=3):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 个人体验事件类
class PersonalExperienceEvent(BaseEvent):
    NAME = 'PersonalExperience'
    FIELDS = [
        'ExpectationMet',   # 满足期望
        'ProductReliability',   # 产品可靠性
        'UserFriendliness',  # 用户友好性
    ]

    def __init__(self, recguid=None):
        super().__init__(
            PersonalExperienceEvent.FIELDS, event_name=PersonalExperienceEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'ExpectationMet',
        ])

    def is_good_candidate(self, min_match_count=3):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 比较评价事件类
class ComparativeEvaluationEvent(BaseEvent):
    NAME = 'ComparativeEvaluation'
    FIELDS = [
        'CompetitorComparison',  # 竞品比较
        'UniqueSellingPoints',   # 卖点
        'MarketPosition',       # 市场定位
    ]

    def __init__(self, recguid=None):
        super().__init__(
            ComparativeEvaluationEvent.FIELDS, event_name=ComparativeEvaluationEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'CompetitorComparison',
        ])

    def is_good_candidate(self, min_match_count=3):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 再次购买意愿事件类
class RepurchaseIntentionEvent(BaseEvent):
    NAME = 'RepurchaseIntention'
    FIELDS = [
        'LikelihoodOfRepurchase',   # 再次购买的可能性
        'RecommendationLikelihood',  # 推荐的可能性
        'BrandLoyalty',     # 品牌忠诚
        'SatisfactionLevel',    # 满意度
        'FuturePurchasePlans',      # 未来购买计划
    ]

    def __init__(self, recguid=None):
        super().__init__(
            RepurchaseIntentionEvent.FIELDS, event_name=RepurchaseIntentionEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'LikelihoodOfRepurchase',
            'RecommendationLikelihood',
            'SatisfactionLevel',
        ])

    def is_good_candidate(self, min_match_count=2):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 建议改进事件类
class SuggestionForImprovementEvent(BaseEvent):
    NAME = 'SuggestionForImprovement'
    FIELDS = [
        'ProductFeatureSuggestions',    # 产品特性建议
        'ServiceImprovement',   # 服务改善建议
        'DesignEnhancement',    # 设计增强
        'FunctionalAdditions',  # 功能添加
        'FeedbackUtilization',  # 反馈利用
    ]

    def __init__(self, recguid=None):
        super().__init__(
            SuggestionForImprovementEvent.FIELDS, event_name=SuggestionForImprovementEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'ProductFeatureSuggestions',
            'ServiceImprovement',
        ])

    def is_good_candidate(self, min_match_count=2):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


# 情感表达事件类
class EmotionalExpressionEvent(BaseEvent):
    NAME = 'EmotionalExpression'
    FIELDS = [
        'Happiness',
        'Frustration',
        'Satisfaction',
        'Disappointment',
        'Trust',
        'Angry',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EmotionalExpressionEvent.FIELDS, event_name=EmotionalExpressionEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'Satisfaction',
            'Disappointment',
        ])

    def is_good_candidate(self, min_match_count=2):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


common_fields = ['ProductName', 'SatisfactionLevel']

event_type2event_class = {
    PurchaseProcessEvent.NAME: PurchaseProcessEvent,
    LogisticsExperienceEvent.NAME: LogisticsExperienceEvent,
    ProductExperienceEvent.NAME: ProductExperienceEvent,
    AfterSalesServiceEvent.NAME: AfterSalesServiceEvent,
    PersonalExperienceEvent.NAME: PersonalExperienceEvent,
    ComparativeEvaluationEvent.NAME: ComparativeEvaluationEvent,
    RepurchaseIntentionEvent.NAME: RepurchaseIntentionEvent,
    SuggestionForImprovementEvent.NAME: SuggestionForImprovementEvent,
    EmotionalExpressionEvent.NAME: EmotionalExpressionEvent,
}


event_type_fields_list = [
    (PurchaseProcessEvent.NAME, PurchaseProcessEvent.FIELDS),
    (LogisticsExperienceEvent.NAME, LogisticsExperienceEvent.FIELDS),
    (ProductExperienceEvent.NAME, ProductExperienceEvent.FIELDS),
    (AfterSalesServiceEvent.NAME, AfterSalesServiceEvent.FIELDS),
    (PersonalExperienceEvent.NAME, PersonalExperienceEvent.FIELDS),
    (ComparativeEvaluationEvent.NAME, ComparativeEvaluationEvent.FIELDS),
    (RepurchaseIntentionEvent.NAME, RepurchaseIntentionEvent.FIELDS),
    (SuggestionForImprovementEvent.NAME, SuggestionForImprovementEvent.FIELDS),
    (EmotionalExpressionEvent.NAME, EmotionalExpressionEvent.FIELDS),
]