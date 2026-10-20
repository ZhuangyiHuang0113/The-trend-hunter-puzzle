from __future__ import annotations

import re
from dataclasses import dataclass

# These lexicons are weak, reusable observations about how players speak. They are not
# topic labels: a message can match several groups, and free clustering still decides
# which concrete story it belongs to.
GACHA_TERMS = {
    "抽",
    "抽卡",
    "单抽",
    "十连",
    "卡池",
    "池子",
    "角色池",
    "武器池",
    "限定池",
    "常驻池",
    "保底",
    "大保底",
    "小保底",
    "软保底",
    "硬保底",
    "歪了",
    "没歪",
    "没歪出来",
    "出金",
    "金光",
    "双金",
    "三金",
    "欧皇",
    "非酋",
    "非洲人",
    "沉船",
    "天井",
    "井票",
    "概率",
    "机率",
    "爆率",
    "掉率",
    "暗改概率",
    "暗改爆率",
    "抽数",
    "垫池",
    "垫抽",
    "首充",
    "双倍",
    "月卡",
    "大月卡",
    "通行证",
    "礼包",
    "氪金",
    "课金",
    "微氪",
    "中氪",
    "重氪",
    "零氪",
    "白嫖",
    "逼氪",
    "骗氪",
    "氪佬",
    "性价比",
    "充值",
    "返利",
    "氪",
    "歪",
}

CHARACTER_POWER_TERMS = {
    "强度",
    "数值",
    "伤害",
    "输出",
    "dps",
    "刮痧",
    "核爆",
    "爆发",
    "倍率",
    "面板",
    "攻击力",
    "暴击",
    "爆伤",
    "增伤",
    "减抗",
    "减防",
    "穿透",
    "机制",
    "手法",
    "循环",
    "转火",
    "站场",
    "后台",
    "速切",
    "主c",
    "副c",
    "辅助",
    "奶妈",
    "盾辅",
    "生存位",
    "t0",
    "t1",
    "t2",
    "tier",
    "人权",
    "人权卡",
    "必抽",
    "未来可期",
    "下毒",
    "售后",
    "膨胀",
    "数值膨胀",
    "退环境",
    "淘汰",
    "超模",
}

BALANCE_TERMS = {
    "加强",
    "增强",
    "削弱",
    "nerf",
    "buff",
    "平衡",
    "平衡性",
    "暗改",
    "偷改",
    "热修",
    "回调",
    "回滚",
    "改机制",
    "修复",
    "测试服",
    "体验服",
    "正式服",
    "公测",
    "测试服背刺",
    "背刺",
}

GRIND_TERMS = {
    "肝",
    "太肝",
    "长草",
    "坐牢",
    "上班",
    "打卡",
    "日常",
    "周常",
    "每日",
    "每周",
    "清体力",
    "体力",
    "树脂",
    "扫荡",
    "跳过",
    "自动战斗",
    "挂机",
    "重复劳动",
    "刷材料",
    "刷本",
    "刷词条",
    "毕业",
    "养成",
    "养不起",
    "任务",
}

BUILD_TERMS = {
    "装备",
    "武器",
    "专武",
    "圣遗物",
    "遗器",
    "词条",
    "主词条",
    "副词条",
    "套装",
    "毕业装",
    "配装",
    "build",
    "面板",
    "强化",
    "突破",
    "精炼",
    "命座",
    "星魂",
    "共鸣链",
    "潜能",
    "天赋",
}

COMBAT_TERMS = {
    "手感",
    "操作",
    "打击感",
    "闪避",
    "弹反",
    "格挡",
    "连招",
    "前摇",
    "后摇",
    "僵直",
    "霸体",
    "锁敌",
    "索敌",
    "镜头",
    "视角",
    "自动锁定",
    "判定",
    "判定框",
    "hitbox",
    "无敌帧",
    "qte",
    "连携",
    "换人",
    "切人",
}

TECH_TERMS = {
    "bug",
    "崩溃",
    "闪退",
    "卡顿",
    "掉帧",
    "帧数",
    "fps",
    "发热",
    "耗电",
    "内存",
    "优化",
    "加载",
    "延迟",
    "ping",
    "掉线",
    "断连",
    "回档",
    "服务器",
    "炸服",
    "排队",
    "黑屏",
    "白屏",
    "穿模",
    "贴图",
    "模型错误",
    "卡死",
    "修复",
}

STORY_TERMS = {
    "剧情",
    "主线",
    "支线",
    "世界观",
    "设定",
    "伏笔",
    "发刀",
    "刀子",
    "反转",
    "吃书",
    "崩人设",
    "人设",
    "塑造",
    "角色塑造",
    "文案",
    "编剧",
    "剧情跳过",
    "过场",
    "cg",
    "pv",
    "动画",
    "配音",
}

ART_TERMS = {
    "立绘",
    "建模",
    "模型",
    "皮肤",
    "时装",
    "衣装",
    "服装",
    "审美",
    "美术",
    "画风",
    "特效",
    "动作",
    "大招动画",
    "待机动作",
    "脸模",
    "缩水",
    "和谐",
}

EVENT_TERMS = {
    "版本",
    "更新",
    "更新后",
    "活动",
    "版本活动",
    "限时",
    "复刻",
    "联动",
    "新地图",
    "新角色",
    "新武器",
    "前瞻",
    "直播",
    "公告",
    "维护",
    "停服",
    "补偿",
    "延期",
    "跳票",
    "奖励",
    "福利",
    "兑换",
    "活动商店",
}

COMMUNITY_SLANG = {
    "绷不住",
    "蚌埠住了",
    "笑死",
    "草生",
    "逆天",
    "离谱",
    "抽象",
    "出生",
    "畜生",
    "急了",
    "急急急",
    "破防",
    "麻了",
    "麻中麻",
    "寄了",
    "丸辣",
    "完辣",
    "赢麻了",
    "输麻了",
    "狠狠地",
    "狠狠",
    "确实",
    "有一说一",
    "讲道理",
    "小丑",
    "小丑竟是我自己",
    "乐子",
    "吃瓜",
    "带节奏",
    "串子",
    "反串",
    "钓鱼",
    "节奏",
    "洗地",
    "护主",
    "水军",
    "控评",
    "岁月史书",
    "回旋镖",
    "点了",
    "支持",
    "zc",
    "doge",
}

RETENTION_TERMS = {
    "入坑",
    "萌新",
    "新手",
    "回坑",
    "回归",
    "退坑",
    "退游",
    "弃坑",
    "卸载",
    "删游",
    "跑路",
    "养老",
    "长草",
    "afk",
    "回流",
    "留存",
}

FAIRNESS_TERMS = {
    "外挂",
    "脚本",
    "锁头",
    "自瞄",
    "科技",
    "开挂",
    "作弊",
    "封号",
    "误封",
    "检测",
    "反作弊",
    "工作室",
    "刷号",
    "代练",
    "代充",
    "黑卡",
}

GAME_MODE_TERMS = {
    "pvp",
    "pve",
    "竞技场",
    "排名",
    "排行榜",
    "深渊",
    "高难",
    "终局",
    "endgame",
    "副本",
    "boss",
    "周本",
    "团本",
    "肉鸽",
    "roguelike",
    "爬塔",
    "竞速",
}

TRUST_TERMS = {
    "策划",
    "运营",
    "官方",
    "客服",
    "制作组",
    "开发组",
    "项目组",
    "听劝",
    "装死",
    "滑跪",
    "道歉",
    "回应",
    "沟通",
    "信任",
    "背刺",
    "欺骗",
    "虚假宣传",
    "画饼",
    "承诺",
    "暗改",
    "偷偷改",
}

PLAYER_SIGNAL_GROUPS: dict[str, frozenset[str]] = {
    "gacha": frozenset(GACHA_TERMS),
    "character_power": frozenset(CHARACTER_POWER_TERMS),
    "balance": frozenset(BALANCE_TERMS),
    "grind": frozenset(GRIND_TERMS),
    "build": frozenset(BUILD_TERMS),
    "combat": frozenset(COMBAT_TERMS),
    "technical": frozenset(TECH_TERMS),
    "story": frozenset(STORY_TERMS),
    "art": frozenset(ART_TERMS),
    "event": frozenset(EVENT_TERMS),
    "community_slang": frozenset(COMMUNITY_SLANG),
    "retention": frozenset(RETENTION_TERMS),
    "fairness": frozenset(FAIRNESS_TERMS),
    "game_modes": frozenset(GAME_MODE_TERMS),
    "trust": frozenset(TRUST_TERMS),
}

COMPETITOR_GAMES = {
    "原神",
    "genshin",
    "genshin impact",
    "崩坏3",
    "崩坏三",
    "崩坏3rd",
    "honkai impact 3rd",
    "崩坏星穹铁道",
    "星穹铁道",
    "星铁",
    "hsr",
    "绝区零",
    "zzz",
    "鸣潮",
    "wuthering waves",
    "wuwa",
    "战双帕弥什",
    "战双",
    "pgr",
    "深空之眼",
    "aether gazer",
    "尘白禁区",
    "snowbreak",
    "幻塔",
    "tower of fantasy",
    "二重螺旋",
    "duet night abyss",
    "明日方舟",
    "arknights",
    "胜利女神nikke",
    "nikke",
    "碧蓝航线",
    "azur lane",
    "fgo",
    "fate grand order",
}

COMPARISON_MARKERS = {
    "比",
    "不如",
    "还不如",
    "至少",
    "更",
    "没有",
    "像",
    "抄",
    "对标",
    "同样",
    "隔壁",
}
PRAISE_TERMS = {"良心", "大方", "贴心", "真棒", "太棒", "优秀", "赢麻了", "毫无压力"}
CRITICAL_CONTEXT_TERMS = {
    "逼氪",
    "骗氪",
    "暗改",
    "歪了",
    "保底",
    "坐牢",
    "刮痧",
    "削弱",
    "背刺",
    "离谱",
    "逆天",
    "退坑",
    "卸载",
    "爆率",
    "概率",
    "卡顿",
    "崩溃",
    "外挂",
}

SINGLE_CHAR_CONTEXT_PATTERNS = {
    "抽": re.compile(r"(?:\d+\s*抽|抽(?:卡|了|到|个|满|空|歪|出|数))"),
    "氪": re.compile(r"(?:氪(?:金|了|佬|多少|不起)|(?:零|微|中|重)氪)"),
    "歪": re.compile(r"(?:歪(?:了|到|出)|没歪)"),
    "肝": re.compile(r"(?:太肝|肝(?:不动|不起|死|完|材料|活动))"),
}


@dataclass(frozen=True, slots=True)
class SignalProfile:
    groups: tuple[str, ...]
    competitors: tuple[str, ...]
    flags: tuple[str, ...]


def _contains(text: str, term: str) -> bool:
    """Match phrases conservatively; ambiguous one-character slang must stand alone."""
    term = term.lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9 +.-]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    if len(term) == 1:
        contextual = SINGLE_CHAR_CONTEXT_PATTERNS.get(term)
        if contextual and contextual.search(text):
            return True
        return text == term or bool(
            re.search(rf"(?<![\u3400-\u9fff]){re.escape(term)}(?![\u3400-\u9fff])", text)
        )
    return term in text


def analyse_player_signals(text: str) -> SignalProfile:
    lowered = text.lower()
    groups = tuple(
        name
        for name, terms in PLAYER_SIGNAL_GROUPS.items()
        if any(_contains(lowered, term) for term in terms)
    )
    competitors = tuple(
        game for game in sorted(COMPETITOR_GAMES, key=len, reverse=True) if _contains(lowered, game)
    )
    # Prefer the longest alias and avoid returning both e.g. Genshin Impact and Genshin.
    competitors = tuple(
        game
        for index, game in enumerate(competitors)
        if not any(game in earlier for earlier in competitors[:index])
    )
    flags: list[str] = []
    if competitors and any(marker in lowered for marker in COMPARISON_MARKERS):
        flags.append("competitor_comparison")
    if any(term in lowered for term in PRAISE_TERMS) and (
        any(term in lowered for term in CRITICAL_CONTEXT_TERMS)
        or "community_slang" in groups
        or "<emoji:doge" in lowered
    ):
        # This is a review flag, not a sentiment reversal.
        flags.append("possible_irony")
    return SignalProfile(groups, competitors, tuple(flags))
