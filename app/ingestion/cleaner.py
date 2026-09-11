import re
from collections.abc import Iterable


# 增加于阶段 4.1：清理文本中的多余横向空白。
def clean_extra_spaces(text: str) -> str:
    """将连续空格或制表符压缩为一个普通空格。

    实现方式：使用正则表达式只匹配连续的普通空格和制表符，并替换为一个
    空格；不匹配换行符，避免提前改变文档段落结构。连续换行的清理由后续
    阶段 4.2 负责。

    参数：
        text: 待清洗的文本字符串；必须是 str 类型。

    返回：
        清洗后的文本字符串；原文本中的换行数量和位置保持不变。

    异常：
        TypeError: text 不是字符串时抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"[ \t]+", " ", text)


# 增加于阶段 4.2：压缩文本中的连续大量换行。
def clean_repeated_newlines(text: str) -> str:
    """将连续三个及以上换行符压缩为两个换行符。

    实现方式：使用正则表达式查找连续三个及以上的换行符，并替换为两个
    换行符；单个换行和已有的段落间隔保持不变，从而保留基本段落结构。

    参数：
        text: 待清洗的文本字符串；必须是 str 类型。

    返回：
        清洗后的文本字符串；连续大量空行被压缩为一个合理的段落间隔。

    异常：
        TypeError: text 不是字符串时抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\n{3,}", "\n\n", text)


# 增加于阶段 4.3：按显式规则清理页眉和页脚。
def clean_headers_footers(
    text: str,
    header_lines: Iterable[str] = (),
    footer_lines: Iterable[str] = (),
) -> str:
    """删除文本中符合配置规则的页眉和页脚整行内容。

    实现方式：把调用方提供的页眉和页脚规则合并为规范化字符串集合，逐行
    检查去除行尾换行符和首尾空白后的内容；命中规则的行被删除，其他行按
    原样保留，包括原有换行符。规则由调用方显式提供，避免自动猜测并误删
    正文中的有效信息。

    参数：
        text: 待清洗的文本字符串；必须是 str 类型。
        header_lines: 需要删除的页眉文本集合或其他可迭代对象。
        footer_lines: 需要删除的页脚文本集合或其他可迭代对象。

    返回：
        清理页眉页脚后的文本字符串；未命中规则的正文和换行保持不变。

    异常：
        TypeError: text 不是字符串时抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    rules = {
        rule.strip()
        for rule in (*header_lines, *footer_lines)
        if rule.strip()
    }
    cleaned_lines = []
    for line in text.splitlines(keepends=True):
        line_content = line.rstrip("\r\n").strip()
        if line_content not in rules:
            cleaned_lines.append(line)
    return "".join(cleaned_lines)


# 增加于阶段 4.4：组合清洗规则并保留文档结构信息。
def clean_text(
    text: str,
    header_lines: Iterable[str] = (),
    footer_lines: Iterable[str] = (),
) -> str:
    """执行统一文本清洗流程，同时保留章节和条款结构。

    实现方式：先调用 clean_headers_footers() 去除调用方配置的页眉页脚，再
    调用 clean_extra_spaces() 压缩横向空白，最后调用 clean_repeated_newlines()
    压缩过多空行。各步骤只清理明确噪音，不删除标题、编号小节和政策正文。

    参数：
        text: 待清洗的文本字符串；必须是 str 类型。
        header_lines: 需要删除的页眉文本集合或其他可迭代对象。
        footer_lines: 需要删除的页脚文本集合或其他可迭代对象。

    返回：
        完成统一清洗后的文本字符串；章节结构和关键正文内容保持不变。

    异常：
        TypeError: text 不是字符串时由底层清洗函数抛出。
    """
    text_without_headers = clean_headers_footers(text, header_lines, footer_lines)
    text_without_extra_spaces = clean_extra_spaces(text_without_headers)
    return clean_repeated_newlines(text_without_extra_spaces)
