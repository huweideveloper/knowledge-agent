import importlib
from unittest import TestCase


class CleanerTest(TestCase):
    # 增加于阶段 4.1：验证连续横向空白会被压缩且换行结构保留。
    def test_clean_extra_spaces_collapses_horizontal_whitespace(self):
        """验证文本清洗只压缩多余横向空白，不改变换行结构。

        实现方式：加载真实的 cleaner 模块，传入包含连续空格、制表符和空行
        的固定文本，并将返回结果与人工确定的清洗结果比较。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        try:
            cleaner_module = importlib.import_module("app.ingestion.cleaner")
        except ModuleNotFoundError:
            cleaner_module = None

        clean_extra_spaces = getattr(cleaner_module, "clean_extra_spaces", None)
        self.assertTrue(callable(clean_extra_spaces), "多余空格清洗函数尚未实现")

        raw_text = "员工     出差\t标准\n\n上海    住宿标准 600 元/晚"
        expected_text = "员工 出差 标准\n\n上海 住宿标准 600 元/晚"

        self.assertEqual(clean_extra_spaces(raw_text), expected_text)

    # 增加于阶段 4.2：验证连续大量换行会压缩为一个段落间隔。
    def test_clean_repeated_newlines_collapses_excessive_blank_lines(self):
        """验证连续大量换行会压缩为两个换行符，并保留正常换行。

        实现方式：加载真实的 cleaner 模块，传入包含段落间隔、单个换行和
        过多空行的固定文本，并将结果与人工确定的段落文本比较。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        cleaner_module = importlib.import_module("app.ingestion.cleaner")
        clean_repeated_newlines = getattr(cleaner_module, "clean_repeated_newlines", None)
        self.assertTrue(callable(clean_repeated_newlines), "重复换行清洗函数尚未实现")

        raw_text = "第一段\n\n\n\n第二段\n第三段\n\n\n第四段"
        expected_text = "第一段\n\n第二段\n第三段\n\n第四段"

        self.assertEqual(clean_repeated_newlines(raw_text), expected_text)

    # 增加于阶段 4.3：验证配置的页眉页脚规则会被清除且正文保留。
    def test_clean_headers_footers_removes_configured_lines(self):
        """验证页眉页脚规则只删除指定行，不删除正文内容。

        实现方式：加载真实的 cleaner 模块，传入包含公司页眉、资料标识、
        正文和页码页脚的固定文本，并显式提供需要清理的页眉页脚规则。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        cleaner_module = importlib.import_module("app.ingestion.cleaner")
        clean_headers_footers = getattr(cleaner_module, "clean_headers_footers", None)
        self.assertTrue(callable(clean_headers_footers), "页眉页脚清洗函数尚未实现")

        raw_text = (
            "ABC科技有限公司\n"
            "内部资料\n"
            "第三章 差旅管理\n"
            "上海住宿标准 600 元/晚\n"
            "第13页\n"
        )
        expected_text = "第三章 差旅管理\n上海住宿标准 600 元/晚\n"

        cleaned_text = clean_headers_footers(
            raw_text,
            header_lines=("ABC科技有限公司", "内部资料"),
            footer_lines=("第13页",),
        )

        self.assertEqual(cleaned_text, expected_text)

    # 增加于阶段 4.4：验证清洗后章节结构和关键条款仍然保留。
    def test_clean_text_preserves_document_structure(self):
        """验证统一清洗流程去除噪音但保留章节、子章节和关键条款。

        实现方式：加载真实的 cleaner 模块，传入带有页眉页脚、冗余空格和
        过多空行的章节文本，检查清洗结果中的结构标题和政策内容完整保留。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        cleaner_module = importlib.import_module("app.ingestion.cleaner")
        clean_text = getattr(cleaner_module, "clean_text", None)
        self.assertTrue(callable(clean_text), "统一文本清洗函数尚未实现")

        raw_text = (
            "ABC科技有限公司\n"
            "内部资料\n"
            "第三章    差旅管理\n\n\n"
            "3.1   国内出差\n\n\n"
            "3.1.1   住宿标准\n"
            "上海住宿标准 600 元/晚\n"
            "第13页\n"
        )
        expected_text = (
            "第三章 差旅管理\n\n"
            "3.1 国内出差\n\n"
            "3.1.1 住宿标准\n"
            "上海住宿标准 600 元/晚\n"
        )

        cleaned_text = clean_text(
            raw_text,
            header_lines=("ABC科技有限公司", "内部资料"),
            footer_lines=("第13页",),
        )

        self.assertEqual(cleaned_text, expected_text)
