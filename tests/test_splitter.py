import importlib
from unittest import TestCase

from langchain_core.documents import Document


class DocumentSplitterTest(TestCase):
    # 增加于阶段 5.1：验证固定长度切块和 150 字符重叠。
    def test_split_documents_uses_fixed_size_and_overlap(self):
        """验证文档切块长度不超过 800，并保留 150 字符重叠。

        实现方式：加载真实的 splitter 模块，传入一份没有自然分隔符的长文档，
        检查切块数量、最大长度和相邻切块之间的重叠内容。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        try:
            splitter_module = importlib.import_module("app.ingestion.splitter")
        except ModuleNotFoundError:
            splitter_module = None

        split_documents = getattr(splitter_module, "split_documents", None)
        self.assertTrue(callable(split_documents), "文档切块函数尚未实现")

        source_text = "0123456789" * 200
        documents = [Document(page_content=source_text, metadata={"source": "test"})]

        chunks = split_documents(documents)

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk.page_content) for chunk in chunks), 800)
        self.assertEqual(chunks[1].page_content[:150], chunks[0].page_content[-150:])

    # 增加于阶段 5.2：验证每个 Chunk 都有可追溯且唯一的 ID。
    def test_split_documents_assigns_unique_chunk_ids(self):
        """验证 Chunk ID 包含文档、页码和顺序信息，并且不会重复。

        实现方式：加载真实的 splitter 模块，传入带有文档 ID 和零基页码的长
        Document，检查生成 ID 的格式、页码转换和每个 Chunk 的唯一性。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        splitter_module = importlib.import_module("app.ingestion.splitter")
        split_documents = getattr(splitter_module, "split_documents", None)
        self.assertTrue(callable(split_documents), "文档切块函数尚未实现")

        source_text = "0123456789" * 100
        documents = [
            Document(
                page_content=source_text,
                metadata={"document_id": "travel_policy_2026", "page": 11},
            )
        ]

        chunks = split_documents(documents)
        chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]

        self.assertEqual(chunk_ids[0], "travel_policy_2026_p12_chunk_01")
        self.assertEqual(chunk_ids[1], "travel_policy_2026_p12_chunk_02")
        self.assertNotIn(None, chunk_ids)
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    # 增加于阶段 5.3：验证每个 Chunk 继承文档和权限 Metadata。
    def test_split_documents_inherits_access_metadata(self):
        """验证 Chunk 保留文档标识、页码、部门和角色权限信息。

        实现方式：加载真实的 splitter 模块，传入带有业务 Metadata 的长文档，
        检查所有生成 Chunk 是否都保留完整字段和值，并继续包含 chunk_id。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        splitter_module = importlib.import_module("app.ingestion.splitter")
        split_documents = getattr(splitter_module, "split_documents", None)
        self.assertTrue(callable(split_documents), "文档切块函数尚未实现")

        documents = [
            Document(
                page_content="政策条款" * 300,
                metadata={
                    "document_id": "travel_policy_2026",
                    "page": 11,
                    "department": "finance",
                    "allowed_roles": ["employee", "manager"],
                },
            )
        ]

        chunks = split_documents(documents)

        for chunk in chunks:
            self.assertEqual(chunk.metadata["document_id"], "travel_policy_2026")
            self.assertEqual(chunk.metadata["page"], 11)
            self.assertEqual(chunk.metadata["department"], "finance")
            self.assertEqual(
                chunk.metadata["allowed_roles"],
                ["employee", "manager"],
            )
            self.assertIn("chunk_id", chunk.metadata)
