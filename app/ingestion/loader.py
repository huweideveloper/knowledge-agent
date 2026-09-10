import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# 增加于阶段 3.5：记录目录批量加载中的坏文件信息。
logger = logging.getLogger(__name__)


# 增加于阶段 3.1：将单个 PDF 加载为按页拆分的 LangChain Document。
def load_pdf(path: str | Path) -> list[Document]:
    """读取单个 PDF，并将每一页转换为一个 LangChain Document。

    实现方式：先把传入路径转换为 Path 并检查文件是否存在，再使用
    LangChain 的 PyPDFLoader 调用 pypdf 解析 PDF。PyPDFLoader 默认按页
    返回 Document，并保留来源路径和页码等基础元数据。

    参数：
        path: PDF 文件路径，可以是字符串或 pathlib.Path 对象。

    返回：
        按 PDF 页拆分的 Document 列表；列表中的每个元素代表一页内容。

    异常：
        FileNotFoundError: path 不是一个存在的文件时抛出。
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(str(pdf_path)).load()


# 增加于阶段 3.4：批量加载目录下的所有 PDF。
# 修改于阶段 3.5：跳过解析失败的 PDF，并记录文件名和异常原因。
def load_directory(directory: str | Path) -> list[Document]:
    """读取目录下全部 PDF，并合并返回所有页面的 LangChain Document。

    实现方式：先把目录路径转换为 Path 并检查目录是否存在，再按文件名排序
    查找当前目录下的 PDF 文件，逐个调用 load_pdf()，最后按 PDF 文件顺序
    合并每个文件返回的页面文档。单个 PDF 解析失败时记录文件路径和异常原因，
    然后继续加载剩余 PDF。

    参数：
        directory: PDF 文件所在目录，可以是字符串或 pathlib.Path 对象。

    返回：
        所有 PDF 页面对应的 Document 列表；没有 PDF 文件时返回空列表。

    异常：
        NotADirectoryError: directory 不存在或不是目录时抛出。
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    documents: list[Document] = []
    for pdf_path in sorted(directory_path.glob("*.pdf")):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as error:  # 增加于阶段 3.5：隔离单个坏文件的解析异常。
            logger.warning("PDF 加载失败，已跳过文件 %s；原因：%s", pdf_path, error)
    return documents
