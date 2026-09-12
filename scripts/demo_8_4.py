"""阶段 8.4：运行 20 个问题并记录检索命中基线。"""

import sys
from pathlib import Path


# 增加于阶段 8.4：支持从项目根目录直接运行 20 问题基线 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_retriever import RetrievedChunk, search


# 增加于阶段 8.4：定义覆盖各类企业制度的 20 个基线问题及正确文档 ID。
QUESTIONS = (
    ("上海住宿标准", "travel_policy_2026"),
    ("出差伙食补助标准", "travel_policy_2026"),
    ("高铁超过六小时可以买什么座位", "travel_policy_2026"),
    ("VPN密码错误几次会锁定", "vpn_manual_2026"),
    ("VPN权限最长有效期多久", "vpn_manual_2026"),
    ("VPN无法连接怎么办", "vpn_manual_2026"),
    ("年假每年有几天", "leave_policy_2026"),
    ("病假需要什么证明", "leave_policy_2026"),
    ("婚假可以休几天", "leave_policy_2026"),
    ("报销需要上传什么发票", "expense_policy_2026"),
    ("采购金额超过五千元怎么审批", "expense_policy_2026"),
    ("重复发票怎么处理", "expense_policy_2026"),
    ("员工试用期多长时间", "employee_handbook_2026"),
    ("每周最多远程办公几天", "employee_handbook_2026"),
    ("迟到后多久提交考勤更正", "employee_handbook_2026"),
    ("密码必须设置多少位", "it_service_manual_2026"),
    ("软件安装需要谁批准", "it_service_manual_2026"),
    ("电脑故障应该提交什么工单", "it_service_manual_2026"),
    ("病毒钓鱼邮件如何报告", "it_service_manual_2026"),
    ("公司账号可以借给同事使用吗", "it_service_manual_2026"),
)


# 增加于阶段 8.4：读取标准检索对象对应的业务文档 ID。
def _get_document_id(result: RetrievedChunk) -> str | None:
    """从 RetrievedChunk 中提取用于基线比对的 document_id。

    实现方式：优先读取检索 Metadata 中的 document_id；对于阶段 7.3 早期演示
    留下的简化 Point，在 Metadata 缺少 document_id 时回退使用 source。该回退
    只服务于基线统计，不修改检索结果对象或数据库数据。

    参数：
        result: 阶段 8.2 定义的标准 RetrievedChunk 对象。

    返回：
        str | None：可用于与正确文档标签比较的文档 ID；无法识别时返回 None。

    异常：
        TypeError: result 不是 RetrievedChunk 时抛出。
    """
    if not isinstance(result, RetrievedChunk):
        raise TypeError("result must be a RetrievedChunk")
    document_id = result.metadata.get("document_id")
    if isinstance(document_id, str) and document_id.strip():
        return document_id
    return result.source


# 增加于阶段 8.4：执行 20 个问题并输出 Top1/Top3/Top5 命中记录。
def run_demo() -> None:
    """运行 20 个核心问题并建立初始检索命中基线。

    实现方式：逐题调用现有 search() 获取 Top 5，提取每条结果的 document_id，
    分别判断正确文档是否位于 Top 1、Top 3 和 Top 5，最后汇总三种命中率。V1
    只记录基线，不引入人工重排、阈值调参或 LLM 判断。

    参数：
        无入参；使用 QUESTIONS 中预设的 20 个问题和正确文档标签。

    返回：
        无返回值；每题命中情况、汇总数量和验收结论通过标准输出打印。

    异常：
        RuntimeError: 任一问题没有召回正确文档，或检索结果少于 5 条时抛出。
        TypeError、ValueError: 检索参数不符合约束时由 search() 抛出。
    """
    records: list[tuple[str, str, bool, bool, bool]] = []
    for query, expected_document_id in QUESTIONS:
        results = search(query=query, top_k=5)
        if len(results) < 5:
            raise RuntimeError(f"问题未返回 Top 5 结果：{query}")
        result_document_ids = [_get_document_id(result) for result in results]
        records.append(
            (
                query,
                expected_document_id,
                expected_document_id in result_document_ids[:1],
                expected_document_id in result_document_ids[:3],
                expected_document_id in result_document_ids[:5],
            )
        )

    top1_hits = sum(record[2] for record in records)
    top3_hits = sum(record[3] for record in records)
    top5_hits = sum(record[4] for record in records)
    if top5_hits != len(QUESTIONS):
        raise RuntimeError("基线验收失败：存在正确文档未进入 Top 5")

    print("=== 阶段 8.4 20 问题检索基线 Demo ===")
    print(f"问题总数: {len(QUESTIONS)}")
    print("记录字段: 正确 document_id 是否进入 Top1 / Top3 / Top5")
    for index, (query, expected_document_id, top1, top3, top5) in enumerate(
        records,
        start=1,
    ):
        print(
            f"#{index} query={query} expected={expected_document_id} "
            f"Top1={'是' if top1 else '否'} "
            f"Top3={'是' if top3 else '否'} "
            f"Top5={'是' if top5 else '否'}"
        )
    print(f"Top 1 命中: {top1_hits}/{len(QUESTIONS)}")
    print(f"Top 3 命中: {top3_hits}/{len(QUESTIONS)}")
    print(f"Top 5 命中: {top5_hits}/{len(QUESTIONS)}")
    print("关键处理结果: 已为各类企业制度建立可重复的 Top1/Top3/Top5 检索基线")
    print("基线验收: 通过")
    print("最终输出: 已记录 20 个问题的 Top1/Top3/Top5 基线")


# 增加于阶段 8.4：提供 20 问题基线 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
