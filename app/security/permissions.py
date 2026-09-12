"""阶段 12.3：构造 Retrieval 使用的 Qdrant 权限过滤器。"""

from qdrant_client import models

from app.security.user import User


# 增加于阶段 12.3：把用户角色转换为文档 allowed_roles 的 Qdrant Filter。
def build_permission_filter(user: User) -> models.Filter:
    """根据用户角色构造只匹配允许文档的 Qdrant Filter。

    实现方式：校验输入为阶段 12.1 的 User 对象，使用 Qdrant 对数组字段的
    MatchValue 语义匹配 `metadata.allowed_roles` 中的用户 role；过滤器会在向量
    检索前由 Qdrant 应用，避免越权 Chunk 进入后续 RAG Context。

    参数：
        user: 当前请求用户，必须是包含非空 role 的 User 对象。

    返回：
        qdrant_client.models.Filter：匹配用户角色的 Qdrant 必须条件过滤器。

    异常：
        TypeError：user 不是 User 对象时抛出。
    """
    if not isinstance(user, User):
        raise TypeError("user must be a User")
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.allowed_roles",
                match=models.MatchValue(value=user.role),
            )
        ]
    )
