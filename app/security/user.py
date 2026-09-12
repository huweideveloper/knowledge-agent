"""阶段 12.1：定义可供权限判断使用的用户身份模型。"""

from pydantic import BaseModel, ConfigDict, Field


# 增加于阶段 12.1：定义权限过滤使用的用户身份模型。
class User(BaseModel):
    """表示权限过滤所需的用户身份信息。

    实现方式：使用 Pydantic 校验并去除字符串字段两端空白，要求 id、department
    和 role 都是非空字符串，同时拒绝未定义字段，避免后续权限规则依赖歧义输入。

    参数：
        id: 用户唯一标识，必须是非空字符串。
        department: 用户所属部门，必须是非空字符串。
        role: 用户角色，必须是非空字符串；具体可访问角色由后续权限任务判断。

    返回：
        User：包含已校验用户身份字段的 Pydantic 模型对象。

    异常：
        pydantic.ValidationError：字段类型不正确、字段为空或输入包含额外字段时抛出。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    department: str = Field(min_length=1)
    role: str = Field(min_length=1)
