import math

def complexity_factor(mu_p, mu_n, mu_d, n_tools, has_output_schema,
                      wp=0.35, wn=0.15, wd=0.25, wt=0.15, ws=0.10,
                      threshold=50):
    """
    计算一个 MCP server 的 Complexity Factor (0–100)，并区分 Simple/Hard。

    参数：
    - mu_p: 平均参数数量
    - mu_n: tool name 平均 token 数
    - mu_d: tool description 平均 token 数
    - n_tools: tool 总数
    - has_output_schema: 是否有 output schema (True/False)
    - wp, wn, wd, wt, ws: 各个指标的权重 (默认加起来=1)
    - threshold: Simple/Hard 的分界值 (默认 50，可根据实验数据调整)

    返回：
    - factor: 复杂度分数 (0–100)
    - level: "Simple" 或 "Hard"
    """

    # 饱和归一化
    P = 1 - math.exp(-mu_p / 5.0)      # 参数复杂度
    N = 1 - math.exp(-mu_n / 5.0)      # 名称复杂度
    D = 1 - math.exp(-mu_d / 50.0)     # 描述复杂度
    T = 1 - math.exp(-n_tools / 10.0)  # 工具数量复杂度
    S = 0 if has_output_schema else 1  # 没有 schema 就惩罚

    # 计算加权和
    cf_raw = wp*P + wn*N + wd*D + wt*T + ws*S
    factor = round(100 * cf_raw)

    # 分类
    level = "Hard" if factor >= threshold else "Simple"

    return factor, level
