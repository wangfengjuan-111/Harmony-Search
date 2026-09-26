# -*- coding: utf-8 -*-
r"""
===============================================================================
 Solomon C101 带时间窗车辆路径问题 (VRPTW) 求解与对比实验
--------------------------------------------------------------------------------
 实现三种元启发式算法并做统一对比：
   1. 和声搜索  (Harmony Search, HS)
   2. 遗传算法  (Genetic Algorithm, GA)
   3. 禁忌搜索  (Tabu Search, TS)

 统一求解框架（三种算法共用）：
   - 解的表示：巨路线 (Giant Tour)，即 100 个客户的一个排列
   - 解码器  ：Split 最优切分（Prins, 2004）
       把巨路线用动态规划切成若干条「满足容量 + 时间窗约束」的子路线，
       对给定排列而言，Split 能在 O(n^2) 内求出最优的路线划分，
       因此三种算法都只需在排列空间上搜索，评价函数完全一致，公平可比。

 目标函数（字典序 / 加权）：
   cost = 车辆数 * 10000 + 总行驶距离
   即优先最小化车辆数，其次最小化距离（Solomon 基准的标准做法）。
   距离采用 Solomon 官方约定：欧氏距离向下截断到 1 位小数。
   C101 已知最优解：10 辆车，距离 827.3。

 对比指标（与题目要求一一对应）：
   - Best Cost ：500 次迭代中找到的最小目标函数值
   - Avg Cost  ：独立运行 10 次的平均最优值
   - Std       ：10 次最优值的标准差（衡量稳定性）
   - 运行时间  ：单次运行的平均秒数
   - 收敛迭代  ：首次达到「最终最优值」的迭代次数（0 = 初始解即已达到，
                 或整个过程无改进）。原口径「首次达到最优值 90%」在最小化
                 问题中无法直接使用（cost <= 0.9 * 最优不可能发生），
                 详见 convergence_iter() 注释。取 10 次运行的平均值。

 运行方式：
   python c101_solver.py                          # 默认读取 F:\Download\C101.txt
   python c101_solver.py --alg HS                 # 只跑和声搜索
   python c101_solver.py --iter 500 --runs 10     # 指定迭代数与重复次数
   python c101_solver.py --init i1                # 只跑「全部 I1 解」初始策略
                                                  # (默认 both：两种策略都跑并对比)
================================================================================
"""

import math
import random
import time
import argparse
from statistics import mean, stdev

# ---- 绘图（缺 matplotlib 时只跳过画图，不影响求解） ----
try:
    import matplotlib
    # 适配Streamlit Cloud 中文
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号变成方框
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")            # 无界面环境也能保存 PNG
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# 算法可调参数（集中在顶部，便于实验调参）
MAX_ITER        = 500    # 每次运行的最大迭代次数（三种算法统一）
RUNS            = 10     # 独立重复运行次数
BASE_SEED       = 20260923

# ---- 和声搜索 (HS) 参数 ----
HMS   = 40      # 和声记忆库大小 (Harmony Memory Size)
HMCR  = 0.90    # 和声记忆考虑率 (Harmony Memory Considering Rate)
PAR   = 0.30    # 音调调节率 (Pitch Adjusting Rate)

# ---- 遗传算法 (GA) 参数 ----
POP_SIZE = 40   # 种群规模
P_CROSS  = 0.9  # 交叉概率
P_MUT    = 0.2  # 变异概率（个体级）
ELITE    = 2    # 精英保留数量

# ---- 禁忌搜索 (TS) 参数 ----
TABU_TENURE = 15     # 禁忌期限（多少迭代内禁止撤销该移动）
TS_NEIGHBORS = 120   # 每次迭代随机采样的邻域移动数（与 GA/HS 每迭代的
                     # 采样局部搜索预算 120 对齐，保证三算法计算量对等）
TS_INIT_CANDIDATES = 10   # 初始解候选数：HS/GA 各取 HMS/POP 个 I1 初始解的
                          # 最优（约 11 辆车），TS 若只抽 1 个 I1 解，起点受
                          # 单次抽样波动影响（常见 12~14 辆车），收敛曲线起点
                          # 远高于另两者。取 10 个候选的最优以对齐初始解质量。

VEHICLE_WEIGHT = 10000.0   # 车辆数在加权目标中的权重

# ---- 初始解策略（实验变量，main 中在两种模式间切换对比） ----
INIT_MODE       = "i1"    # "i1" = 全部用 I1 解; "mix" = 70% I1 + 30% 强扰动
INIT_MIX_RATIO  = 0.70    # mix 模式下 I1 解所占比例
HEAVY_RAND_RATE = 0.35    # 强扰动强度（对比 I1 轻扰动的默认 0.06）


# ==============================================================================
# 1. 数据解析
# ==============================================================================
def load_solomon(path):
    """
    解析 Solomon 标准格式实例文件。

    文件格式示例：
        C101                      <- 实例名
        VEHICLE
        NUMBER     CAPACITY
          25         200           <- 车辆数上限 / 载重容量
        CUSTOMER
        CUST NO.  XCOORD. ...    <- 表头
           0      40   50  0  0  1236  0     <- 仓库 (客户 0)
           1      45   68  10 912 967  90     <- 客户 1..100

    返回:
        inst: dict，包含
            n        : 客户总数（不含仓库）
            cap      : 车辆载重
            vehicles : 可用车辆数上限
            demand[k] / ready[k] / due[k] / service[k] : 各节点属性（0 号为仓库）
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip() != ""]

    inst = {"name": lines[0].strip()}

    # --- 读取车辆信息：NUMBER 与 CAPACITY 所在行的下一行 ---
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("NUMBER"):
            vals = lines[idx + 1].split()
            inst["vehicles"] = int(vals[0])
            inst["cap"] = int(vals[1])
            break

    # --- 读取客户信息：表头 "CUST NO." 之后的每一行 ---
    coords, demand, ready, due, service = [], [], [], [], []
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("CUST NO."))
    for ln in lines[start + 1:]:
        p = ln.split()
        if len(p) < 7:
            continue
        coords.append((float(p[1]), float(p[2])))  # (x, y)
        demand.append(float(p[3]))                 # 需求量
        ready.append(float(p[4]))                  # 最早开始服务时间
        due.append(float(p[5]))                    # 最晚开始服务时间
        service.append(float(p[6]))                # 服务时长

    inst["coords"] = coords
    inst["demand"] = demand
    inst["ready"] = ready
    inst["due"] = due
    inst["service"] = service
    inst["n"] = len(coords) - 1                    # 减去仓库
    return inst


# ==============================================================================
# 2. 公共数据结构（全局缓存，供解码器高频访问）
# ==============================================================================
N = 0            # 客户数
CAP = 0          # 容量
MAX_VEH = 0      # 车辆数上限
dist = []        # 距离矩阵 (N+1)x(N+1)，dist[a][b]，0 号为仓库
demand = ready = due = service = []   # 各节点属性列表（索引 0 为仓库）
COORDS = []      # 各节点坐标 (x, y)，索引 0 为仓库
INF = float("inf")


def build_instance(inst):
    """把解析出的实例装入全局变量，并预计算距离矩阵。"""
    global N, CAP, MAX_VEH, dist, demand, ready, due, service, COORDS
    N = inst["n"]
    CAP = inst["cap"]
    MAX_VEH = inst["vehicles"]
    demand, ready, due, service = inst["demand"], inst["ready"], inst["due"], inst["service"]
    COORDS = inst["coords"]

    # Solomon 官方约定：距离 = 欧氏距离向下截断到 1 位小数
    # 例如 sqrt(200)=14.142... -> 14.1。这是与已知最优解 827.3 可比的唯一口径。
    pts = inst["coords"]
    m = len(pts)
    dist = [[0.0] * m for _ in range(m)]
    for a in range(m):
        xa, ya = pts[a]
        row_a = dist[a]
        for b in range(a + 1, m):
            xb, yb = pts[b]
            d = int(math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) * 10) / 10.0  # 截断
            row_a[b] = d
            dist[b][a] = d


# ==============================================================================
# 3. 解码器：Split 最优切分（三种算法共用的评价函数）
# ==============================================================================
def split(perm):
    """
    对巨路线 perm（客户的排列，长度 N）做最优切分。

    原理（Prins 2004 的 Split 过程）：
      把巨路线看作 (0, p1, p2, ..., pN, 0) 的一条辅助有向图，
      弧 (i, j) 表示「用一条新路线服务客户 p[i+1..j]」。
      该弧可行当且仅当这段客户序列满足：
        - 总需求 <= 车辆容量
        - 按顺序到达每个客户时，可在其时间窗内开始服务
          （到达早于 ready 则原地等待，即允许等待）
      弧的代价 = 10000（车辆数惩罚）+ 该路线的行驶距离。
      用动态规划求从 0 到 N 的最短路，即得该排列的最优路线划分。

    由于每条路线的代价都包含同一常数 10000，最短路自动实现
    「先少用车、再省距离」的字典序目标。

    返回:
        vehicles : 使用的车辆数
        distance : 总行驶距离
        routes   : 路线列表，每条是不含仓库的客户编号列表
    """
    dp = [INF] * (N + 1)     # dp[i] = 服务前 i 个客户的最小加权代价
    prev = [0] * (N + 1)     # 记录最短路的前驱，用于还原路线
    dp[0] = 0.0

    # 局部变量缓存，避免循环内反复做全局查找（Python 提速关键）
    D = dist
    dem = demand
    rdy = ready
    dd = due
    svc = service
    W = VEHICLE_WEIGHT

    for i in range(N):
        base = dp[i]
        if base == INF:
            continue
        # 从位置 i 出发，逐个向后扩展客户，构造一条可能的子路线
        load = 0.0       # 当前子路线的累计载重
        t = 0.0          # 离开上一个节点的时间（0 = 仓库出发时刻）
        last = 0         # 上一个节点编号（0 = 仓库）
        rd = 0.0         # 当前子路线的累计距离（不含回仓）
        for j in range(i, N):
            c = perm[j]
            # ---- 容量约束：装不下就停止扩展 ----
            nload = load + dem[c]
            if nload > CAP:
                break
            load = nload
            # ---- 时间窗约束：到达 -> 可能等待 -> 最晚开始时刻检查 ----
            arrive = t + D[last][c]          # 到达 c 的时刻
            st = rdy[c] if rdy[c] > arrive else arrive  # 开始服务 = max(ready, 到达)
            if st > dd[c]:                   # 超过 due 则该子路线不可行，停止扩展
                break
            t = st + svc[c]                  # 离开时刻 = 开始服务 + 服务时长
            rd += D[last][c]
            last = c
            # 尝试用「新路线 = perm[i..j]」更新 dp[j+1]
            cand = base + W + rd + D[c][0]   # + 回到仓库的距离
            if cand < dp[j + 1]:
                dp[j + 1] = cand
                prev[j + 1] = i

    # ---- 回溯还原路线划分 ----
    routes = []
    j = N
    while j > 0:
        i = prev[j]
        routes.append(perm[i:j])
        j = i
    routes.reverse()

    vehicles = len(routes)
    distance = dp[N] - vehicles * W
    return vehicles, distance, routes


def cost_of(perm):
    """只返回加权代价（车辆数*10000 + 距离），供搜索内部快速快速使用。"""
    v, d, _ = split(perm)
    return v * VEHICLE_WEIGHT + d


# ==============================================================================
# 3.5 初始解构造：随机化 Solomon 贪心插入启发式
# ==============================================================================
def greedy_init(rng, rand_rate=0.06, mu=1.0, lam=1.0):
    """
    随机化的 Solomon I1 贪心插入构造启发式，生成一组可行路线。

    C101 是「紧时间窗 + 地理聚簇」实例，纯随机排列解码出 20~50 条路线，
    搜索很难在 500 迭代内爬回 10 辆车水平；因此三种算法统一从质量良好、
    且带随机扰动的贪心解出发（rand_rate 控制扰动强度以保证多样性）。

    I1 准则（Solomon 1987）：
      - 弧插入代价  c11(i,u,j) = d(i,u) + d(u,j) - μ·d(i,j)
        （衡量把 u 插进弧 (i,j) 的「弯曲程度」）
      - 客户吸引力  c12(u) = λ·d(0,u) - min_pos c11(i,u,j)
        （离仓库越远、且能便宜地插进现有路线的客户越优先被插入）
      每一步选 c12 最大的可行客户 u，插入到其 c11 最小的可行位置。

    随机化扰动：
      - 以 rand_rate 概率把「种子客户」从 due 最早者改为随机未分配客户；
      - 以 rand_rate 概率把「被插客户」从 c12 最优者改为随机可行客户。

    返回: 路线列表（每条为不含仓库的客户列表）
    """
    unrouted = set(range(1, N + 1))
    routes = []
    while unrouted:
        # ---- 第 1 步：选种子客户开启新路线（紧迫者优先 + 随机扰动） ----
        if rng.random() < rand_rate:
            seed = rng.choice(tuple(unrouted))
        else:
            seed = min(unrouted, key=lambda c: (due[c], ready[c]))
        unrouted.discard(seed)
        route = [seed]

        # ---- 第 2 步：按 I1 准则反复插入可行客户 ----
        while True:
            best_u, best_pos, best_c12 = None, None, -INF
            pool = []                      # 所有有可行插入位置的客户（供随机扰动）
            for u in unrouted:
                # 找 u 的最优可行插入位置（c11 最小且插入后整条路线仍可行）
                c11_best, pos_best = INF, None
                for pos in range(len(route) + 1):
                    i = route[pos - 1] if pos > 0 else 0          # 前驱（0=仓库）
                    j = route[pos] if pos < len(route) else 0     # 后继（0=仓库）
                    c11 = dist[i][u] + dist[u][j] - mu * dist[i][j]
                    if c11 >= c11_best:
                        continue
                    # 可行性检查：插入 u 会顺延后继客户，需整条路线重查时间窗
                    if _route_feasible(route[:pos] + [u] + route[pos:]):
                        c11_best, pos_best = c11, pos
                if pos_best is None:
                    continue               # u 无法插入当前路线
                pool.append(u)
                c12 = lam * dist[0][u] - c11_best
                if c12 > best_c12:
                    best_c12, best_u, best_pos = c12, u, pos_best
            if best_u is None:
                break                      # 当前路线无法再吸收任何客户，关闭它
            if rng.random() < rand_rate and pool:
                best_u = rng.choice(pool)  # 随机可行客户（扰动）
                # 该客户取其自身的最优可行位置
                c11_best, best_pos = INF, None
                for pos in range(len(route) + 1):
                    i = route[pos - 1] if pos > 0 else 0
                    j = route[pos] if pos < len(route) else 0
                    c11 = dist[i][best_u] + dist[best_u][j] - mu * dist[i][j]
                    if c11 < c11_best and _route_feasible(route[:pos] + [best_u] + route[pos:]):
                        c11_best, best_pos = c11, pos
            route = route[:best_pos] + [best_u] + route[best_pos:]
            unrouted.discard(best_u)
        routes.append(route)
    return routes


def _route_feasible(cand):
    """
    检查路线序列 cand 是否可行：
      容量不超限 + 从仓库出发顺推时刻（允许在客户处等待），
      每个客户都能在其 due 之前开始服务。
    """
    load = 0.0
    t = 0.0
    prev = 0
    for c in cand:
        load += demand[c]
        if load > CAP:
            return False
        arrive = t + dist[prev][c]
        st = ready[c] if ready[c] > arrive else arrive
        if st > due[c]:
            return False
        t = st + service[c]
        prev = c
    return True


def routes_to_perm(routes):
    """把路线列表拼接成巨路线排列（解码器 split 能将其还原为同样划分）。"""
    perm = []
    for r in routes:
        perm.extend(r)
    return perm


def initial_routes(rng):
    """
    按当前全局 INIT_MODE 生成一个初始解（路线集合）。

      - "i1" ：全部用随机化 I1 贪心解（轻扰动 rand_rate=0.06，高质量）
      - "mix"：以 INIT_MIX_RATIO=70% 概率用 I1 解（保留高质量），
               其余 30% 对 I1 解做较大扰动（rand_rate=0.35，大幅提升多样性）。
               纯随机排列在 C101 上会解码出 20~50 条路线，搜索难以在
               有限迭代内爬回 10 车水平，故 30% 部分采用强扰动 I1 而非纯随机。
    """
    if INIT_MODE == "mix" and rng.random() > INIT_MIX_RATIO:
        return greedy_init(rng, rand_rate=HEAVY_RAND_RATE)
    return greedy_init(rng)


def initial_perm(rng):
    """initial_routes 的巨路线（排列）版本，供 HS / GA 初始化使用。"""
    return routes_to_perm(initial_routes(rng))


# ==============================================================================
# 3.6 路线集合邻域与局部搜索（三种算法共用的搜索机制）
# ==============================================================================
def route_dist(seq):
    """单条路线的行驶距离（含从仓库出发与返回仓库）。"""
    d = dist[0][seq[0]]
    for a, b in zip(seq, seq[1:]):
        d += dist[a][b]
    return d + dist[seq[-1]][0]


def eval_routes(routes):
    """评价一个路线集合：返回 (车辆数, 总距离, 加权代价)。"""
    v = len(routes)
    d = sum(route_dist(r) for r in routes)
    return v, d, v * VEHICLE_WEIGHT + d


def sample_neighbor(routes, rng):
    """
    在路线集合上随机采样一个邻域移动，生成候选解。
    返回 (新路线集合, 禁忌签名)；若移动不可行（违反容量/时间窗）返回 (None, None)。

    四类经典 VRPTW 移动：
      relocate : 把 1~3 个连续客户搬到任意路线的任意位置（可跨路线，
                 是减少车辆数的主要手段；同一路线内等价于 Or-opt）
      swap     : 交换两个客户（可跨路线）
      2-opt*   : 两条路线在各自某位置断开并交换尾部，连接方式不变，
                 是消除路线间交叉的主要手段
      2-opt    : 单条路线内倒置一段，消除路线内交叉
    """
    rts = [r[:] for r in routes]
    m = len(rts)
    kind = rng.random()

    if kind < 0.40:                       # ---- relocate（段长 1~3） ----
        src = rng.randrange(m)
        if m == 1 and len(rts[src]) <= 1:
            return None, None
        L = rng.randint(1, min(3, len(rts[src])))
        a = rng.randrange(len(rts[src]) - L + 1)
        seg = rts[src][a:a + L]
        rest = rts[src][:a] + rts[src][a + L:]
        # 目标路线：允许同一条（等价 Or-opt）；若源路线被搬空则只能选其他路线
        choices = [k for k in range(m) if k != src] if not rest else list(range(m))
        dst = rng.choice(choices)
        if dst == src:
            # 同一路线内搬移：在 rest 中重新插入 seg
            pos = rng.randrange(len(rest) + 1)
            new_seq = rest[:pos] + seg + rest[pos:]
            if not _route_feasible(new_seq):
                return None, None
            out = [new_seq if k == src else r for k, r in enumerate(rts)]
        else:
            pos = rng.randrange(len(rts[dst]) + 1)
            new_seq = rts[dst][:pos] + seg + rts[dst][pos:]
            if not _route_feasible(new_seq):
                return None, None
            out = []
            for k, r in enumerate(rts):
                if k == src:
                    if rest:                 # 源路线被搬空则删除
                        out.append(rest)
                elif k == dst:
                    out.append(new_seq)
                else:
                    out.append(r)
        return out, ("mv", tuple(sorted(seg)))

    elif kind < 0.60:                     # ---- swap（交换两个客户） ----
        r1 = rng.randrange(m)
        r2 = rng.randrange(m)
        i = rng.randrange(len(rts[r1]))
        j = rng.randrange(len(rts[r2])) if r2 != r1 else rng.randrange(len(rts[r2]))
        if r1 == r2 and i == j:
            return None, None
        c1, c2 = rts[r1][i], rts[r2][j]
        rts[r1][i], rts[r2][j] = c2, c1
        if not _route_feasible(rts[r1]) or not _route_feasible(rts[r2]):
            return None, None
        return rts, ("sw", frozenset((c1, c2)))

    elif kind < 0.80:                     # ---- 2-opt*（交换两路线尾部） ----
        if m < 2:
            return None, None
        r1, r2 = rng.sample(range(m), 2)
        a = rng.randint(1, len(rts[r1]))     # r1 保留前 a 个
        b = rng.randint(1, len(rts[r2]))     # r2 保留前 b 个
        new1 = rts[r1][:a] + rts[r2][b:]
        new2 = rts[r2][:b] + rts[r1][a:]
        if not new1 or not new2:
            return None, None
        if not _route_feasible(new1) or not _route_feasible(new2):
            return None, None
        sig = ("x", frozenset((rts[r1][a - 1], rts[r2][b - 1])))
        out = []
        for k, r in enumerate(rts):
            if k == r1:
                out.append(new1)
            elif k == r2:
                out.append(new2)
            else:
                out.append(r)
        return out, sig

    else:                                 # ---- 2-opt（路线内倒置） ----
        r1 = rng.randrange(m)
        if len(rts[r1]) < 3:
            return None, None
        a, b = sorted(rng.sample(range(len(rts[r1]) + 1), 2))
        new1 = rts[r1][:a] + rts[r1][a:b][::-1] + rts[r1][b:]
        if not _route_feasible(new1):
            return None, None
        sig = ("v", rts[r1][a] if a < len(rts[r1]) else 0, rts[r1][b - 1])
        rts[r1] = new1
        return rts, sig


def improve_routes(routes, rng, budget=120):
    """
    共享的「采样式局部搜索」：反复随机采样邻域移动，接受第一个改进解，
    直到连续一整轮（budget//4 次采样）无改进或总采样数达到 budget。
    三种算法以完全相同的参数调用，保证计算量与搜索强度对等。
    返回 (改进后的路线集合, 其代价)。
    """
    cur = [r[:] for r in routes]
    cur_cost = eval_routes(cur)[2]
    used = 0
    while used < budget:
        round_no_improve = True
        sweep = budget // 4
        for _ in range(sweep):
            cand, _sig = sample_neighbor(cur, rng)
            used += 1
            if cand is None:
                continue
            c = eval_routes(cand)[2]
            if c < cur_cost - 1e-9:
                cur, cur_cost = cand, c
                round_no_improve = False
            if used >= budget:
                break
        if round_no_improve:
            break
    return cur, cur_cost



# ==============================================================================
# 4. 和声搜索 (Harmony Search)
# ==============================================================================
def harmony_search(max_iter=MAX_ITER, hms=HMS, hmcr=HMCR, par=PAR, seed=0):
    """
    和声搜索模拟乐手即兴演奏的过程：
      - 和声记忆库 HM   ：保存至今最好的 HMS 个解（相当于种群）
      - 记忆考虑 (HMCR) ：以概率 HMCR 从记忆库中「单个」donor 和声按顺序
                          继承基因 —— 利用历史经验（见下方实现要点）
      - 音调调节 (PAR)  ：解码成路线集合后，以概率 PAR 做 1~3 次
                          随机可行邻域移动 —— 局部扰动（见主循环注释）
      - 随机选择 (1-HMCR)：从全部取值中随机取 —— 全局探索
      - 多样化：连续 60 次迭代无改进，对历史最优加扰动再抛光注回库
                （与 TS 的重启机制对齐，防止库顶被初始解永久占据）
      - 每次即兴出一个新和声，若优于库中最差解则替换之

    排列编码下的实现要点：
      - 记忆考虑采用「单 donor 顺序继承」：每次即兴只选一个库中和声为
        donor，逐位置以概率 HMCR 沿 donor 环形扫描取下一个未使用的基因
        （等价于与单个父代做 OX 式重组，能完整保留其路线片段）。
        排列编码下不适合教科书式的「每个位置从随机 donor 借同一位置基因」：
        多个 donor 的基因按位置拼接会打碎路线片段结构。
      - 音调调节实现为：基因级 = 与最近 4 位内的近邻位置交换（局部微调）；
        路线级 = 对解码后的路线集合做 1~3 次邻域移动尝试，仅接受改进。
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # ---- 初始化：HMS 个初始解（巨路线），按 INIT_MODE 生成 ----
    hm = []
    for _ in range(hms):
        p = initial_perm(rng)
        hm.append((cost_of(p), p))
    hm.sort(key=lambda x: x[0])
    best_cost, best_perm = hm[0]
    history = [best_cost]
    stall = 0                        # 距上次全局最优改进的迭代数

    # ---- 主循环：每次迭代即兴一首新和声 ----
    for it in range(1, max_iter + 1):
        # --- 单 donor 顺序继承：本次即兴只从一个库中和声借基因 ---
        donor = hm[rng.randrange(hms)][1]
        ptr = 0                          # donor 上的环形扫描指针
        new_p = []
        remaining = set(range(1, N + 1))  # 尚未填入的客户集合（客户编号 1..N）
        for k in range(N):
            # --- 记忆考虑：沿 donor 顺序取下一个未使用基因（保留路线片段） ---
            if rng.random() < hmcr:
                gene = donor[ptr]
                while gene not in remaining:
                    ptr = (ptr + 1) % N
                    gene = donor[ptr]
                ptr = (ptr + 1) % N
            # --- 随机选择：从全部未用客户中随机取（全局探索） ---
            else:
                gene = rng.choice(tuple(remaining))
            new_p.append(gene)
            remaining.discard(gene)
            # --- 音调调节：与近邻位置（最近 4 位内）交换，实现局部微调 ---
            #     （与任意靠前位置交换破坏性太强，会打碎 donor 的路线片段）
            if rng.random() < par:
                j = rng.randrange(max(0, len(new_p) - 4), len(new_p))
                new_p[-1], new_p[j] = new_p[j], new_p[-1]

        # --- 评价：解码 + 共享采样局部搜索（Lamarckian，改进写回新和声） ---
        _, _, rts = split(new_p)
        rts, c = improve_routes(rts, rng)
        # --- 音调调节（路线级）：以 PAR 概率对解码后的路线集合做
        #     1~3 次邻域移动尝试，仅接受有改进的移动（局部微调而非扰动） ---
        if rng.random() < par:
            for _ in range(rng.randint(1, 3)):
                cand, _sig = sample_neighbor(rts, rng)
                if cand is not None:
                    c_cand = eval_routes(cand)[2]
                    if c_cand < c:
                        rts, c = cand, c_cand
        new_p = routes_to_perm(rts)

        # --- 更新记忆库：优于库中最差则替换，保持库按代价升序 ---
        if c < hm[-1][0]:
            hm[-1] = (c, new_p)
            hm.sort(key=lambda x: x[0])
        # --- 更新全局最优 ---
        if c < best_cost:
            best_cost, best_perm = c, new_p[:]
            stall = 0
        else:
            stall += 1
        # --- 多样化：连续 60 次迭代无改进，则对历史最优做 3 次随机扰动
        #     再抛光，把结果注回记忆库（与 TS 的重启机制对齐）。没有这一步，
        #     库顶会被初始 I1 解永久占据：新和声是 donor 的劣化拷贝，
        #     永远挤不进库顶，搜索将完全退化为「报告初始解」 ---
        if stall >= 60:
            stall = 0
            rts2 = split(best_perm)[2]
            for _ in range(3):
                cand, _k = sample_neighbor(rts2, rng)
                if cand is not None:
                    rts2 = cand
            # 抛光预算 2400：HS 缺少 TS 那样的跨迭代累积下降，单靠 120 预算
            # 的常规抛光无法从扰动中恢复并突破局部最优，故重启时给足预算
            rts2, c2 = improve_routes(rts2, rng, budget=2400)
            if c2 < hm[-1][0]:
                hm[-1] = (c2, routes_to_perm(rts2))
                hm.sort(key=lambda x: x[0])
            if c2 < best_cost:
                best_cost, best_perm = c2, routes_to_perm(rts2)[:]
        history.append(best_cost)

    v, d, routes = split(best_perm)
    return {"cost": best_cost, "vehicles": v, "distance": d,
            "routes": routes, "history": history,
            "time": time.perf_counter() - t0}


# ==============================================================================
# 5. 遗传算法 (Genetic Algorithm)
# ==============================================================================
def _ox(p1, p2, rng):
    """
    顺序交叉 (Order Crossover, OX)：排列编码的标准交叉算子。
      1. 在父代 1 上随机截取一段 [a, b)，原样复制给子代；
      2. 从父代 2 中按顺序取出「不在该段中」的剩余客户，
         从 b 开始绕圈填入子代空位。
    这样能大量保留父代中相邻客户对的相对次序（路线片段）。
    """
    n = len(p1)
    a, b = sorted(rng.sample(range(n + 1), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    used = set(child[a:b])
    # 从父代 2 的位置 b 开始环形扫描，填入未使用的客户
    fill = [g for g in p2[b:] + p2[:b] if g not in used]
    k = 0
    for i in range(b, n):
        child[i] = fill[k]; k += 1
    for i in range(0, a):
        child[i] = fill[k]; k += 1
    return child


def _mutate(p, rng):
    """
    变异算子：随机片段倒位 (Inversion)。
    倒位对路径类问题特别有效——它同时改变片段两端与外界的连接方式，
    而片段内部相邻关系保留，相当于一次大步长的 2-opt 邻域移动。
    """
    a, b = sorted(rng.sample(range(len(p) + 1), 2))
    p[a:b] = reversed(p[a:b])


def genetic_algorithm(max_iter=MAX_ITER, pop_size=POP_SIZE, pc=P_CROSS,
                      pm=P_MUT, elite=ELITE, seed=0):
    """
    标准置换型遗传算法：
      - 初始化：随机排列种群
      - 选择  ：锦标赛选择（规模 3）
      - 交叉  ：顺序交叉 OX，概率 pc
      - 变异  ：片段倒位，概率 pm
      - 保留  ：精英策略，每代最优 elite 个个体直接进入下一代
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # ---- 初始化种群：按 INIT_MODE 生成（质量与多样性的权衡） ----
    pop = []
    for _ in range(pop_size):
        p = initial_perm(rng)
        pop.append((cost_of(p), p))
    pop.sort(key=lambda x: x[0])
    best_cost, best_perm = pop[0]
    history = [best_cost]

    # ---- 主循环：逐代进化 ----
    for it in range(1, max_iter + 1):
        next_pop = pop[:elite]                      # 精英直接保留

        # 生成后代直到补满种群
        children = []
        while len(children) < pop_size - elite:
            # 锦标赛选择：各取 3 个随机个体，留下代价最小者
            def tourney():
                cands = rng.sample(pop, 3)
                return min(cands, key=lambda x: x[0])[1]
            pa, pb = tourney(), tourney()
            # 交叉
            if rng.random() < pc:
                child = _ox(pa, pb, rng) if rng.random() < 0.5 else _ox(pb, pa, rng)
            else:
                child = pa[:]                        # 不交叉则克隆父代
            # 变异
            if rng.random() < pm:
                _mutate(child, rng)
            children.append(child)

        # 模因处理 (Memetic)：对本代最优后代做共享采样局部搜索，
        # 改进结果写回（Lamarckian），使其以改进后的形态参与后续进化
        best_child = min(children, key=lambda p: cost_of(p))
        _, _, rts = split(best_child)
        rts, c_best = improve_routes(rts, rng)
        next_pop.append((c_best, routes_to_perm(rts)))

        # 其余后代直接评价（仅 split 解码，控制计算量）
        for child in children:
            if child is best_child:
                continue
            next_pop.append((cost_of(child), child))

        pop = sorted(next_pop, key=lambda x: x[0])
        if pop[0][0] < best_cost:
            best_cost, best_perm = pop[0][0], pop[0][1][:]
        history.append(best_cost)

    v, d, routes = split(best_perm)
    return {"cost": best_cost, "vehicles": v, "distance": d,
            "routes": routes, "history": history,
            "time": time.perf_counter() - t0}


# ==============================================================================
# 6. 禁忌搜索 (Tabu Search)
# ==============================================================================
def tabu_search(max_iter=MAX_ITER, tenure=TABU_TENURE,
                n_neighbors=TS_NEIGHBORS, seed=0):
    """
    禁忌搜索直接在「路线集合」解空间上运行，用共享邻域 (sample_neighbor)
    生成候选移动，并用「禁忌表」阻止刚做过的移动在短期内被撤销，从而
    跳出局部最优：
      - 每次迭代随机采样 n_neighbors 个可行候选移动，选其中最优者执行；
      - 禁忌表：记录移动签名 (类型 + 涉及客户)，tenure 代内禁止重复执行；
      - 特赦准则 (Aspiration)：候选优于历史最优时无视禁忌；
      - 多样化：连续 60 次迭代无改进，则从历史最优出发加随机扰动重启。
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # ---- 初始解：与 HS/GA 对齐，从若干个随机化 I1 候选中取最优者 ----
    #      （单一 I1 解的车辆数受抽样波动影响大，起点会比 HS/GA 高 1~3 辆车）
    cur, cur_cost = None, INF
    for _ in range(TS_INIT_CANDIDATES):
        cand = initial_routes(rng)
        c = eval_routes(cand)[2]
        if c < cur_cost:
            cur, cur_cost = cand, c
    best_routes, best_cost = [r[:] for r in cur], cur_cost
    history = [best_cost]
    tabu = {}   # 移动签名 -> 解禁迭代号

    # ---- 主循环 ----
    for it in range(1, max_iter + 1):
        best_mv_cost, best_mv_routes, best_mv_key = INF, None, None
        # 随机采样邻域，找当前迭代的最优可用移动
        attempts = 0
        got = 0
        while got < n_neighbors and attempts < n_neighbors * 3:
            attempts += 1
            cand, key = sample_neighbor(cur, rng)
            if cand is None:
                continue                     # 不可行移动，重新采样
            got += 1
            c = eval_routes(cand)[2]
            # 特赦：优于历史最优的移动无视禁忌；否则禁忌中的移动跳过
            if key in tabu and tabu[key] > it and c >= best_cost:
                continue
            if c < best_mv_cost:
                best_mv_cost, best_mv_routes, best_mv_key = c, cand, key

        # 执行选中的移动，并登记禁忌（tenure 代内不得再做同一签名移动）
        if best_mv_routes is not None:
            cur, cur_cost = best_mv_routes, best_mv_cost
            tabu[best_mv_key] = it + tenure

        # 更新历史最优
        if cur_cost < best_cost:
            best_cost, best_routes = cur_cost, [r[:] for r in cur]

        # 多样化：连续 60 次迭代无改进，则从历史最优加扰动重启
        if it % 60 == 0 and cur_cost > best_cost:
            cur = [r[:] for r in best_routes]
            for _ in range(3):               # 施加 3 次随机可行移动作为扰动
                cand, _k = sample_neighbor(cur, rng)
                if cand is not None:
                    cur, cur_cost = cand, eval_routes(cand)[2]

        history.append(best_cost)

    v, d = eval_routes(best_routes)[:2]
    return {"cost": best_cost, "vehicles": v, "distance": d,
            "routes": best_routes, "history": history,
            "time": time.perf_counter() - t0}


# ==============================================================================
# 7. 实验框架：独立重复运行 + 指标统计
# ==============================================================================
def convergence_iter(history, final_best):
    """
    收敛迭代次数：best-so-far 首次达到「最终最优值」的迭代数。

    说明：题目原口径「首次达到最优值 90%」在最小化问题中无法直接使用
    （cost <= 0.9 * 最优 不可能发生，最优已是下界）。本实现采用的口径为：
    history 中首次出现与该次运行最终最优值相等的迭代号（0 = 初始解即已
    达到）。history[i] 记录第 i 次迭代结束时的 best-so-far，单调不增。
    """
    for i, h in enumerate(history):
        if h <= final_best + 1e-9:
            return i
    return len(history) - 1


def run_experiment(alg_name, alg_func, runs=RUNS, max_iter=MAX_ITER):
    """独立运行 runs 次，统计 Best / Avg / Std / 时间 / 收敛迭代。"""
    results, times, convs = [], [], []
    best_overall = None

    for r in range(runs):
        seed = BASE_SEED + r * 1000          # 每次运行不同但可复现的种子
        res = alg_func(max_iter=max_iter, seed=seed)
        results.append(res["cost"])
        times.append(res["time"])
        convs.append(convergence_iter(res["history"], res["cost"]))
        if best_overall is None or res["cost"] < best_overall["cost"]:
            best_overall = res
        v, d = res["vehicles"], res["distance"]
        print(f"  [{alg_name}] 第 {r + 1:2d} 次: cost={res['cost']:9.1f} "
              f"(车辆 {v}, 距离 {d:.1f})  用时 {res['time']:6.2f}s  "
              f"收敛迭代 {convs[-1]}")

    stats = {
        "best": min(results),
        "avg": mean(results),
        "std": stdev(results) if runs > 1 else 0.0,
        "time": mean(times),
        "conv": mean(convs),
        "best_res": best_overall,
    }
    return stats


def print_routes(res):
    """打印某次运行找到的最优路线明细。"""
    for k, r in enumerate(res["routes"], 1):
        path = " -> ".join(str(c) for c in r)
        print(f"    路线{k:2d}: 0 -> {path} -> 0")


# ==============================================================================
# 7.5 可视化：路线图与收敛曲线
# ==============================================================================
def plot_routes(res, title, fname):
    """
    路线可视化：
      - 车场（客户 0）用黑色方块表示
      - 不同车辆（子路线）用不同颜色的连线表示行驶路径
      - 客户点用圆圈表示，圆圈面积与需求量成正比
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab20.colors          # 20 色循环，足够 25 辆车

    # ---- 连线：每条子路线 0 -> c1 -> ... -> ck -> 0 一种颜色 ----
    for k, r in enumerate(res["routes"]):
        path = [0] + list(r) + [0]
        ax.plot([COORDS[i][0] for i in path], [COORDS[i][1] for i in path],
                "-", color=colors[k % len(colors)], lw=1.3, zorder=1)

    # ---- 客户点：圆圈面积与需求量成正比 ----
    cx = [COORDS[i][0] for i in range(1, N + 1)]
    cy = [COORDS[i][1] for i in range(1, N + 1)]
    ax.scatter(cx, cy, s=[demand[i] * 6 for i in range(1, N + 1)],
               facecolors="white", edgecolors="dimgray", linewidths=0.8,
               zorder=3, label="客户（大小∝需求量）")

    # ---- 车场：黑色方块 ----
    ax.scatter(COORDS[0][0], COORDS[0][1], marker="s", s=180, c="black",
               zorder=4, label="车场")

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_convergence(histories, title, fname):
    """
    收敛曲线：横轴为迭代次数，纵轴为目标函数值（车辆数×10000 + 总距离）。
    histories: {算法显示名: history 列表}，多条曲线画在同一张图上。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        ax.plot(range(len(h)), h, lw=1.4, label=name)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("目标函数值（车辆数×10000 + 总距离）")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_convergence_detail(histories, title, fname):
    """
    收敛曲线（细分版）：每个算法一个子图，各自独立的 y 轴范围。
    若某算法起点明显偏高（如 TS 未做多候选初始解时），合并图上其高起点会
    拉大纵轴范围，把其他算法曲线压成难以分辨的横线；此图消除该影响，
    便于比较各算法内部的下降过程与收敛代数。
    """
    names = list(histories)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2),
                             sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        h = histories[name]
        ax.plot(range(len(h)), h, lw=1.4)
        ax.set_title(f"{name}（终值 {h[-1]:.0f}）", fontsize=10)
        ax.set_xlabel("迭代次数")
        ax.grid(alpha=0.3)
        ax.margins(y=0.08)
    axes[0].set_ylabel("目标函数值（车辆数×10000 + 总距离）")
    fig.suptitle(title + " —— 各算法独立纵轴", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    global INIT_MODE
    parser = argparse.ArgumentParser(description="Solomon C101 VRPTW: HS/GA/TS 对比")
    parser.add_argument("instance", nargs="?", default=r"F:\Download\C101.txt")
    parser.add_argument("--iter", type=int, default=MAX_ITER)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--alg", choices=["HS", "GA", "TS", "all"], default="all")
    parser.add_argument("--init", choices=["i1", "mix", "both"], default="both",
                        help="初始解策略：i1=全部 I1 解, mix=70%% I1 + 30%% 强扰动, "
                             "both=两种都跑并对比")
    args = parser.parse_args()

    # ---- 载入实例 ----
    inst = load_solomon(args.instance)
    build_instance(inst)
    print(f"实例 {inst['name']}: {inst['n']} 个客户, 容量 {inst['cap']}, "
          f"车辆上限 {inst['vehicles']}, 迭代 {args.iter}, 独立运行 {args.runs} 次")
    print(f"目标: cost = 车辆数*{VEHICLE_WEIGHT:.0f} + 总距离  "
          f"(C101 已知最优: 10 车 / 827.3)\n")

    algs = {"HS": harmony_search, "GA": genetic_algorithm, "TS": tabu_search}
    if args.alg != "all":
        algs = {args.alg: algs[args.alg]}
    zh = {"HS": "和声搜索", "GA": "遗传算法", "TS": "禁忌搜索"}
    modes = {"i1": "全部 I1 解", "mix": "70% I1 + 30% 强扰动"}
    if args.init != "both":
        modes = {args.init: modes[args.init]}

    # ---- 逐模式实验 ----
    summary = {mode: {} for mode in modes}
    for mode, mode_zh in modes.items():
        INIT_MODE = mode
        print(f"\n########## 初始解策略: {mode_zh} ##########")
        for name, fn in algs.items():
            print(f"===== {name} ({zh[name]}) =====")
            summary[mode][name] = run_experiment(name, fn,
                                                 runs=args.runs, max_iter=args.iter)
            s = summary[mode][name]
            print(f"  >> {name}: Best={s['best']:.1f}  Avg={s['avg']:.1f}  "
                  f"Std={s['std']:.2f}  时间={s['time']:.2f}s  "
                  f"收敛迭代≈{s['conv']:.0f}\n")

    # ---- 各模式内部汇总表 ----
    for mode, mode_zh in modes.items():
        print(f"{'=' * 78}\n初始解策略: {mode_zh}")
        print(f"{'算法':<6}{'Best Cost':>12}{'Avg Cost':>12}{'Std':>10}"
              f"{'时间(s)':>10}{'收敛迭代':>10}   最优解(车/距离)")
        print("-" * 78)
        for name, s in summary[mode].items():
            br = s["best_res"]
            print(f"{zh[name]:<6}{s['best']:>12.1f}{s['avg']:>12.1f}{s['std']:>10.2f}"
                  f"{s['time']:>10.2f}{s['conv']:>10.0f}   "
                  f"{br['vehicles']}车 / {br['distance']:.1f}")
    print("=" * 78)

    # ---- 两种初始解策略的对比表 ----
    if len(modes) == 2:
        print("\n初始解策略对比（全部 I1 解 vs 70% I1 + 30% 强扰动）:")
        print(f"{'算法':<6}{'策略':<20}{'Best Cost':>12}{'Avg Cost':>12}{'Std':>10}"
              f"{'时间(s)':>10}{'收敛迭代':>10}")
        print("-" * 78)
        for name in algs:
            for mode, mode_zh in modes.items():
                s = summary[mode][name]
                print(f"{zh[name]:<6}{mode_zh:<20}{s['best']:>12.1f}{s['avg']:>12.1f}"
                      f"{s['std']:>10.2f}{s['time']:>10.2f}{s['conv']:>10.0f}")
            print("-" * 78)
        print("说明: mix 模式下 30% 初始解由 rand_rate=0.35 的强扰动 I1 生成，"
              "多样性更高但平均质量略低；对比两表可评估其对收敛性与稳定性的影响。")

    # ---- 可视化：每种策略下各算法的最优路线图 + 收敛曲线 ----
    if HAS_MPL:
        for mode, mode_zh in modes.items():
            conv_data = {}
            for name in algs:
                res = summary[mode][name]["best_res"]
                plot_routes(res, f"{inst['name']} {zh[name]} 最优路线（{mode_zh}）",
                            f"routes_{name}_{mode}.png")
                conv_data[zh[name]] = res["history"]
            plot_convergence(conv_data, f"{inst['name']} 收敛曲线（{mode_zh}）",
                             f"convergence_{mode}.png")
            plot_convergence_detail(conv_data, f"{inst['name']} 收敛曲线（{mode_zh}）",
                                    f"convergence_detail_{mode}.png")
        print("\n已生成图片: routes_算法_模式.png（路线图）、"
              "convergence_模式.png（合并收敛曲线）、"
              "convergence_detail_模式.png（各算法独立纵轴），见当前目录")
    else:
        print("\n未安装 matplotlib，已跳过绘图（pip install matplotlib 后重跑即可）")

    # ---- 展示总体最优解的路线 ----
    top = min((s["best_res"] for mode in modes for s in summary[mode].values()),
              key=lambda r: r["cost"])
    print(f"\n总体最优解 (cost={top['cost']:.1f}, {top['vehicles']} 辆车, "
          f"距离 {top['distance']:.1f}):")
    print_routes(top)


if __name__ == "__main__":
    main()
