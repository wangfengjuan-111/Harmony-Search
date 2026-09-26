# -*- coding: utf-8 -*-
r"""
================================================================================
 和声搜索 (Harmony Search) VRP 交互演示 —— Streamlit 版
--------------------------------------------------------------------------------
 运行方式:
     streamlit run hs_vrp_demo.py
     (首次运行需安装: pip install streamlit)

 功能:
     左侧控制栏 : 最大迭代次数 / 初始解来源 / HMS / HMCR / PAR / bw 等滑块
                  [单步迭代] [连续运行/停止] [重置] 三个按钮
     右侧可视化 : ① 当前最优 VRP 路线图（实时更新）
                  ② 收敛曲线（实时更新，灰点是每次即兴的新和声，蓝线是 best-so-far）
                  ③ 当前最优 / HM 最差 / HM 平均 / 停滞代数 等统计

 规模说明（完整 100 客户）:
     教科书式 HS（无重启机制）在 100 客户上收敛很慢 —— 随机排列出发
     几十代才省下几辆车，之后长期走平；建议把最大迭代次数拉到 500~1000，
     并保持较大的 rand_rate / 较差的初始解来源，才能看到完整的下降过程。

 参数敏感性演示（把抽象结论变成实时可见的曲线变化）:
     拖动 HMCR 0.85 → 0.95 : 收敛更快，但曲线过早走平（早熟）；
     拖低到 0.80           : 随机性更强，收敛慢但更难陷入局部最优。
     拖动 PAR  0.30 → 0.50 : 局部微调更频繁，灰点波动更大，有机会碰出更好解。
     拖动 bw   大 → 小     : 前期大范围探索（交换到很远的位置），
                             后期精细开发（只能与紧邻基因交换）。

 实现:
     复用 c101_solver.py 的实例解析 / Split 解码器 / I1 初始解 / 局部搜索。
     即兴过程按教科书式 HS 三算子实现，逐次可单步观察：
       记忆考虑 (HMCR) : 逐位置以概率 HMCR 沿单个 donor 环形扫描继承基因
       音调调节 (PAR)  : 以概率 PAR 把刚放入的基因与最近 bw 个位置之一交换
       随机选择        : 以概率 1-HMCR 从未使用客户中随机取（全局探索）
================================================================================
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from matplotlib import font_manager
    
# 注册项目内的黑体字体
font_manager.fontManager.addfont('SimHei.ttf')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False # 修复负号方框
    
import matplotlib.pyplot as plt

import c101_solver as S

# ---- 载入 Solomon C101 基准实例（本文件同目录下的 C101.txt）----
_HERE = os.path.dirname(os.path.abspath(__file__))
INST_PATH = os.path.join(_HERE, "C101.txt")
if not os.path.exists(INST_PATH):
    INST_PATH = r"F:\Download\C101.txt"
BASE_INST = S.load_solomon(INST_PATH)


def build_full_instance():
    """载入完整 C101（100 客户 + 仓库）到 c101_solver 全局变量。"""
    inst = dict(BASE_INST)
    inst["n"] = len(BASE_INST["coords"]) - 1
    S.build_instance(inst)


# ==============================================================================
# HS 单步即兴（与 c101_solver.harmony_search 同构，但参数实时读自滑块；
# 停滞重启做成可开关 —— 关掉就能亲眼看到早熟/长期停滞现象）
# ==============================================================================
def hs_step():
    ss = st.session_state
    rng = ss.rng
    hmcr, par, bw = ss.hmcr, ss.par, ss.bw
    hm = ss.hm
    hms = len(hm)

    # --- 记忆库中随机选一个 donor 和声 ---
    donor = hm[rng.randrange(hms)][1]
    ptr = 0
    new_p = []
    remaining = set(range(1, S.N + 1))
    for _k in range(S.N):
        if rng.random() < hmcr:          # 记忆考虑：沿 donor 环形取下一个未用基因
            gene = donor[ptr]
            while gene not in remaining:
                ptr = (ptr + 1) % S.N
                gene = donor[ptr]
            ptr = (ptr + 1) % S.N
        else:                            # 随机选择：全局探索
            gene = rng.choice(tuple(remaining))
        new_p.append(gene)
        remaining.discard(gene)
        if rng.random() < par:           # 音调调节：与最近 bw 位内随机位置交换
            j = rng.randrange(max(0, len(new_p) - bw), len(new_p))
            new_p[-1], new_p[j] = new_p[j], new_p[-1]

    # --- 评价 + 可选的局部搜索抛光 ---
    _v, _d, rts = S.split(new_p)
    if ss.polish > 0:
        rts, c = S.improve_routes(rts, rng, budget=ss.polish)
    else:
        c = S.eval_routes(rts)[2]
    new_p = S.routes_to_perm(rts)

    # --- 更新记忆库与全局最优 ---
    if c < hm[-1][0]:
        hm[-1] = (c, new_p)
        hm.sort(key=lambda x: x[0])
    ss.last_new = c
    if c < ss.best_cost:
        ss.best_cost, ss.best_perm = c, new_p[:]
        ss.last_improve = ss.it + 1

    # --- 停滞重启（逃逸机制，复用 c101_solver.harmony_search 的方案）：
    #     连续 restart_stall 代无改进时，对历史最优做 3 次路线级随机扰动，
    #     再用大预算抛光（常规抛光无法从扰动中恢复并突破局部最优），
    #     结果注回记忆库 —— 否则库顶被锁死，搜索退化为反复报告现有解 ---
    if ss.restart_on and (ss.it + 1 - ss.last_improve) >= int(ss.restart_stall):
        ss.last_improve = ss.it + 1          # 重置停滞计数
        ss.restarts += 1
        rts2 = S.split(ss.best_perm)[2]
        for _ in range(3):
            cand, _k = S.sample_neighbor(rts2, rng)
            if cand is not None:
                rts2 = cand
        rts2, c2 = S.improve_routes(rts2, rng,
                                    budget=max(200, int(ss.polish) * 10))
        if c2 < hm[-1][0]:
            hm[-1] = (c2, S.routes_to_perm(rts2))
            hm.sort(key=lambda x: x[0])
        if c2 < ss.best_cost:
            ss.best_cost, ss.best_perm = c2, S.routes_to_perm(rts2)[:]

    ss.it += 1
    ss.history.append(ss.best_cost)
    ss.new_costs.append(c)


# ==============================================================================
# 会话状态初始化（重置时重建和声记忆库）
# ==============================================================================
def init_session():
    ss = st.session_state
    build_full_instance()
    rng = random.Random(int(ss.seed))
    hm = []
    for _ in range(int(ss.hms)):
        if ss.init_src == "随机排列（最差，看完整下降过程）":
            p = list(range(1, S.N + 1))
            rng.shuffle(p)
            rts = S.split(p)[2]
        elif ss.init_src == "强扰动 I1（较差，rand_rate=0.35）":
            rts = S.greedy_init(rng, rand_rate=0.35)
        else:  # 高质量 I1
            rts = S.greedy_init(rng, rand_rate=0.06)
        hm.append((S.eval_routes(rts)[2], S.routes_to_perm(rts)))
    hm.sort(key=lambda x: x[0])
    ss.hm = hm
    ss.best_cost, ss.best_perm = hm[0]
    ss.history = [ss.best_cost]
    ss.new_costs = []
    ss.it = 0
    ss.last_improve = 0
    ss.last_new = None
    ss.restarts = 0
    ss.running = False
    ss.rng = rng
    # 记录本次构建所用的环境参数，用于检测变动后自动重置
    ss.built_env = (ss.init_src, int(ss.hms), ss.rand_rate, int(ss.seed))


# ==============================================================================
# 绘图
# ==============================================================================
def draw_routes():
    """当前最优解的路线图：车场黑方块 / 各车辆不同颜色 / 客户圆圈∝需求量。"""
    v, d, routes = S.split(st.session_state.best_perm)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    colors = plt.cm.tab20.colors
    for k, r in enumerate(routes):
        path = [0] + list(r) + [0]
        ax.plot([S.COORDS[i][0] for i in path], [S.COORDS[i][1] for i in path],
                "-", color=colors[k % len(colors)], lw=1.2, zorder=1)
    cx = [S.COORDS[i][0] for i in range(1, S.N + 1)]
    cy = [S.COORDS[i][1] for i in range(1, S.N + 1)]
    ax.scatter(cx, cy, s=[S.demand[i] * 6 for i in range(1, S.N + 1)],
               facecolors="white", edgecolors="dimgray", linewidths=0.8, zorder=3)
    ax.scatter(S.COORDS[0][0], S.COORDS[0][1], marker="s", s=170, c="black", zorder=4)
    ax.set_title(f"当前最优：{v} 辆车 / 距离 {d:.1f} / cost {v * 10000 + d:.1f}",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def draw_convergence():
    """收敛曲线：横轴迭代次数，双纵轴 —— 左轴（蓝）跟随 best-so-far，
    右轴（灰）跟随每次即兴的新和声。初始解质量好时两条序列的数量级
    相差悬殊，共用一个轴会把 best-so-far 的下降压成直线。"""
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax2 = ax.twinx()
    ax.plot(range(len(ss.history)), ss.history, lw=1.6, color="tab:blue",
            label="全局最优 (best-so-far)【左轴】")
    if ss.new_costs:
        ax2.scatter(range(1, len(ss.history)), ss.new_costs, s=6, color="gray",
                    alpha=0.35, label="每次即兴的新和声【右轴】", zorder=2)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("全局最优目标函数值", color="tab:blue")
    ax2.set_ylabel("新和声目标函数值", color="dimgray")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="dimgray")
    ax.margins(y=0.08)
    ax2.margins(y=0.08)
    ax.set_title("收敛曲线（双纵轴：左=最优，右=新和声）", fontsize=11)
    # 合并两个轴的图例
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# 参数解读：根据当前滑块值实时给出搜索行为预判
# ==============================================================================
def interpret_params(hmcr, par, bw):
    msgs = []
    if hmcr >= 0.93:
        msgs.append(f"**HMCR={hmcr:.2f}（偏高）**：几乎只从记忆库继承基因 → "
                    f"利用性强、收敛快，但库内多样性迅速流失，曲线容易**过早走平（早熟）**。")
    elif hmcr >= 0.85:
        msgs.append(f"**HMCR={hmcr:.2f}（适中）**：记忆继承为主、随机探索为辅，通常最平衡。")
    else:
        msgs.append(f"**HMCR={hmcr:.2f}（偏低）**：随机基因比例大 → 探索性强、"
                    f"灰点波动剧烈，收敛慢但更**难陷入局部最优**。")
    if par >= 0.45:
        msgs.append(f"**PAR={par:.2f}（偏高）**：音调调节频繁 → 局部微调多、"
                    f"曲线波动大，有机会碰出更好解，但每次即兴扰动也更强。")
    elif par <= 0.15:
        msgs.append(f"**PAR={par:.2f}（偏低）**：几乎不做局部微调 → 曲线平滑，"
                    f"但精细开发能力弱。")
    else:
        msgs.append(f"**PAR={par:.2f}（适中）**：适度的局部微调频率。")
    if bw >= 7:
        msgs.append(f"**bw={bw}（偏大）**：交换步长大 → 前期**大范围全局探索**。")
    elif bw <= 3:
        msgs.append(f"**bw={bw}（偏小）**：只能与紧邻基因交换 → **精细的局部开发**。")
    else:
        msgs.append(f"**bw={bw}（适中）**：步长适中。")
    return msgs


# ==============================================================================
# 页面结构
# ==============================================================================
st.set_page_config(page_title="和声搜索 VRP 交互演示", layout="wide")
st.title("🎼 和声搜索 VRP 交互演示（Solomon C101）")
st.caption("目标 cost = 车辆数×10000 + 总距离。改编自 c101_solver.py，"
           "HS 即兴过程按教科书三算子逐步实现。使用完整 C101（100 个客户），"
           "从随机排列出发，让观众看到完整的收敛过程与参数影响。")

# ---------------- 左侧控制栏 ----------------
with st.sidebar:
    st.header("控制面板")

    # 三种预设：一键切到典型参数组合，便于课堂对比演示
    st.subheader("预设风格")
    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("探索型", key="preset_explore", use_container_width=True,
                  help="HMCR 低 + 大 bw：随机性强，收敛慢但不易早熟"):
        st.session_state.hmcr = 0.80
        st.session_state.par = 0.40
        st.session_state.bw = 8
        st.rerun()
    if pc2.button("平衡型", key="preset_balance", use_container_width=True,
                  help="文献推荐区间：HMCR≈0.90, PAR≈0.30, bw 适中"):
        st.session_state.hmcr = 0.90
        st.session_state.par = 0.30
        st.session_state.bw = 4
        st.rerun()
    if pc3.button("开发型", key="preset_exploit", use_container_width=True,
                  help="HMCR 高 + 小 bw：快速收敛，容易过早走平"):
        st.session_state.hmcr = 0.97
        st.session_state.par = 0.15
        st.session_state.bw = 2
        st.rerun()

    st.divider()
    st.subheader("HS 参数")

    st.slider("HMS 和声记忆库大小", 10, 60, 20, 5, key="hms",
              help="点【重置】后生效")
    st.slider("HMCR 和声记忆考虑率", 0.60, 1.00, 0.85, 0.01, key="hmcr",
              help="每个位置从记忆库继承基因的概率；越高越利用，越低越探索")
    st.slider("PAR 音调调节率", 0.00, 0.60, 0.30, 0.01, key="par",
              help="每个基因被局部交换微调的概率；越高曲线波动越大")
    st.slider("bw 音调调节步长（可交换的位置窗口）", 1, 10, 4, 1, key="bw",
              help="音调调节时能与刚放入基因交换的近邻位置数；大=全局探索，小=精细开发")

    st.divider()
    st.subheader("逃逸机制（防早熟）")
    st.checkbox("停滞重启", key="restart_on", value=True,
                help="连续 restart_stall 代无改进时，对历史最优做路线级扰动 + "
                     "大预算抛光后注回记忆库（复用 c101_solver 的重启方案）。"
                     "关闭后可观察「库顶锁死 → 曲线长期走平」的早熟现象")
    st.slider("重启阈值（连续无改进代数）", 20, 200, 60, 10, key="restart_stall",
              help="阈值越小重启越频繁：逃逸能力强，但每次重启都是一次局部扰动，"
                   "曲线跳变更剧烈")

    st.divider()
    st.subheader("环境参数（初始解相关变动即自动重置）")
    st.slider("最大迭代次数", 100, 2000, 500, 100, key="max_it",
              help="连续运行到达上限后自动停止；单步迭代也会被拦下。"
                   "100 客户规模建议 500~1000 次才看得出趋势")
    st.caption("客户规模固定为完整 C101 的 100 个客户。")
    st.radio("初始解来源", ["随机排列（最差，看完整下降过程）",
                            "强扰动 I1（较差，rand_rate=0.35）",
                            "高质量 I1（rand_rate=0.06）"],
             key="init_src",
             help="选「高质量 I1」可复现「库顶被初始解占据、曲线始终走平」的死锁现象")
    st.slider("初始解扰动强度 rand_rate", 0.0, 0.5, 0.35, 0.05, key="rand_rate",
              help="仅「强扰动 I1」来源时生效")
    st.slider("局部搜索抛光预算", 0, 200, 60, 10, key="polish",
              help="每次即兴后允许的邻域采样次数；0 = 纯 HS（参数影响最纯粹）")
    st.number_input("随机种子", 0, 10**9, 42, key="seed")

    st.divider()
    st.subheader("运行控制")
    b1, b2 = st.columns(2)
    if b1.button("单步迭代", key="step", use_container_width=True, type="primary"):
        if "hm" not in st.session_state:
            init_session()
        if st.session_state.it >= int(st.session_state.max_it):
            st.info(f"已达最大迭代次数 {int(st.session_state.max_it)}，"
                    f"调大上限或点【重置】重新开始。")
        else:
            hs_step()
    if b2.button("⏸ 停止" if st.session_state.get("running") else "▶ 连续运行",
                 key="run", use_container_width=True, type="primary"):
        if "hm" not in st.session_state:
            init_session()
        st.session_state.running = not st.session_state.get("running", False)
    if st.button("🔄 重置", key="reset", use_container_width=True):
        init_session()
        st.rerun()
    st.slider("每帧迭代数（连续运行速度）", 1, 100, 20, 1, key="batch")

    st.divider()
    st.caption(
        "**参数实验指引**\n"
        "- HMCR 0.85→0.95：收敛更快，但曲线可能过早平缓（早熟）\n"
        "- HMCR 0.85→0.80：随机性更强，收敛慢但可能跳出局部最优\n"
        "- PAR 0.30→0.50：局部微调更频繁，曲线波动更大\n"
        "- bw 由大调小：观察从大范围探索到精细开发的转变\n"
        "- 初始解来源选「高质量 I1」：曲线从第 0 代就走平（死锁现象）\n"
        "- 关掉「停滞重启」：观察库顶锁死后的长期停滞；打开后每次重启"
        "灰点带跳变、蓝线继续下棘轮")

# ---------------- 初始化 / 环境参数变动自动重置 ----------------
if "hm" not in st.session_state:
    with st.spinner("正在初始化和声记忆库..."):
        init_session()
elif st.session_state.built_env != (st.session_state.init_src,
                                   int(st.session_state.hms),
                                   st.session_state.rand_rate,
                                   int(st.session_state.seed)):
    with st.spinner("环境参数已变动，正在重建和声记忆库..."):
        init_session()

# ---------------- 连续运行：本帧先推进，再渲染，最后触发下一帧 ----------------
if st.session_state.running:
    for _ in range(st.session_state.batch):
        if st.session_state.it >= int(st.session_state.max_it):
            st.session_state.running = False
            break
        hs_step()

# ---------------- 右侧可视化区 ----------------
ss = st.session_state

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("迭代次数", ss.it)
m2.metric("当前最优", f"{ss.best_cost:.1f}")
m3.metric("HM 最差", f"{ss.hm[-1][0]:.1f}")
m4.metric("HM 平均", f"{sum(c for c, _ in ss.hm) / len(ss.hm):.1f}")
m5.metric("已停滞代数", ss.it - ss.last_improve)
m6.metric("已重启次数", ss.restarts)

if not ss.restart_on and ss.it - ss.last_improve > 30:
    st.warning("已超过 30 代无任何改进 —— 这就是「早熟」：库顶被初始解占据，"
               "新和声难以超越。试试调低 HMCR、调大 PAR/bw，"
               "或在左侧打开【停滞重启】逃逸机制。")

if ss.it >= int(ss.max_it):
    st.info(f"已达最大迭代次数 {int(ss.max_it)}（可在左侧调大上限后继续运行，"
            f"或点【重置】重新开始）。")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(draw_routes())
with col2:
    st.pyplot(draw_convergence())

st.subheader("当前参数对搜索行为的影响")
for msg in interpret_params(ss.hmcr, ss.par, ss.bw):
    st.markdown(f"- {msg}")

with st.expander("和声记忆库 (HM) 状态"):
    costs = [c for c, _ in ss.hm]
    st.write(f"库内代价：最优 {min(costs):.1f} / 中位 {sorted(costs)[len(costs)//2]:.1f}"
             f" / 最差 {max(costs):.1f}")
    st.write(f"代价去重后 {len(set(costs))} / {len(costs)} 个"
             f"（去重比例越低说明多样性流失越严重）")
    st.bar_chart({"HM 中各和声代价": dict(enumerate(sorted(costs), 1))})

# ---------------- 触发下一帧 ----------------
if ss.running:
    st.rerun()
