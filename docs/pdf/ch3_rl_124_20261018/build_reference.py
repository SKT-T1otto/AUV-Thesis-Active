from pathlib import Path
import re, json, hashlib, sys, html
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TMP = ROOT / 'tmp/pdfs/ch3_rl_124_20261018'
PDF = OUT / '第三章强化学习方案124_原理实现与实验设计_20261018.pdf'
pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Math', 'C:/Windows/Fonts/cambria.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Code', 'C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNB', italic='CN', boldItalic='CNB')
NAVY = colors.HexColor('#14324a')
TEAL = colors.HexColor('#087e82')
GRAY = colors.HexColor('#526576')
INK = colors.HexColor('#243646')
PALE = colors.HexColor('#edf5f7')
LINE = colors.HexColor('#d7e3e9')
W, H = A4
LEFT, WIDTH = 48, W - 96
styles = {
    'body': ParagraphStyle('body', fontName='CN', fontSize=10.1, leading=17.0, textColor=INK, wordWrap='CJK', spaceAfter=8),
    'small': ParagraphStyle('small', fontName='CN', fontSize=8.7, leading=13.6, textColor=GRAY, wordWrap='CJK', spaceAfter=5),
    'h2': ParagraphStyle('h2', fontName='CNB', fontSize=12.3, leading=19, textColor=TEAL, wordWrap='CJK', spaceBefore=7, spaceAfter=7),
    'formula': ParagraphStyle('formula', fontName='Math', fontSize=12.5, leading=21.5, textColor=NAVY, wordWrap='LTR'),
    'cell': ParagraphStyle('cell', fontName='CN', fontSize=8.8, leading=14.2, textColor=INK, wordWrap='CJK'),
    'headcell': ParagraphStyle('headcell', fontName='CNB', fontSize=9.0, leading=14.2, textColor=colors.white, wordWrap='CJK'),
    'code': ParagraphStyle('code', fontName='Code', fontSize=8.0, leading=12.5, textColor=NAVY, wordWrap='LTR'),
}
PAGES = []

def page(title, section, *blocks):
    PAGES.append({'title': title, 'section': section, 'blocks': list(blocks)})
def p(text): return ('p', text)
def s(text): return ('s', text)
def h(text): return ('h', text)
def eq(text, no=None): return ('eq', text, no)
def box(title, text): return ('box', title, text)
def table(headers, rows, widths=None): return ('table', headers, rows, widths)
def code(text): return ('code', text)
def link(url, label=None): return f'<link href="{html.escape(url, quote=True)}" color="#087e82">{html.escape(label or url)}</link>'

# All pages below are a research design, not an implementation or experiment log.
page('第三章强化学习研究方案', '研究参考 / 2026.10.18',
    s('面向当前四智能体地图规划与残差控制系统 · 方案 1、2、4 · v1.0'),
    h('研究主线：面向混合控制事件的差分价值学习'),
    p('研究重点是：在地图与规划器已决定导航目标之后，集中 Critic 能否正确评价合法残差动作对接触、保持中断、碰撞，以及跨发现与交接过程的实际影响。先确认强化学习确有可改善的控制权限，再设计新的价值监督。'),
    table(['原编号 / 方案', '在统一方法中的角色', '当前建议'], [
        ['1 非局部动作价值一致性', '用两条真实闭环干预轨迹监督原 Q 的有限动作价值差', '优先完成机制筛选'],
        ['2 多智能体交互价值校准', '用四条干预轨迹校准原 Q 的非加性共同作用', '交互显著时再扩展'],
        ['4 停止时间 Bellman 备份', '研究事件边界、共同物理时刻与标签误差', '作为标签机制与对照'],
    ], [158, 222, 119]),
    box('建议路线', '先做方案 1 的最小可证伪版本，再比较固定时域与方案 4；方案 2 单独通过交互效应检验后加入。三者属于同一研究链条，不宜直接写成三个独立主要创新。'),
    h('文档状态'),
    p('已核查相关论文、当前代码控制链及若干解析例子；没有实施新算法，没有启动训练，没有建立性能通过结论。本文件中的参数、预算和目录均为后续研究设计。'),
    s('适用仓库：E:/gym/code/WORKSPACE/AUV-Thesis。保护既有奖励、观测、动作、规划器、网络维度、回放与历史证据。文中代码位置为 2026-10-18 阅读快照，后续以文件与函数名核对。'),
)

page('如何使用这份参考', '阅读导航与证据等级',
    table(['页码', '内容', '首先解决的问题'], [
        ['3-4', '当前系统、阶段边界与接口', '残差究竟能改变什么？'],
        ['5-6', '相关工作与共同数学基础', '哪些思想已经存在？标签对应哪个 Q？'],
        ['7-10', '方案 1：原理、损失、动作提议、落地', '有限动作后果是否被 Critic 学错？'],
        ['11-13', '方案 2：四支干预、阶段、训练', '共同作用是否足够强且可辨识？'],
        ['14-18', '方案 4：停止、误差、裁剪、离策略', '多步标签是否语义正确且有净收益？'],
        ['19-20', '工程结构、快照与正确性检查', '能否可靠地产生真实闭环分支？'],
        ['21-25', '预算、对照、指标、消融与推进门槛', '怎样排除数据和计算量的混淆？'],
        ['26-28', '算例、论文表述与参考文献', '什么结论现在可以写、以后才能写？'],
    ], [50, 220, 229]),
    h('三种证据等级'),
    p('<b>已有结论：</b>由公开论文或当前源代码直接支持，例如动作限幅、搜索者冻结、已有梯度监督与多步离策略校正。文献以 [编号] 引用。'),
    p('<b>条件推导：</b>在固定完整状态、固定未来策略、合法停止规则等假设下得到的代数关系；不自动扩展为当前部分可观测函数逼近系统的收敛定理。'),
    p('<b>研究假设：</b>差分校准能够改善 Actor 更新，事件时域能够改善标签，交互校准值得额外成本。这些都必须通过对照实验，而不是依靠命名成立。'),
    box('最重要的验收标准', '主结论必须落在独立测试场景的最终策略表现上；Critic 误差下降、轨迹图好看或训练回报增加都不能单独替代任务成功率与控制效率的证据。'),
)

page('当前任务：地图决定去哪里，残差影响怎么到达', '系统依据 / 代码核查',
    p('四个智能体由三个搜索者与一个执行者构成。地图与规划模块决定搜索导航目标、路径及执行者待机/接管目标；Actor 输出三维归一化加速度残差。方案应围绕实际路径、到达时间和事件后果展开，不能把区域分配能力归给低层 Actor。'),
    eq('u<sub>i</sub> = clip<sub>[-1,1]</sub>(μ<sub>i</sub>(o<sub>i</sub>)),   u<sub>i</sub> = 0  if finished<sub>i</sub>', '1'),
    eq('a<sub>i</sub><super>phys</super> = clip<sub>limits</sub>(κ<sub>i</sub> a<sub>i</sub><super>prior</super> + α<sub>i</sub> D<sub>i</sub> u<sub>i</sub>)', '2'),
    p('D_i 是轴向加速度尺度，κ_i 是先验强度，α_i 是残差比例。真实执行还受阻力、流场、速度上限、边界与碰撞协议影响。相同归一化动作差不保证不同物理加速度，更不保证改变事件。'),
    table(['接口 / 设定', '阅读确认及设计影响'], [
        ['Actor / Critic', '单体 28D 观测、3D 动作；联合观测 112D + 联合动作 12D = 124D 集中 Critic。保持现有网络合同。'],
        ['参考效率配置', '搜索残差比例 0.20、执行残差比例 0.15；γ=0.95；参考 dt=0.2。实际运行必须保存 resolved config，不能使用构造器默认比例代替。'],
        ['奖励与优化', '以已授权 team_mean_v1 适配后的团队奖励讨论，不添加交互奖励或事件奖励。B2/B3 直接基线强制残差正则为 0；不要照搬效率配置中的其他正则值。'],
        ['部署信息', 'Actor 不获得隐藏目标或完整地图。B3 BoundaryEncoder 的联合公共上下文应单独披露，不能只凭输入维度相同认定信息完全一致。'],
    ], [108, 391]),
    s('依据：core/env/uav_env.py::_apply_agent_dynamics、_get_obs；core/runtime/engine.py::CH3_SHARED_TRAINING_CONFIG、CH3_PROTOCOL_CONFIGS；chapter3_bser/experiments/baselines/common/model.py。'),
)

page('事件不是同一种边界：必须按代码定义', '系统依据 / 分阶段适用性',
    table(['事件', '当前实现的关键语义', '对研究的影响'], [
        ['首次发现', '三个搜索者与移动目标的相对扫掠距离达到传感阈值；记录离散 found_step。', '不能只比较步末距离；未发现不能从最终目标轨迹反推“已知”。'],
        ['发现后冻结', '发布发现时将前三个搜索者设为 finished；动力学入口将其动作清零。', '发现后搜索者当前动作应零效应；方案 2 不能研究两名搜索者共同保持。'],
        ['接触', '本步相对扫掠最小距离不超过捕获半径。', '一次接触不等于完整保持；接触次数只是辅助指标。'],
        ['完整保持', '起点和终点距离均不超过半径才递增保持计数，否则清零。', '重点研究接触到保持的转化与保持中断；按配置读取半径和所需步数。'],
        ['真正终止', '成功、碰撞或任务时限依选定协议执行。严格碰撞终止有其既定执行次序。', '发现/交接通常继续 bootstrap；不能把事件都标成 done。'],
    ], [76, 226, 197]),
    h('快照必须位于同一个运行时决策边界'),
    p('基线运行时的一个闭环步骤不仅是 env.step：之后还会获取 provider 快照、更新 controller、编译并安装 guidance、刷新观测。建议在这些操作完成、下一次 Actor 决策之前保存 S_t；事件检测同时比较本步环境事件与 guidance 的前后版本。'),
    box('接口缺口', '现有 HGR runtime.snapshot 只接受可靠交接后的存活决策状态。它不是搜索全阶段的通用快照。后续需在第三章独立实现全闭环快照，或先用“重置 + 原动作前缀回放”恢复状态；不能直接宣称接口已经可用。'),
    s('代码锚点：uav_env.py::_maybe_detect_swept（1919）、_publish_detection（1896）、_update_capture（2174）；hgr/runtime.py::snapshot（约 241）；baselines/common/runtime.py::step。行号为阅读时记录。'),
)

page('研究现状：可借鉴什么，不能声称什么', '交叉核验 / 最接近的工作',
    table(['工作', '已有核心机制', '本方案的区别与必要对照'], [
        ['MAGE [1]；CACTO-SL [2]；Sobolev RL [3]', '利用模型或轨迹优化提供价值导数监督，训练更适合策略更新的 Critic。', '真实闭环有限动作干预不要求探测、规划器可微；但“为 Actor 改善 Critic”及导数监督本身不是新意。'],
        ['分布价值梯度 [4]', '用随机世界模型同时处理回报与梯度分布。', '不能靠“考虑噪声”区分；需证明真实事件后果相对于平滑/导数目标的作用。'],
        ['Relative Value Learning [5]', '直接学习不同状态之间的相对价值及成对 Bellman 目标。', '这里比较同一状态的动作干预，沿用原标量 Q；价值差本身已存在。'],
        ['PAVE [6]', '约束 Critic 诱导的动作梯度场波动，并保留局部曲率。', '这里拟合真实分支回报差而非仅抑制几何波动；有限差分和不改 Actor 均不构成区别。'],
        ['DAWN [7]', '基策略数据作为价值锚点，配合 Critic 归一化改进残差价值学习。', '必须设置数据锚定强对照；其合成动作输入的尺度结论不能直接套到本项目的原始残差输入。'],
        ['DCG [8]；COIN [9]', '成对 payoff/交互结构；个体-全局 Critic 与反事实基线。', '方案 2 校准原 Q 的四支差中差，不增加图网络或个体目标；两两交互思想并不新。'],
        ['Retrace [10]；AHAC [11]；Q-chunking [12,13]', '多步离策略校正、接触相关自适应时域、动作段价值学习。', '方案 4 研究固定单步动作 Q 的标签时域与组内误差；可变时域本身不足作为主要创新。'],
    ], [106, 185, 208]),
    box('查重结论的边界', '截至本轮所读论文，尚未确认有工作完整实现“同一闭环世界、合法单步干预、事件相关差分校准、共同停止时刻误差分析”这一组合；这不是证明无人做过。投稿或定稿前仍须继续检索引用链，并用实验确认机制贡献。'),
)

page('共同基础：完整状态、公共输入与目标值', '数学依据 / 全部方案共用',
    p('令 S_t 为完整模拟器状态，包含物理状态、隐藏目标、地图、规划器、交接队列与随机数等；X_t=(o_1,…,o_4) 为 Critic 的公共联合输入。S_t 可以作为训练仿真的恢复依据，但不能输入部署 Actor。'),
    eq('Q<super>π</super>(S,u) = E[∑<sub>k=0</sub><super>T-1</super> γ<super>k</super> r<sub>t+k</sub> | S<sub>t</sub>=S, u<sub>t</sub>=u, π thereafter]', '3'),
    eq('q<super>π</super>(X,u) = E[Q<super>π</super>(S,u) | X<sub>t</sub>=X]', '4'),
    p('式 (3) 是完整状态下、当前动作固定、未来恢复同一策略 π 的动作价值。式 (4) 是在某一明确状态分布下的条件平均；X 未必满足马尔可夫性，分布随策略变化也会改变。因此本文不为 124D 网络宣称精确 Bellman 收缩或全局收敛。'),
    h('合法干预与目标策略冻结'),
    p('动作候选只使用公共 X、允许的动作边界和固定采样规则。每组分支恢复同一 S，仅当前步动作不同，下一步起由一个版本固定的目标策略 π̄ 及原规划器正常闭环运行。标签不反传梯度；记录策略哈希与标签生成时刻。'),
    eq('V<sub>tar</sub>(X) = min{Q<sub>tar,1</sub>(X, μ<sub>tar</sub>(X)), Q<sub>tar,2</sub>(X, μ<sub>tar</sub>(X))}', '5'),
    p('下标 tar 表示冻结目标网络，等价于文中带横线的记号。式 (5) 对每个现有智能体的双 Critic 分别计算；团队奖励相同不意味着要合成新网络。若选择 B3，目标策略的上下文构造也必须按其原接口执行。'),
    box('同世界配对的价值', '相同状态和合法随机数耦合能减少场景差异带来的混淆，但不保证方差必降：Var(Y+ - Y-) = Var(Y+) + Var(Y-) - 2Cov(Y+,Y-)。随机流耦合须保持每支正确边际分布，分支事件导致随机调用次序改变时尤其需要检查。'),
)

page('方案 1：用真实动作后果校准价值差', '原编号 1 / 原理与目标',
    h('要解决的具体问题'),
    p('普通一步 TD 主要约束 Q 的绝对值。有限样本和函数逼近下，两个邻近合法动作的误差可能方向相反，使 Critic 对“更能保持接触”或“更可能碰撞”的动作给出错误排序。新增监督应直接来自实际闭环后果，而非另加一个事件预测头。'),
    eq('u<super>±</super> = clip<sub>[-1,1]</sub>(u<super>0</super> ± εd),   u<super>0</super>=μ<sub>tar</sub>(X)', '6'),
    eq('Y<super>b</super> = ∑<sub>k=0</sub><super>h_b-1</super> γ<super>k</super> r<sub>t+k</sub><super>b</super> + γ<super>h_b</super>(1-d<sub>b</sub>) V<sub>tar</sub>(X<sub>t+h_b</sub><super>b</super>)', '7'),
    eq('ΔQ<sub>φ</sub> = Q<sub>φ</sub>(X,u<super>+</super>) - Q<sub>φ</sub>(X,u<super>-</super>),   ΔY=Y<super>+</super>-Y<super>-</super>', '8'),
    eq('L<sub>Q</sub> = L<sub>TD</sub> + λ<sub>Δ</sub> E[w (ΔQ<sub>φ</sub> - stopgrad(ΔY))<super>2</super>]', '9'),
    p('h_b 为预先定义的有限时域或停止时刻；d_b 只表示真正终止。初版可先用 h=1 且沿用原目标裁剪得到严格对齐的目标；多步式 (7) 与现有裁剪语义的关系见第 16-18 页。不能忽略该问题直接替换。'),
    h('为什么可能改善学习'),
    p('同一状态下的差分削弱了共同价值偏移，训练重点转向 Actor 真正需要区分的动作效应。跨事件的有限动作对还可暴露单点平滑梯度看不到的阈值后果。不过，这只是有依据的假设：跨阈值的割线并不是该点真实导数。'),
    s('建议先验证执行者接触、保持中断与碰撞，再验证搜索残差对发现时间的间接影响。若每次只改变一步就几乎不影响后果，该方向应停止或重新定义研究对象，不能暗中改成动作段控制。'),
    ('diagram', 'pair'),
)

page('方案 1：它增加的是误差方向约束', '原编号 1 / 损失解释与 Actor 联系',
    p('令同一批两个分支的预测残差为 e+=Q+ - Y+、e-=Q- - Y-。先用平方损失说明机制，实际代码原 TD 为 SmoothL1；二者不能在论证时混为一谈。'),
    eq('L = e<sub>+</sub><super>2</super> + e<sub>-</sub><super>2</super> + λ<sub>Δ</sub>(e<sub>+</sub>-e<sub>-</sub>)<super>2</super>', '10'),
    eq('L = (1+2λ<sub>Δ</sub>) z<sub>diff</sub><super>2</super> + z<sub>mean</sub><super>2</super>', '11'),
    p('其中 z_diff=(e+ - e-)/√2，z_mean=(e+ + e-)/√2。差异方向权重变大，共同偏移方向不变。若两个分支 TD 误差本来都为零，新约束也为零；它并未创造新的零误差固定点。'),
    eq('∇<sub>φ</sub>L<sub>Δ</sub> = 2λ<sub>Δ</sub>(ΔQ-ΔY)[∇<sub>φ</sub>Q(X,u<super>+</super>)-∇<sub>φ</sub>Q(X,u<super>-</super>)]', '12'),
    h('Actor 初版保持现有更新'),
    eq('∇<sub>θ_i</sub>J ≈ E[∇<sub>u_i</sub>Q<sub>φ_i,1</sub>(X,u) ∇<sub>θ_i</sub>μ<sub>i</sub>(o<sub>i</sub>)]', '13'),
    p('其他智能体动作继续按原实现停止梯度。方案的因果链是“动作效应标签更准确 → 原 Q 的相关动作几何更可靠 → 原 Actor 更新更有效”，每个箭头都需要证据。有限差分误差变小不能保证 Actor 处导数正确，也不能推出单调策略改进。'),
    box('必需的相同数据对照', '比较同样的 S、u+、u-、Y+、Y-：仅分支 TD vs 分支 TD + 差分损失，并控制优化步数及采样权重。否则收益可能全部来自额外仿真数据。对原 SmoothL1 的损失保持原实现；差分项先用平方损失，稳健 Huber 作为单独消融。'),
)

page('方案 1：动作怎样选，数据怎样采', '原编号 1 / 可执行干预设计',
    table(['设计项', '建议的最小实现', '必须记录'], [
        ['改变哪个动作', '初期仅改变执行者当前 3D 残差，其他当前动作固定；下一步全部恢复 π̄。', '干预角色、动作维、首步及后续策略版本。'],
        ['方向 d', '公共导航方向的平行/垂直方向与固定世界轴；速度为零或方向退化时使用确定轴。', '方向生成只依赖可用信息，不读取隐藏目标。'],
        ['幅度 ε', '机制筛选可试 0.05、0.15、0.30 三档归一化幅度；再保留两档，均须真实限幅。', '原始动作差、残差加速度差、合成物理加速度差。'],
        ['状态分层', '普通运动、近障、接触前、已有保持四层；阈值依据公共观测预注册。', '场景、阶段、样本概率、是否来自不同回合。'],
        ['搜索扩展', '先单独检验搜索者一步残差能否改变发现或跨发现接管状态。', '地图与导航版本固定，发现时间包含失败。'],
    ], [86, 238, 175]),
    h('避免动作饱和与样本选择制造假信号'),
    p('u+ 与 u- 限幅后可能相同；物理先验叠加限幅后又可能相同。日志应分别给出归一化动作不同率、物理动作不同率、轨迹不同率和事件不同率。不得为了产生事件而放大残差上限。发现后被冻结的搜索者可提供“真实零效应”负对照。'),
    p('主实验从自然训练占用分布抽样，记录不成功与同事件分支。边界压力集可额外评估，但不能仅保留隐藏真值显示“恰好跨阈值”的成功样本。若采用已知分层概率 q_s，目标占比为 p_s，可使用 w=p_s/q_s；截断权重会引入偏差，必须披露。'),
    box('独立性', '确定性环境中，同一 S、相同动作、相同策略重复四次会得到同一标签；这不是四个独立观测。随机置信区间应来自不同合法世界/回合，或任务本身真实存在的随机性，不能为制造重复而改变目标策略。'),
)

page('方案 1：具体落地与诊断输出', '原编号 1 / 实施步骤',
    h('最小算法流程'),
    code('1  Sample a valid decision state S and public input X.\n2  Freeze target-policy / target-critic version v.\n3  Propose u_plus, u_minus from X; save proposal probability.\n4  For each branch: restore S; intervene for ONE step.\n5  Continue the original closed loop under frozen policy v.\n6  Build Y_plus, Y_minus using one declared label protocol.\n7  Fit both existing critics with TD and paired contrast loss.\n8  Update the original actor on natural replay observations.\n9  Audit held-out contrasts and actual policy returns.'),
    p('Actor 更新初版仍从原训练回放取公共观测；分支缓存只用于 Critic，避免同时改变 Actor 的状态分布。分支的当前动作可以偏离目标策略，当前动作之后的后缀必须跟随已冻结策略；这是 Q(S,u) 的语义，不是全程 on-policy 要求。'),
    table(['日志', '用途'], [
        ['contrast_rmse / tie-aware sign_accuracy', '检查动作差分预测；在标签接近零或噪声带内不强行计正负。'],
        ['physical_action_distinct / event_change_rate', '检查残差是否真的有控制权限。'],
        ['bootstrap_mass / clip_rate / policy_version_age', '检查收益是否由不可靠尾值、裁剪或陈旧标签驱动。'],
        ['actor_step_return_delta', '小步 Actor 更新前后在配对场景的真实后缀收益变化；这是诊断而非额外策略搜索。'],
        ['branch_steps / restore_time / critic_steps', '完整披露训练额外计算。'],
    ], [205, 294]),
    box('方案 1 的立项判断', '仅当合法一步残差存在有意义后果、原 Critic 对这些后果估计失准，而且相同数据/预算下差分监督优于仅 TD，才值得进入较长训练。若只改善排序而不能改善最终控制，应按机制负结果记录。'),
)

page('方案 2：四次干预识别非加性共同作用', '原编号 2 / 原理与公式',
    p('两个智能体都改变动作时，团队收益变化未必等于两个单独改变之和。例如多个搜索者对同一发现机会产生重复作用，或者搜索者提前发现与执行者接管位置共同影响后续完成时间。是否存在这种现象必须实测。'),
    eq('u<super>ab</super> = (u<sub>i</sub><super>a</super>, u<sub>j</sub><super>b</super>, u<sub>-ij</sub><super>0</super>),   a,b ∈ {0,1}', '14'),
    eq('I<sub>ij</sub><super>Y</super> = Y<super>11</super> - Y<super>10</super> - Y<super>01</super> + Y<super>00</super>', '15'),
    eq('I<sub>ij</sub><super>Q</super> = Q(X,u<super>11</super>)-Q(X,u<super>10</super>)-Q(X,u<super>01</super>)+Q(X,u<super>00</super>)', '16'),
    eq('L<sub>Q</sub> = L<sub>TD</sub> + λ<sub>I</sub> E[w (I<sub>ij</sub><super>Q</super> - stopgrad(I<sub>ij</sub><super>Y</super>))<super>2</super>]', '17'),
    table(['解析例子', '(Y00,Y10,Y01,Y11)', 'I 的含义'], [
        ['加性收益', '(0,1,1,2)', '0：没有该二阶共同作用'],
        ['AND 共同成功', '(0,0,0,1)', '+1：单独改变无效、共同改变有效'],
        ['OR 重复成功', '(0,1,1,1)', '-1：两个改变争取到同一份收益'],
    ], [116, 148, 235]),
    p('I 是对原团队回报的差中差，不是新奖励，也不是每个智能体的 Shapley 值。正负都可能合理，不能强制“协同项为正”。这也不是把原 Q 分解成单体与两两 payoff 网络。'),
    s('与 DCG [8]、COIN [9] 的实质区别在训练信号来源与参数化：使用真实四支闭环后果监督现有集中标量 Q，不引入图结构、个体-全局多目标或交互预测头。该区别仍需证明有用，数学差中差本身并不新。'),
    ('diagram', 'four'),
)

page('方案 2：在当前任务中，什么交互值得测', '原编号 2 / 阶段与候选选择',
    table(['阶段 / 智能体对', '可以存在的实际机制', '研究边界'], [
        ['发现前：搜索者-搜索者', '两者对首次发现机会形成互补或重复；发现触发搜索冻结和任务模式切换。', '不把整个地图的区域分配变化归给残差；保持原闭环规划。'],
        ['发现前：搜索者-执行者', '搜索者改变发现时刻，执行者改变当时的位置/速度，影响发现后的拦截与保持。', '可能需要跨发现较长后缀；单步残差交互也可能非常弱。'],
        ['发现后：搜索者-执行者', '搜索者当前动作已被清零，其当前位置可能影响代价，但该动作本身不再起作用。', 'I 应为零，作为负对照；不能当作协作收益。'],
        ['发现后：执行者与自己', '只有执行者活动，可研究单体不同方向残差的耦合。', '这不属于多智能体方案 2，不混入主指标。'],
    ], [118, 207, 174]),
    h('可执行的干预与抽样'),
    p('四支共享同一 S、同一其他动作、同一未来策略版本。各智能体的 0 是参考动作，1 是一项合法有限偏移；两个动作偏移都只施加当前一步。当前四体共有六个不同配对，但仅在动作仍有效的阶段抽样。'),
    p('初期每个快照最多取两个预先指定的活动智能体对，搜索者配对与搜索者-执行者配对各一组。配对规则依据角色与公共几何；避免查看隐藏真值后挑选最大 I。只有精确相同的 S、完整 u00、未来策略及随机耦合时，才允许复用 00 分支。'),
    box('立项前提', '在独立快照中观察到可重复的 I，且其幅度超过标签误差与数值容差；再判断一阶动作差分是否已足够解释。不能仅因为任务包含多个智能体就默认二阶校准有效。'),
)

page('方案 2：损失为什么可能改善联合价值', '原编号 2 / 推导、成本与局限',
    p('将四个分支的误差按 (00,10,01,11) 排列为 e，令 c=(1,-1,-1,1)。二阶损失只强化“联合改变相对于单独改变”的误差方向。'),
    eq('L<sub>I</sub> = λ<sub>I</sub>(c<super>T</super>e)<super>2</super>,   c=(1,-1,-1,1)<super>T</super>', '18'),
    eq('L = e<super>T</super>(I<sub>4</sub> + λ<sub>I</sub>cc<super>T</super>)e', '19'),
    p('平方 TD 例子中该误差度量有一个特征值 1+4λ_I，其余三个为 1。它加强一个交互方向，保留其他方向；不是新增独立信息，四支都拟合精确时交互误差自然为零。必须有“四支数据仅 TD”的对照。'),
    h('噪声与可辨识性'),
    eq('Var(I<super>Y</super>) = c<super>T</super>Σc', '20'),
    p('Σ 是四支回报的联合协方差。共同随机数可能降低也可能增加某些组合的噪声，不能只按单支方差判断。确定性场景没有同状态蒙特卡洛方差，主要问题变为尾值误差、状态抽样与数值恢复误差。'),
    h('与方案 1 的组合应后置'),
    eq('L<sub>Q</sub> = L<sub>TD</sub> + λ<sub>Δ</sub>L<sub>contrast</sub> + λ<sub>I</sub>L<sub>interaction</sub>', '21'),
    p('先单独评估交互项，再评估组合，避免同时调两个权重造成解释困难。两项分别按组数取平均；不要把同一组的多种对比当作独立样本。若按幅度归一化，尺度只能从训练集冻结估计并在对照中一致使用。'),
    box('不能保证的事情', '只校准二阶项不构成完整联合价值分解；三阶作用可能存在。集中 Q 更准确，也不保证局部观测 Actor 能实现任意联合最优动作。本方案不引入联合最优动作教师或额外部署信息。'),
)

page('方案 4：在事件处截断，但保留未来价值', '原编号 4 / 停止时间原理',
    p('对完整状态、固定未来策略与合法有界停止时间 τ，折扣回报可按停止时刻分解。这是停止时间/多步 Bellman 思想的直接应用，不是新的 Bellman 定理。本文研究它如何服务于真实控制事件的差分标签。'),
    eq('τ = min{H<sub>max</sub>, inf{k ≥ 1 : E<sub>t+k</sub>=1}}', '22'),
    eq('Q<super>π</super>(S,u) = E[∑<sub>k=0</sub><super>τ-1</super> γ<super>k</super>r<sub>t+k</sub> + γ<super>τ</super>(1-d<sub>τ</sub>)V<super>π</super>(S<sub>t+τ</sub>)]', '23'),
    p('条件是状态/历史足够描述过程，停止规则只使用截至当前的信息，并且回报可积。这里 H_max 有限、任务有限，有界性易于控制。但用 V̄(X) 替代精确 Vπ(S) 后会产生近似误差，原 TD 目标裁剪也会改变语义。'),
    table(['建议事件', '检测规则'], [
        ['导航切换', '比较闭环步骤前后的有效导航目标或分配版本；排除浮点微小漂移。'],
        ['发现 / 交接', '记录原环境和交接状态迁移，不额外提前告知 Actor；通常非终止。'],
        ['首次接触 / 保持中断', '接触由扫掠语义判断；保持由原 full_hold 与计数清零判断。'],
        ['碰撞 / 成功 / 任务时限', '沿用原协议终止及其优先次序；提前终止后不再 env.step。'],
    ], [129, 370]),
    box('不要把每个保持计数递增都算作事件', '否则接近目标时每一步 τ=1，方法退化为一步备份。保持开始、保持中断和任务成功应分别记录；事件时域上限是训练标签截断，不等于环境 done。'),
    s('Retrace [10] 说明普通旧轨迹多步备份需要离策略处理；AHAC [11] 和 Q-chunking [12,13] 已覆盖其他自适应时域机制，不能用“接近事件缩短时域”单独主张主要创新。'),
)

page('方案 4：分支应各停各的，还是共同停止', '原编号 4 / 三种协议与实现',
    table(['协议', '明确规则', '优点与代价'], [
        ['P_ind：各支首事件', '各分支使用自己的 τ_b=min(首事件,Hmax)。', '便宜且易解释；折扣长度不同可能放大共同尾值偏移。'],
        ['P_fix：共同固定 H', '所有分支走到同一物理步 H，真正终止的分支以吸收态记账。', '最清楚的对照；可能多做无必要仿真。'],
        ['P_min：组内首事件', '组内任一分支首次发生事件时，全组在该物理步停止，上限 Hmax。', '共同折扣、成本小；可能频繁 τ=1，尚未显现其他分支后果。'],
        ['P_all：各支均遇首事件', '记录每支首次事件；全组都已遇事件时停止，或到 Hmax。', '更充分覆盖分支后果；成本更高，必须正确处理已终止分支。'],
    ], [120, 203, 176]),
    p('P_min 和 P_all 都可在分支组的联合信息流中在线判断：每一步推进所有未终止分支、更新已发生事件标记，再决定是否停止。随机耦合还须非预见：给定组内过去，各支条件转移仍符合原任务；仅保证边际分布正确还不够。不得事后选择让标签最好看或回报最高的截断点。'),
    h('组内终止分支的记账'),
    p('一支在第 t_b 步真正终止后，其后奖励为 0、继续价值为 0，不再调用 env.step。共同 h 目标写作先累积到终止、其后零填充。另一支仍正常运行；只保留共同物理时间标尺，不强迫两支相同事件或相同导航。'),
    h('建议比较顺序'),
    p('先比较 P_ind 与 P_fix；若观察到不同停止时刻的系统误差，再加入 P_min。P_all 留作后续消融。对事件协议，固定时域对照应同时提供相近平均时域版本与同总模拟步预算版本，不能只与 H=1 比。'),
    box('折扣按物理步计算', '原 γ=0.95 时，H=5、10、20 的尾值权重约为 0.774、0.599、0.358。不能每遇到一个事件只乘一次 γ，否则优化目标已经改变。'),
)

page('共同停止时刻能抵消哪一种标签误差', '原编号 4 / 条件推导',
    p('暂不考虑非线性目标裁剪。设两支均非终止，近似尾值为 V̂_b=V_b+e_b；相对于精确继续价值的差分标签误差为：'),
    eq('ε<sub>Δ</sub> = γ<super>τ+</super> e<sub>+</sub> - γ<super>τ-</super> e<sub>-</sub>', '24'),
    p('即使两支尾值误差恰好相同 e+=e-=c，不同停止时刻仍留下偏移：'),
    eq('ε<sub>Δ</sub> = c(γ<super>τ+</super> - γ<super>τ-</super>)', '25'),
    p('共同非终止时刻 h 则有：'),
    eq('ε<sub>Δ</sub><super>common</super> = γ<super>h</super>(e<sub>+</sub>-e<sub>-</sub>)', '26'),
    eq('ε<sub>I</sub><super>common</super> = γ<super>h</super>(e<sub>11</sub>-e<sub>10</sub>-e<sub>01</sub>+e<sub>00</sub>)', '27'),
    p('共同加性偏移可以抵消；四支系数之和为零，同理成立。这是支持“共同物理时刻”对照的一个具体理由。γ=.95、c=1、τ+=2、τ-=10 时，不同停止误差约 0.303763；共同 h 且两支误差相同则为 0。'),
    eq('|ε<sub>Δ</sub>| ≤ γ<super>τ+</super>|e<sub>+</sub>| + γ<super>τ-</super>|e<sub>-</sub>|', '28'),
    box('结论的严格边界', '上述抵消要求两支均非终止、尾值误差含有共同偏移，且未受非线性裁剪影响。终止掩码不同、误差不相关、事件后的状态差异很大时不保证抵消。共同 h 也可能增加成本或方差，不能据此宣称总是优于各支停止。'),
    s('可验证假设：在相同分支上，P_fix/P_min 是否降低相对于完整后缀参考值的 ΔY / IY 误差；同时比较方差、计算步数与最终策略。解析算例不是项目性能证据。'),
)

page('现有目标裁剪：不能略过的实现分叉', '原编号 4 / 目标算子一致性',
    p('当前 core/algorithms/maddpg.py::_compute_target_q 在一步目标末尾裁剪到 [-10,10]，并取两个目标 Critic 的较小值。令 C(z)=clip(z,-10,10)，当前目标为：'),
    eq('y<sub>1</sub> = C(r<sub>t</sub> + γ(1-d<sub>t+1</sub>)min(Q<sub>tar,1</sub>,Q<sub>tar,2</sub>))', '29'),
    p('直接把多步奖励相加后只裁剪一次，一般不等价于沿用一步裁剪语义。确定性反例：γ=.95，r0=-2，r1=3，末端 V=10。'),
    eq('C(-2 + 0.95 C(3 + 0.95 × 10)) = 7.5', '30'),
    eq('C(-2 + 0.95 × 3 + 0.95<super>2</super> × 10) = 9.875', '31'),
    table(['研究路径', '可以做什么', '不能声称什么'], [
        ['A0：严格目标对齐', '方案 1/2 先用同样的一步裁剪目标构造差分；原网络与 TD 定义保持。', '不能据此宣称已实现跨事件多步价值传播。'],
        ['C0：仅诊断标签', '离线比较原始多步标签、共同停止协议及完整后缀参考，同时报告裁剪率。', '诊断标签不能未经审查直接混入原训练目标。'],
        ['C1：独立目标协议', '后续明确审查独立章节方法的标签定义，固定 H 与事件 H 使用相同规则。', '不能说它与原一步算法具有相同固定点；不能静默修改冻结基线。'],
    ], [107, 212, 180]),
    p('若所有相关中间目标都在裁剪区间内，可在这些样本上使用无裁剪推导；仅末尾未裁剪并不够。随机环境还存在 E[C(Z)]≠C(E[Z])，逐样本嵌套裁剪也不能自动等同于重复作用期望 Bellman 算子。'),
    box('本次研究建议', '先完成 A0 与 C0。C1 只有在明确目标协议、对照与合同演化后再实施。当前文档没有更改目标、核心代码或既有奖励；保持这一前置门槛比“形式上套一个多步公式”更重要。'),
)

page('多步后缀、离策略与标签陈旧问题', '原编号 4 / 可落地的初版选择',
    h('为什么不能直接拼接旧回放'),
    p('现有回放是单步合同，没有保证连续片段、完整状态快照、行为策略密度及目标策略支持条件。确定性 Actor 加探索噪声不能直接提供 Retrace 所需的可用概率比。把旧行为后缀当作目标策略后缀，会改变标签语义。'),
    eq('ρ<sub>t</sub> = π(u<sub>t</sub>|history<sub>t</sub>) / β(u<sub>t</sub>|history<sub>t</sub>)', '32'),
    p('式 (32) 需要明确的随机策略及支持覆盖；确定性目标的 Dirac 分布不能靠写出密度比解决。初版优先从合法状态产生新鲜后缀：当前动作可干预，后续无探索噪声、按冻结目标策略闭环运行。旧前缀只用来恢复当前状态，不当作目标后缀。'),
    h('策略版本与缓存'),
    p('每个分支组固定目标 Actor/Critic 的版本；生成过程中不软更新该版本。缓存包含策略与目标哈希、生成步数、场景和标签协议。首版缓存只在同一短训练窗口使用；窗口结束丢弃或重建，避免把不同未来策略的 Y 混成同一个 Q 监督。'),
    h('原奖励、完整后果与真正终止'),
    p('发现、交接通常不是终止，标签必须保留后续接管与保持价值。终止掩码读取原任务结果，不由事件检测器推断。达到 H_max 只是截断；如果环境未终止，尾值仍保留。若为了审计走到原回合末尾，应计入所有真实仿真步。'),
    h('C1 的一个可复现候选定义'),
    eq('y<sub>τ</sub> = C(∑<sub>k=0</sub><super>τ-1</super>γ<super>k</super>r<sub>t+k</sub> + γ<super>τ</super>(1-d<sub>τ</sub>)V<sub>tar</sub>(X<sub>t+τ</sub>))', '33'),
    p('若后续选择此候选，普通采样状态与干预状态的 Critic 标签均按式 (33) 生成；固定 H 与事件 τ 仅改变停止协议。分支 TD 与差分项使用同一组 y，原 replay 可继续服务 Actor。原一步算法保留为独立对照。该定义是新的有限后缀代理目标，裁剪或尾值误差存在时不能宣称等价于原算子。'),
    box('当前可实施边界', '快照、事件记录与诊断可以先在第三章独立命名空间完成。C1 多步训练不能与原一步裁剪目标混用后仍声称同一真实 Q。若新标签定义与原目标不一致，应先隔离实验并重新界定优化的代理目标，再评估实际策略收益。'),
    s('与 Q-chunking [12,13] 的区别：本文当前动作仍是一个 3D 残差，后续由固定策略接管。若持续干预多步，应改称动作序列价值并改变接口；不能继续使用当前单步 Q 的说明。'),
)

page('工程实现：只在第三章增加独立研究层', '具体实现 / 模块与算法骨架',
    p('建议新建 chapter3_bser/experiments/event_value_calibration/（暂定名）。这是目录设计，不是本次已创建的算法。继承所选 B2/B3 的 Actor、双 Critic、优化器与执行接口；分支数据放独立有界缓存，原 replay 数据结构保持。'),
    table(['拟新增模块', '职责'], [
        ['snapshot_adapter.py', '打包/恢复全闭环状态，或用确定性前缀回放；验证 RNG 与闭包/缓存恢复。'],
        ['event_observer.py', '读取既有结果与状态迁移，产生事件标签；不改奖励、终止或控制逻辑。'],
        ['intervention_sampler.py', '公共信息动作提议、幅度、活动角色配对与采样概率。'],
        ['branch_runner.py / labels.py', '冻结目标版本、只干预当前步、执行完整闭环，构造已声明协议标签。'],
        ['branch_cache.py / losses.py', '保存成对/四元组；在原 Q 上计算差分损失，标签停止梯度。'],
        ['trainer.py / audit.py', '复用原训练更新节奏，记录额外成本、正确性与机制指标。'],
    ], [179, 320]),
    code('for ordinary_update in baseline_schedule:\n    B = original_replay.sample()\n    G = bounded_branch_cache.sample_same_version()\n    for each existing agent critic pair:\n        loss = original_TD(B) + matched_branch_TD(G)\n        loss += lambda_delta * contrast_loss(G)\n        loss += lambda_I * interaction_loss(G)  # optional\n        optimize_existing_critics(loss)\n    update_original_actors(B)\n    update_targets_on_original_schedule()'),
    s('上述骨架的 matched_branch_TD 与差分标签必须同义。A0 可直接使用原一步目标；C1 未解决前不把多步标签代入。λ_I 初版为 0。不调用 HGR 的边界回报预测、前缀纠偏或门控，不新增预测头。'),
)

page('数据结构与正确性：先证明分支可信', '具体实现 / 快照、缓存与检查',
    table(['字段组', '最小字段'], [
        ['来源', 'scene_id、episode_id、decision_step、snapshot_hash、resolved_config_hash、source_hash。'],
        ['策略与输入', 'target_actor_hash、target_critic_hash、X[4,28]、u[branch,4,3]；冻结策略版本。'],
        ['干预元数据', 'group_id、pair_ids、direction、epsilon、proposal_probability、stratum、sample_weight。'],
        ['轨迹与标签', '逐步原团队奖励、事件位、真正 done、各支 τ、共同 h、尾值、Y、裁剪前后值及激活率。'],
        ['物理与成本', '首步先验/残差/合成加速度、位置速度、原生仿真步数、恢复耗时、分支耗时。'],
    ], [103, 396]),
    h('上线训练前必须通过的检查'),
    p('<b>恢复一致性：</b>相同快照、相同动作序列得到相同位置、地图、知识、交接、导航与奖励；分支执行前后不污染主环境或主随机数流。先 CPU 确定性检查，GPU 差异另设合理容差。'),
    p('<b>零干预一致性：</b>u+=u- 时 ΔY=0；四支动作重复时 IY=0。交换 +/- 标签应反号；交换智能体 i,j 后交互值不变。'),
    p('<b>冻结零效应：</b>发现后只改变搜索者动作，真实轨迹与回报不变；Critic 若仍敏感，记录为模型误差。'),
    p('<b>事件与终止：</b>扫掠接触不等于 full_hold；发现不截断未来奖励；任务终止不 bootstrap；H_max 截断不伪造 done；共同停止的吸收态记账正确。'),
    p('<b>接口与回归：</b>28D/3D/124D、原 replay、原奖励与 provenance 全部不变；新章节独立测试。若后续涉及共享核心或目标合同演化，应按 AGENTS.md 运行相应完整回归，不能削弱历史校验。'),
    s('不提交模型、检查点或长输出；不覆盖 outputs 用户保留文件。代码读取哈希随文档附录保存，表示本轮阅读来源，不修改任何冻结 provenance manifest。'),
)

page('实验阶段 A：先量控制权限与标签质量', '实验设计 / 低成本机制筛选',
    table(['阶段', '明确设计', '输出 / 上限示例'], [
        ['A1 单体作用', '执行阶段四层各取 8 个合法快照，共 32 个；3 个方向、2 个幅度、2 支，Hmax=10。', '最多 32×3×2×2×10=3,840 分支步；加快照恢复/审计另计。'],
        ['A2 交互作用', '发现前选 24 个快照，每个 2 对活动智能体，四支；Hmax=10。', '最多 24×2×4×10=1,920 分支步；复用可降成本但需精确条件。'],
        ['A3 标签审计', '预先抽取约 10% 的干预组，完整走到原回合终止；同一轨迹派生多种停止标签。', '报告所有后缀仿真步及恢复步，不能只报前 H 步。'],
        ['A4 搜索权限', '独立加入搜索快照；比较真实发现时间、失败率与跨发现的接管状态。', '若效应微弱，不把搜索加速作为 RL 主张。'],
    ], [82, 242, 175]),
    p('这些数量是机制筛选起点，不是统计充分性的保证。快照应覆盖多个回合，不能让同一回合大量邻近帧充当独立样本。现有地图证据中的 Actor 调用为 0，不能替代这一步的强化学习机制证据。'),
    h('通过与停止规则'),
    p('先在开发集估计标签误差/数值容差，预先设定有实际意义的阈值：如对成功率的最小可接受差、接管/保持时间差或统一奖励尺度下的效应下限；不使用跨任务固定的 |I|>0.1。观察效应区间与尾值误差带是否可区分。'),
    box('预算诚信', '没有可用检查点时，先用原规划器/零残差策略采集可达状态，再安排经用户明确选择的有界 pilot；不要自动恢复历史 Phase 1C 或启动训练。确定性重复不能当成多个随机种子。这里没有执行上述实验。'),
)

page('实验阶段 B：用最少对照拆出真正贡献', '实验设计 / 算法比较矩阵',
    table(['组别', '数据 / 标签', '新增机制', '回答的问题'], [
        ['B0 原算法', '原 replay、原一步目标', '无', '当前起点是什么？'],
        ['B1 等更新', '原 replay、同量 Critic 更新', '无新分支', '收益是否仅来自更多优化？'],
        ['B2 分支 TD', '相同分支、A0 标签', '仅增加真实动作数据', '额外仿真本身有多大用？'],
        ['B3 方案 1', '与 B2 完全相同', 'TD + ΔQ 校准', '损失约束是否有独立收益？'],
        ['B4 固定 H', '独立审查的 C1 协议', '仅固定多步标签', '普通 n-step 是否足够？'],
        ['B5 事件 H', '同一 C1、同预算', '仅方案 4', '事件停止本身是否有效？'],
        ['B6 方案 1+4', '与 B4/B5 匹配', '差分 + 事件标签', '二者组合是否优于各自？'],
        ['B7 四支 TD', '与方案 2 同四支数据', '无交互校准', '四次干预的数据收益？'],
        ['B8 方案 2', '与 B7 完全相同', '交互校准；再测组合', '二阶项是否值得成本？'],
    ], [75, 148, 123, 153]),
    h('至少一种强相关算法对照'),
    p('优先加入 DAWN 风格的基策略数据锚定对照 [7]；若归一化需要改变冻结网络，则只能作为独立研究变体，不修改原网络。PAVE [6] 是另一项相关对照，按论文目标实现并说明连续动作多智能体适配。不要只用弱一步基线来证明“新损失有效”。'),
    p('不要求一次把全部组跑满。先完成 B0-B3；A2 通过才考虑 B7-B8，C1 目标审查通过才考虑 B4-B6。开发阶段最多保留两种时域和三个权重中的少量组合，预注册选择规则，避免隐形大规模调参。'),
    box('两套公平预算都要报', '其一是总原生环境步数（主轨迹 + 分支 + 恢复 + 审计），其二是训练墙钟时间。固定同一更新数可隔离优化量，但无法同时保证墙钟完全相同；采用性能-成本曲线并完整披露。'),
)

page('评价指标与统计：最终策略优先', '实验设计 / 指标定义',
    eq('SR = (1/N)∑<sub>n=1</sub><super>N</super> 1{mission success in scene n}', '34'),
    eq('T<sub>find,capped</sub><super>mean</super> = (1/N)∑<sub>n=1</sub><super>N</super> min(T<sub>find,n</sub>, T<sub>max,n</sub>)', '35'),
    p('未发现的场景令 T_find=∞，截断后记为 T_max，不能删除失败后只比较成功回合发现时间。对成功时间同样给出含失败的截断均值，并同时报告成功率；这不是任意缺失数据条件下的无偏生存估计。'),
    table(['层次', '推荐指标与解释'], [
        ['主要任务', '完整任务成功率、任务总回报；失败含碰撞/超时。预先指定主要指标与实际意义阈值。'],
        ['执行控制', '接触后最终完成比例、完整保持转化率、保持中断次数、碰撞率、截断成功时间。条件指标需同步报告分母，避免只筛容易接触的回合。'],
        ['搜索贡献', '截断发现时间、发现率、发现前碰撞、发现时执行者位置/速度；固定地图版本。'],
        ['价值机制', 'ΔQ RMSE、I Q RMSE、带零效应区间的排序准确率、相对于完整后缀的标签误差。尺度归一化用训练尺度并冻结。'],
        ['工程成本', '总仿真步、主步/分支步比、每更新分支数、恢复时间、训练墙钟、推理延迟。'],
    ], [84, 415]),
    h('建议的正式统计协议（后续人工决定执行）'),
    p('开发与测试场景清单分离，训练种子与评估场景种子分离。资源允许时采用至少 5 个独立训练种子、固定 100 个未参与调参的测试场景；每种子都评估同一批场景。数量可按 pilot 方差调整，禁止看测试结果调超参数。'),
    p('以训练种子为算法重复单位，报告跨种子均值/区间与每个种子原始结果；对方法差异保留场景配对，可做种子与场景的分层 bootstrap。不能把分支、时间步或同一场景多次重复当独立 N。多组探索比较应披露选择过程。'),
)

page('消融：让每个结论都有对应的证据', '实验设计 / 超参数与地图归因',
    table(['问题', '消融设计', '有效结论'], [
        ['是事件监督还是普通平滑？', '公共方向多尺度 ΔQ vs 相同计算的 Q 平滑/梯度正则；包含同事件样本。', '不能仅由跨阈值样本上的好结果推广全部状态。'],
        ['时域选得好吗？', 'P_ind / P_fix / P_min；Hmax∈{5,10,20}，先开发后锁定。', '用相同已生成轨迹比较标签，再做等预算策略比较。'],
        ['尾值可靠吗？', '完整后缀审计、不同停止时刻、尾值共同偏移诊断、裁剪激活率。', '共同偏移抵消是条件机制，不能当普适降噪。'],
        ['交互有必要吗？', '四支 TD vs 一阶差分 vs 交互差分；活动对与冻结搜索者负对照。', '只在真实 I 显著且优于一阶版本时保留。'],
        ['差分权重敏感吗？', 'λ∈{0.1,0.3,1.0} 仅作起始候选；报告实际损失/梯度比例。', '不从测试集选最佳权重，不无限扩展网格。'],
        ['收益来自地图吗？', '固定地图主比较；再做 地图旧/新 × 原RL/新RL 的 2×2 配对。', '分开报告地图主效应、RL 主效应及交互。'],
        ['搜索残差参与多少？', '评估时关闭搜索残差、保留执行残差；再分别关闭执行残差。', '这是策略权限诊断，不替代重新训练消融。'],
    ], [125, 213, 161]),
    p('初版建议 λ_I=0，λ_Δ 从 0.1 起做小范围开发；若差分损失尺度很大，先检查标签与采样，而非靠裁剪掩盖。若做尺度归一化/Huber，作为独立设计项记录，所有等数据对照保持一致。'),
    box('超参数不是结论', '上述数值是可执行的起始设计，没有项目性能验证。保持原奖励、γ、动作幅度合同与网络维度；不要用改变任务难度或奖励系数掩盖方法的失败。'),
)

page('推进顺序与否决条件', '实施路线 / 可交付里程碑',
    table(['里程碑', '可交付物', '继续条件'], [
        ['M0 设计冻结', '确定 B2 或 B3、地图版本、任务协议、目标裁剪、场景拆分与预算文件。', '合同与对照清楚，基线可复现；不自动恢复历史训练。'],
        ['M1 分支可信', '快照/回放接口、零干预/冻结动作检查、事件定义与日志。', '恢复一致性和无污染检查通过。'],
        ['M2 作用可辨识', 'A1/A2 效应分布、完整后缀审计、标签误差与成本。', '合法单步残差的效应超过误差带；交互单独判断。'],
        ['M3 A0 小型比较', 'B0-B3 的独立 pilot、等数据对照、价值与策略指标。', '差分项有数据之外的收益，没有安全或成功率退化。'],
        ['M4 时域 / 交互', 'C1 目标语义审查后 B4-B6；A2 通过后 B7-B8。', '新增复杂度确有净收益，否则保留最小版本。'],
        ['M5 正式评价', '经用户明确选择的训练预算、锁定超参、独立种子/测试、成本曲线。', '最终策略稳定改善，论文主张与证据对应。'],
    ], [94, 225, 180]),
    h('应当停止或降级的情况'),
    p('① 合法一步残差只造成数值级变化，没有实际事件/回报影响；② 相同分支 TD 已达到同样效果；③ 价值差更准但 Actor 更新或最终策略没有改善；④ 交互低于尾值/恢复误差；⑤ 共同停止时刻的额外成本超过标签收益；⑥ 多步标签与原目标语义冲突尚未解决。'),
    h('未来正确性测试范围'),
    p('局部新增模块先做任务范围检查；涉及共享核心、运行时或环境合同演化时按仓库要求运行完整相关回归与冻结等价/来源检查。不得通过修改历史哈希、删除来源记录或弱化测试来放行。正式训练不作为代码验证步骤自动启动。'),
    s('本文件完成的是研究设计与文档交付。没有 PRRAC dry-run、100 回合 pilot 或正式性能实验；不表示 Phase 1C、1000 回合运行或论文评价完成。'),
)

page('解析自检与可写成论文的研究问题', '附录 A / 不含项目性能结果',
    table(['自检', '数值 / 结果', '说明'], [
        ['中途事件与继续价值', '六步回报 2.450061875；正确事件分解一致，误差约 4.44×10^-16；错误截断为 0.9025。', '发现/交接不应误当终止。'],
        ['两支差分误差度量', 'λ=1 时特征值为 1、3。', '加强差异方向，不创造新零误差解。'],
        ['四支交互', '加性 0、AND +1、OR -1；λ=1 的误差度量为 1、1、1、5。', '交互不总为正，也不是新奖励。'],
        ['单步 vs 持续干预', '双积分器：一次干预最终位置 0.16；连续 20 步干预为 1.68，并在第 16 步跨阈值。', '二者不是同一个 Q(S,u) 标签。'],
        ['停止与裁剪', '不同 τ 的共同偏移误差约 0.303763；递归裁剪 7.5 vs 末端裁剪 9.875。', '共同时间与目标语义都需要检查。'],
    ], [121, 232, 146]),
    h('建议的论文研究问题'),
    p('<b>RQ1：</b>在真实残差权限内，现有 Critic 对有限动作造成的控制事件后果是否存在系统误判？<br/><b>RQ2：</b>相同分支数据与计算预算下，直接差分校准是否改善最终执行策略？<br/><b>RQ3：</b>共同事件时域是否改善差分标签质量，其净收益来自哪里？<br/><b>RQ4：</b>发现前的真实二阶动作作用是否足够强，交互校准是否超出一阶监督？'),
    box('可使用的暂定贡献表述', '“针对地图先验驱动的多智能体残差控制，研究合法单步干预下的有限动作效应估计，设计原集中 Q 的差分监督，并检验事件停止规则与二阶交互校准的作用。”只有实验完成后，才能补写在哪些指标、哪些预算与哪些场景上优于具体方法。'),
    s('不建议写：“首次提出差分价值学习”“首次考虑多智能体交互”“证明现有 124D 系统收敛”“显著提升搜索效率”。本轮资料无法支持这些无条件主张。'),
)

page('参考文献与阅读入口（一）', '附录 B / 原始论文链接',
    p('[1] D’Oro, P.; Jaśkowski, W. How to Learn a Useful Critic? Model-based Action-Gradient-Estimator Policy Optimization. NeurIPS, 2020. MAGE。<br/>' + link('https://proceedings.neurips.cc/paper_files/paper/2020/hash/03255088ed63354a54e0e5ed957e9008-Abstract.html', 'NeurIPS 原文入口')),
    p('[2] Alboni et al. CACTO-SL: Using Sobolev learning to improve continuous actor-critic with trajectory optimization. L4DC / PMLR 242, 2024.<br/>' + link('https://proceedings.mlr.press/v242/alboni24a.html')),
    p('[3] Schramm, F.; Perrin-Gilbert, N.; Carpentier, J. First-order Sobolev Reinforcement Learning. arXiv:2511.19165, 2025.<br/>' + link('https://arxiv.org/abs/2511.19165')),
    p('[4] Debes, B.; Tuytelaars, T. Distributional value gradients for stochastic environments. arXiv:2601.20071v3, 2026；所读 PDF 标记 ICLR 2026。<br/>' + link('https://arxiv.org/abs/2601.20071')),
    p('[5] Höftmann, M.; Robine, J.; Harmeling, S. Relative Value Learning. arXiv:2607.21120, 2026；论文及作者页面标记 ICLR 2026。<br/>' + link('https://arxiv.org/abs/2607.21120')),
    p('[6] Lee, J. W.; Kwak, K.; Kim, D.; Hwang, H. Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods. ICML / PMLR 306, 2026. PAVE。<br/>' + link('https://proceedings.mlr.press/v306/lee26aj.html')),
    p('[7] Ma, G. et al. What Makes Value Learning Efficient in Residual Reinforcement Learning? ICML / PMLR 306, 2026. DAWN。<br/>' + link('https://proceedings.mlr.press/v306/ma26s.html')),
    s('文献 [2]-[7] 的本地 PDF 与版本/哈希记录见：文献/第三章文献/RL创新候选交叉核验_20261018/manifest.json。本文主要依据已有下载正文及本轮官方/作者入口复核，不以搜索摘要代替方法差异。'),
)

page('参考文献、代码来源与下一步', '附录 B / 阅读与复现索引',
    p('[8] Böhmer, W.; Kurin, V.; Whiteson, S. Deep Coordination Graphs. ICML / PMLR 119, 2020.<br/>' + link('https://proceedings.mlr.press/v119/boehmer20a.html')),
    p('[9] Zhang, Y. et al. COIN: Collaborative Interaction-Aware Multi-Agent Reinforcement Learning for Self-Driving Systems. arXiv:2603.24931, 2026，作者预印本。<br/>' + link('https://arxiv.org/abs/2603.24931')),
    p('[10] Munos, R. et al. Safe and Efficient Off-Policy Reinforcement Learning. NIPS, 2016. Retrace。<br/>' + link('https://arxiv.org/abs/1606.02647')),
    p('[11] Georgiev, S. et al. Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation. ICML / PMLR 235, 2024.<br/>' + link('https://proceedings.mlr.press/v235/georgiev24a.html')),
    p('[12] Li, Q.; Zhou, Z.; Levine, S. Reinforcement Learning with Action Chunking. NeurIPS, 2025；所读为 arXiv 最新修订。<br/>' + link('https://arxiv.org/abs/2507.07969')),
    p('[13] Gireesh, N.; Ju, Y.; Wang, H. Adaptive Q-Chunking for Offline-to-Online Reinforcement Learning. arXiv:2605.05544, 2026，预印本。<br/>' + link('https://arxiv.org/abs/2605.05544')),
    h('本地核查材料'),
    s('同目录随附研究内容 JSON、可编辑 Markdown、PDF 构建脚本及制作核查记录。代码来源哈希只记录本轮读到的文件，不覆盖任何历史来源清单。解析算例源于相邻文献目录的 formula_checks.json 与 selected_124_formula_checks.json；本次新增自检仍只验证代数关系。'),
    box('下一步优先事项', '先固定一条 B2 或 B3 基线，完成完整闭环状态恢复与 A1 单步作用筛选；若控制效应和 Critic 误判确实存在，再做 A0 的等分支 TD / 差分监督比较。共同停止时域与多智能体交互都应服从这一证据顺序。'),
)

def para_draw(c, text, sty, x, y, width):
    obj = Paragraph(text, styles[sty])
    _, height = obj.wrap(width, 2000)
    obj.drawOn(c, x, y-height)
    return y-height-styles[sty].spaceAfter

def draw_table(c, b, y):
    _, headers, rows, widths = b
    widths = widths or [WIDTH/len(headers)]*len(headers)
    if abs(sum(widths)-WIDTH)>0.1:
        widths=[v*WIDTH/sum(widths) for v in widths]
    data = [[Paragraph(str(t), styles['headcell']) for t in headers]]
    data += [[Paragraph(str(t), styles['cell']) for t in row] for row in rows]
    t=Table(data, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('LINEBELOW',(0,0),(-1,0),0.6,NAVY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f7'), colors.white]),
        ('LINEBELOW',(0,1),(-1,-1),0.35,LINE),
    ]))
    _, th=t.wrap(WIDTH,2000)
    t.drawOn(c,LEFT,y-th)
    return y-th-12

def draw_diagram(c, kind, y):
    """Compact vector schematic; training-only supervision, not a new network."""
    height=104 if kind=='pair' else 108
    top=y-8
    c.setFont('CN',8.5);c.setFillColor(GRAY)
    caption='图 1  同一快照，两条单步干预后缀' if kind=='pair' else '图 2  四个联合动作组合，后续均恢复同一策略'
    c.drawString(LEFT,top,caption)
    def rect(x,yy,w,hh,text):
        c.setFillColor(PALE);c.setStrokeColor(LINE)
        c.roundRect(x,yy-hh,w,hh,4,fill=1,stroke=1)
        obj=Paragraph(text,styles['cell']);_,ph=obj.wrap(w-10,hh)
        obj.drawOn(c,x+5,yy-(hh-ph)/2-ph)
    def arrow(x1,y1,x2,y2):
        c.setStrokeColor(TEAL);c.setLineWidth(.9);c.line(x1,y1,x2,y2)
        c.line(x2-4,y2+2,x2,y2);c.line(x2-4,y2-2,x2,y2)
    if kind=='pair':
        rect(LEFT,top-30,81,38,'同一状态 S<br/>公共输入 X')
        rect(LEFT+112,top-19,94,29,'当前步 u+');rect(LEFT+112,top-61,94,29,'当前步 u-')
        rect(LEFT+240,top-19,155,29,'原闭环 + 冻结策略');rect(LEFT+240,top-61,155,29,'原闭环 + 冻结策略')
        arrow(LEFT+82,top-49,LEFT+110,top-34);arrow(LEFT+82,top-49,LEFT+110,top-76)
        for yy in [top-34,top-76]: arrow(LEFT+207,yy,LEFT+238,yy)
        c.setFont('Math',12);c.setFillColor(NAVY)
        c.drawString(LEFT+416,top-38,'Y+');c.drawString(LEFT+416,top-80,'Y-')
        arrow(LEFT+396,top-34,LEFT+411,top-34);arrow(LEFT+396,top-76,LEFT+411,top-76)
    else:
        rect(LEFT,top-32,80,36,'同一 S / X')
        labels=['00','10','01','11']
        c.setStrokeColor(TEAL);c.setLineWidth(.9)
        c.line(LEFT+81,top-50,LEFT+95,top-50)
        c.line(LEFT+95,top-50,LEFT+95,top-15)
        c.line(LEFT+95,top-15,LEFT+445,top-15)
        for k,lab in enumerate(labels):
            x=LEFT+106+k*99
            c.line(x+42,top-15,x+42,top-23)
            c.line(x+40,top-20,x+42,top-23);c.line(x+44,top-20,x+42,top-23)
            rect(x,top-25,84,27,f'当前步 u{lab}')
            c.setFont('Math',12);c.setFillColor(NAVY);c.drawString(x+27,top-77,'Y'+lab)
            c.setStrokeColor(TEAL);c.line(x+42,top-53,x+42,top-63)
        c.setFont('CN',8);c.setFillColor(GRAY)
        c.drawString(LEFT+110,top-97,'交互标签：Y11 - Y10 - Y01 + Y00；不加入环境奖励。')
    return y-height-16

def render():
    c=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle('第三章强化学习方案 1、2、4：原理、实现与实验设计')
    c.setAuthor('Codex · 研究设计参考')
    c.setSubject('Research design only; no training or performance claim')
    qa=[]
    for num, pg in enumerate(PAGES,1):
        c.setFillColor(TEAL); c.rect(0,H-10,W,10,fill=1,stroke=0)
        c.setFont('CN',8);c.setFillColor(GRAY)
        c.drawString(LEFT,H-36,'AUV THESIS  /  CHAPTER 3  /  RESEARCH DESIGN')
        c.drawRightString(W-LEFT,H-36,f'2026.10.18  ·  v1.0')
        c.setFillColor(TEAL);c.setFont('CNB',9)
        c.drawString(LEFT,H-66,pg['section'])
        y=H-82
        title=Paragraph(pg['title'],ParagraphStyle('title',fontName='CNB',fontSize=20 if num==1 else 17.5,leading=27,textColor=NAVY,wordWrap='CJK'))
        _,th=title.wrap(WIDTH,200)
        title.drawOn(c,LEFT,y-th);y-=th+18
        c.bookmarkPage(f'page{num}')
        c.addOutlineEntry(pg['title'],f'page{num}',level=0,closed=False)
        for b in pg['blocks']:
            if b[0]=='p': y=para_draw(c,b[1],'body',LEFT,y,WIDTH)
            elif b[0]=='s': y=para_draw(c,b[1],'small',LEFT,y,WIDTH)
            elif b[0]=='h': y-=6;y=para_draw(c,b[1],'h2',LEFT,y,WIDTH)
            elif b[0]=='table': y=draw_table(c,b,y)
            elif b[0]=='diagram': y=draw_diagram(c,b[1],y)
            elif b[0]=='eq':
                text=b[1]
                obj=Paragraph(text,styles['formula'])
                _,fh=obj.wrap(WIDTH-63,2000)
                c.setFillColor(PALE);c.roundRect(LEFT,y-fh-17,WIDTH,fh+11,4,fill=1,stroke=0)
                obj.drawOn(c,LEFT+12,y-fh-11)
                if b[2]:
                    c.setFillColor(GRAY);c.setFont('CN',8)
                    c.drawRightString(W-LEFT-10,y-fh/2-9,f'({b[2]})')
                y-=fh+28
            elif b[0]=='box':
                header=Paragraph(b[1],styles['h2']);body=Paragraph(b[2],styles['body'])
                _,hh=header.wrap(WIDTH-27,2000);_,bh=body.wrap(WIDTH-27,2000)
                height=hh+bh+25
                c.setFillColor(PALE);c.roundRect(LEFT,y-height,WIDTH,height,5,fill=1,stroke=0)
                c.setFillColor(TEAL);c.rect(LEFT,y-height,3,height,fill=1,stroke=0)
                header.drawOn(c,LEFT+13,y-10-hh);body.drawOn(c,LEFT+13,y-15-hh-bh)
                y-=height+12
            elif b[0]=='code':
                txt=html.escape(b[1]).replace(' ','&#160;').replace('\n','<br/>')
                obj=Paragraph(txt,styles['code']);_,ch=obj.wrap(WIDTH-24,2000)
                c.setFillColor(PALE);c.roundRect(LEFT,y-ch-18,WIDTH,ch+18,4,fill=1,stroke=0)
                obj.drawOn(c,LEFT+12,y-9-ch);y-=ch+30
            if y<53: raise RuntimeError(f'Page {num} overflow after {b[0]}: y={y:.1f}')
        c.setStrokeColor(LINE);c.setLineWidth(0.5);c.line(LEFT,43,W-LEFT,43)
        c.setFont('CN',8);c.setFillColor(GRAY)
        c.drawString(LEFT,28,'研究设计参考 · 所有实验待执行')
        c.drawRightString(W-LEFT,28,f'{num:02d} / {len(PAGES):02d}')
        qa.append({'page':num,'title':pg['title'],'body_bottom_y':round(y,2)})
        c.showPage()
    c.save()
    (OUT/'研究内容.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2),encoding='utf-8')
    md=[]
    for n,pg in enumerate(PAGES,1):
        md += [f"## {n}. {pg['title']}",f"*{pg['section']}*",'']
        for b in pg['blocks']:
            if b[0] in ('p','s','h','eq'):
                tx=b[1]
                tx=re.sub(r'<super>(.*?)</super>',r'^(\1)',tx)
                tx=re.sub(r'<sub>(.*?)</sub>',r'_(\1)',tx)
                tx=re.sub(r'<link href="(.*?)"[^>]*>(.*?)</link>',r'[\2](\1)',tx)
                tx=tx.replace('<br/>','\n')
                tx=re.sub(r'<[^>]+>','',tx)
                md += [('### ' if b[0]=='h' else '')+html.unescape(tx),'']
            elif b[0]=='box': md += ['**'+b[1]+'**',re.sub('<[^>]+>','',b[2]),'']
            elif b[0]=='code': md += ['```text',b[1],'```','']
            elif b[0]=='table':
                md += ['| '+' | '.join(b[1])+' |','| '+' | '.join(['---']*len(b[1]))+' |']
                md += ['| '+' | '.join(row)+' |' for row in b[2]]
                md += ['']
            elif b[0]=='diagram': md += ['（PDF 中含单步干预分支示意图。）','']
    (OUT/'第三章强化学习方案124_可编辑参考.md').write_text('\n'.join(md),encoding='utf-8')
    (TMP/'layout_check.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(PDF),'pages':len(PAGES),'bytes':PDF.stat().st_size,'minimum_body_bottom':min(q['body_bottom_y'] for q in qa)},ensure_ascii=False))

if __name__=='__main__': render()
