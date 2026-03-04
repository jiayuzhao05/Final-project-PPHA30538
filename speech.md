用 LLM/文本情绪，从金融新闻中提取出一个 ‘银行体系压力’ 指标，并且看看这些压力在美国不同州之间是怎样分布的。

我会依次介绍探索的问题 用的哪些数据 如何建模 以及目前初步结果和streamlit demo

研究问题：
> 1.新闻情绪能不能预测州级银行基本面的恶化？ 这里的基本面主要用 FDIC 的 ROA 等盈利指标来度量。
> 2.相比传统的 FDIC 指标，比如历史 ROA、NIM、资产和存款增速，新闻情绪是不是更早、更强地反映未来的压力？
> 3.这种压力是不是有明显的地理集聚——是不是总是集中在某些州

方法 借鉴CTR/CVR 框架：
- 广告里，CTR 是‘会不会点’，CVR 是‘点了之后会不会转化’，两者相乘是 eCPM。
- 在这个项目里，bad_year 看成 CTR：某个州在某一年是不是进入‘坏年份’；
- severity 看成 CVR：一旦是坏年份，ROA 会恶化到什么程度；
- 两者相乘得到州级的 StressScore

Data
A:新闻情绪数据
- 来源:金融新闻/分析师标题的数据集 analyst_ratings_processed.csv，有标题、时间戳和对应的股票。
- 用 all-data.csv 里带标签的文本，训练一个轻量情绪分类器：TF‑IDF 特征加 Logistic 回归，大概区分正面、负面、中性。
- 只保留包含监管关键词的标题，比如 fdic, sec, fed, regulation, bank run, regional bank, stress test，把这些新闻按年份聚合，得到每个年份的平均情绪值、负面概率和新闻数量。

B:FDIC 的州–年财务数据
> - 来自 Summary_data_states.csv，里面有每个州每年的总资产、净利润、净息差、存款、银行数量等等。
> - 在这个基础上构造了 ROA、ΔROA 等一系列监管视角下的基本面指标。
> 清洗之后的结果合并到一个 state_year_panel 里，后面的静态图和 Streamlit 都是基于这个面板做的。”

Variables & Two-Stage Model
在州–年层面上，首先定义：
> - ROAs,t=NETINCs,t/ASSETs,t，也就是 FDIC 数据里的净利润除以资产；
> - ΔROAs,t=ROAs,t−ROAs,t−1，衡量 ROA 相比上一年的变化。
> 用每个州的历史分布来定义‘坏年份’：
> - 如果某一年的ΔROA 掉到这个州历史的 20% 分位数以下，记作 bad_year = 1，否则是 0；
> - 定义 severity = -min(ΔROA, 0)，也就是只看下降部分，下降越多，严重程度越高。

> 模型实现：
> - Stage 1 用 Logistic 回归预测p(bad_year)，特征包括上一期的 ROA、ΔROA、资产和存款的增速、NIM，还有刚才说的文本情绪指标 sent_mean、sent_neg_share、news_count；
> - Stage 2 用 Ridge 回归预测 severity，再和 Stage 1 的概率相乘，得到州–年的 StressScore。
> StressScore 是后面地图、排名和决策层的核心输入。”

Static Plot 1 – Sentiment vs ΔROA
> 每一个点代表一个州–年组合：
> - 横轴是监管相关新闻的负面概率均值 sent_neg_share；
> - 纵轴是下一期的ΔROA。

> 点云整体是有一点向右下方延伸的趋势：
> - 在负面情绪比较高的区域，ΔROA 更有可能变成负数，也就是下一年 ROA 明显恶化。
> 这张图不是严格的因果推断，但它给了比较直观的信号：
> 当围绕银行体系的监管和风险新闻变得更悲观时，后面一年州级银行盈利的下行风险确实在上升。
> 这算是对 Q1 的一个直觉上的支持。”

Static Plot 2 – Spatial Maps
把 2014–2020 所有年份的 bad_year 做了平均：
> \[
> \text{bad\year\_share}_s = \\frac{\\#\\{t: bad\_year{s,t}=1\\}}{\\#\\{t\\}}
> \]

颜色越深的州，过去这几年更经常处在‘坏年份’。
压力并不是平均分布的，而是明显集中在若干州和地区，这为 Q3 提供了证据：市场和基本面对负面新闻的反应在地理上是有集聚的


Model Evaluation
这页是一个初步的模型评估
用 2014–2019 年训练模型，2020 年做测试，对比两种特征集合：
> - 只用 FDIC 财务变量的 FDIC_only；
> - FDIC 财务变量加上新闻情绪的 FDIC_plus_sentiment。

> 从结果来看：
> - 在预测 bad_year 的 AUC 上，两者都在 0.62 左右，加了情绪以后变化不大；
> - 在预测 severity 的 R² 上，目前还比较弱，属于早期的 baseline。

反映了:
> 一，情绪确实包含一些信号，但用这么粗的‘按年聚合’和简单情绪模型，信息还是被稀释；
> 二，模型本身比较原始，现在用线性模型，只是运用CTR/CVR 框架迁移过来，并没有真正实现 Wide&Deep 或 PPO/Bandit。

这次 presentation 只是‘work in progress’的 checkpoint，final 报告里会尝试两件事情：
> - 把情绪做得更细，例如按季度甚至更短窗口聚合；
> - 换成Wide&Deep结构，看看非线性模型能不能把这些弱信号放大出来。”

Streamlit Demo
做了一个面向州–年面板的交互式 dashboard：
> - 左边可以选择年份、选择展示的指标，比如 StressScore、p_bad_year、DROA 或者 sent_neg_share；
> - 右边是一张可交互的地图和一张散点图，可以同时看到地理分布和‘情绪–ΔROA’的关系；
> - 下面有一个表格，列出当前年份各州的主要指标，方便按 StressScore 排序、对比。

> 后续如果时间允许，会在 app 里加入：
> - 不同监测策略的比较，比如按历史 ROA 排 vs 按 StressScore 排，
> - 简化的 Bandit 策略，模拟监管在有限注意力下怎样挑选最值得盯的州。


Connection to policy

- 第一，**指标本身是监管视角的**。我不是在预测股价，而是用 FDIC 的 ROA、NIM 等基本面数据来刻画“银行体系什么时候处在坏状态”。这更贴近监管部门真正关心的变量。
- 第二，**StressScore 可以直接用来做“有限注意力下的监管排序”**。如果监管者每年只能重点盯少数几个州，或者在一个州内只能抽查一部分机构，那么按照 `StressScore` 排序，就给出了一个“在哪些地方更可能物有所值”的清单。这和课堂上讲的资源配置 / targeting 问题是同一类决策。
- 第三，从地理分布看，**压力是高度集中的**。长期 bad_year share 比较高的州，往往和我们印象中的脆弱地区、地产泡沫严重或产业单一的地区是重合的。把这种“基于文本的早期信号 + FDIC 数据”的方法用在实际监管中，可以帮助监管部门更早识别这些高风险地区，提前做压力测试、资本缓冲或者现场检查。

所以整体上，这个项目既是一套机器学习/LLM 的预测问题，也是一套“在固定预算下如何更聪明地分配监管注意力”的 policy 问题。