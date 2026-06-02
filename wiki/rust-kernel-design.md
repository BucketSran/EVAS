# EVAS Rust Kernel Wiki

## 为什么要做这件事

EVAS 现在是 pure-Python behavioral Verilog-A evaluator。根据 r14 的同片速度实验，EVAS 在总量上比 Spectre AX 快，但慢 row 呈现出两类不同问题：

- `vbr1_l1_gain_estimator`、`vbr1_l2_gain_extraction_convergence_measurement_flow` 这类 measurement-heavy row：EVAS 的 accepted steps 比 Spectre AX 少，但每一步更贵。
- `vbr1_l1_pfd_small_phase_error_response` 这类 PFD/PLL row：EVAS 的步数远多于 AX，所以瓶颈主要是事件调度和步长控制，而不只是 Python 开销。

因此 Rust 方向不应该是把 Python 代码逐行翻译成 Rust。更合理的目标是：保留 Python 前端，把 transient hot loop、节点电压、模型状态、事件队列这些核心仿真内核移到 Rust。

## 当前工作假设

继续留在 Python 里的部分：

- Verilog-A parsing 和兼容性检查。
- SCS/netlist materialization。
- benchmark harness 和 checker。
- 报告生成、调试日志、实验管理。

迁移到 Rust 的部分：

- transient 主循环。
- node voltage storage。
- model state storage。
- timer/cross/breakpoint event queue。
- checker 必需信号的 sparse recording。

## 为什么这样会快

可以用一个简单公式理解：

```text
EVAS 内核耗时 ≈ 仿真步数 × 每一步的执行成本 + 记录/输出/检查开销
```

Rust 化主要针对前两项：

1. 降低每一步的执行成本。
2. 减少无意义的仿真步数。

### 每一步成本为什么会下降

当前 Python 热路径更接近：

```python
v_in = node_voltages["vin"]
v_clk = node_voltages["clk"]
node_voltages["vout"] = gain * v_in
```

这对人易读，但每步都包含字符串 hash、dict 查找、Python float object 运算、动态属性访问、解释器 dispatch 和函数调用。

Rust/indexed backend 的目标形态是：

```rust
curr[out_id] = gain * curr[in_id];
```

底层变成连续内存中的 `f64` 数组和整数 index。CPU 可以更好地利用 cache、寄存器、分支预测和编译器优化。这里没有魔法，关键是把“按名字查对象”变成“按编号读写连续数组”。

### 步数为什么会下降

PFD/PLL 这类 event-heavy row 很多时间点没有真正事件。fixed-step 会不断检查：

```text
t=1: 没事件
t=2: 没事件
...
t=60062: 有事件
```

event queue 的目标是直接跳到下一个可能有事件的时间：

```text
next_event_time = heap.pop()
time = next_event_time
```

所以 PFD/PLL 类 row 的优化重点不是只换 Rust，而是先减少无意义步数，再把剩下的 hot loop native 化。

## Python Indexed 化的定位

Python indexed 化不是终点，而是 Rust backend 的 IR/data-layout 准备层。

```text
Python indexed 化 = 把 Verilog-A/netlist 中的名字整理成稳定编号
Rust 化 = 用这些编号在 native hot loop 中高速执行
```

原因是：单纯在 Python 里把 `dict[str, float]` 换成 `list[float]` 仍然要经过 Python 解释器循环，所以收益有限；但这个编号化过程是 Rust backend 必须依赖的输入格式。

## 不允许功能删减

Rust/indexed backend 只能作为现有 EVAS 支持面的加速实现，不能收缩 EVAS 当前可仿真的行为。迁移原则：

- Python parser/compiler/netlist front-end 保持现有支持范围。
- 新 backend 默认先作为 opt-in 路径，不替换默认 Python backend。
- 每个迁移功能都要有 Python backend vs indexed/Rust backend parity test。
- 不支持的语义必须回退到 Python backend，不能 silently 改行为。
- 任何 EVAS 原本能仿真的例子和测试都必须继续可跑。

现有 EVAS 支持面需要至少覆盖：

- waveform primitives: `dc`、`pulse`、`pwl`、`ramp`、`sine`。
- events: `cross`、`above`、`timer`、`initial_step`、`final_step`。
- transition/slew、event-time interpolation、dynamic refine。
- arrays、integer/real/string parameters、case/for/while/if 控制流。
- file I/O system tasks、random streams、temperature/`$vt`。
- hierarchy/instance `node_map`、`@parent:`、bus/escaped node name。
- differential contribution、`idtmod`、`$bound_step`。
- Spectre netlist parsing、save expansion、source conversion、CSV schema。

当前可用的基础验证命令：

```bash
python3 -m pytest tests/test_indexed_backend.py -q
python3 -m pytest tests/test_backend_migration_capability_manifest.py -q
python3 -m pytest tests/test_engine.py tests/test_netlist.py -q
python3 -m pytest tests/test_examples.py -q
python3 -m pytest tests -q
```

迁移到 Rust backend 前，应该额外建立 original EVAS capability smoke matrix：

- 覆盖来源是 `evas/examples/backend_migration_capability_manifest.json`。
- manifest 必须列出当前 bundled example 中全部 `tb_*.scs`，现在是 5 个 group、16 个 testbench。
- 保留 `tests/test_examples.py` 的五个 example group：
  `adc_dac_ideal_4b`、`clk_div`、`comparator`、`digital_basics`、`noise_gen`。
- 已有 validator 覆盖 11 个 testbench。
- 额外 5 个 testbench 先以 schema-only 进入迁移覆盖：
  `tb_adc_dac_ideal_4b_ramp.scs`、`tb_adc_dac_ideal_4b_sine1000.scs`、
  `tb_clk_div_div2.scs`、`tb_clk_div_div8.scs`、`tb_inverter_chain.scs`。
- 每个迁移覆盖项后续都要比较 Python backend 与 indexed/Rust backend 的 `tran.csv` schema、关键 waveform、validator 结果和错误/警告行为。

第一条安全补丁原则：先增加 sidecar metadata 和独立 helper，不改变当前 string-keyed public contract。也就是说，`SimResult.signals`、CSV header、checker 输入仍然是字符串信号名；node id 只作为内部加速元数据存在。

## 速度实验覆盖口径

这里要分清两套覆盖：

- `evas/examples/backend_migration_capability_manifest.json` 是 backend 迁移门槛。它回答的是：新 backend 是否保留了 EVAS bundled examples 里原本能仿真的全部能力。
- `behavioral-veriloga-eval/benchmark-vabench-release-v1/reports/speed_debug_artifact.json` 这类 release speed artifact 才是论文速度 claim 的来源。它回答的是：在 vaBench release 的同片任务上，EVAS 和 Spectre/AX 的 wall time 对比是多少。

因此后续不能只用热点 row 或少量 examples 证明“EVAS 更快”。速度实验应该覆盖 release 中原本 EVAS 能通过的同片任务，并记录每个 row 的 EVAS/Spectre/AX 设置、wall time、checker 结果和 parity 状态。examples manifest 只保证 Rust/indexed backend 没有缩小 EVAS 的基础支持面，不替代 release-wide speed rerun。

## 大量 Rust 化时的修改面

| 层级 | 当前 Python 做法 | Rust/indexed 目标 | 加速来源 | 风险 |
|---|---|---|---|---|
| 节点表示 | `dict[str, float]` | `Vec<f64>` + node id | 去掉字符串/dict 查找 | 节点 alias/hierarchy 映射错误 |
| 节点读取 | `_get_voltage("vin")` | 编译期 `vin -> id` | 直接数组读取 | event context/interpolation 语义 |
| 输出写回 | `_set_output(name, value)` | `curr[out_id] = value` | 直接数组写入 | `node_map`、`@parent` 层级 |
| 状态变量 | Python object/dict 属性 | state id + `Vec<f64>` | 降低 model state 访问成本 | integer/real 类型语义 |
| 主循环 | Python `while time < tstop` | Rust transient loop | 降低解释器开销 | dynamic step/refine 语义 |
| `prev_nv` | 每步 `dict copy` | double buffer / array snapshot | 降低分配和复制成本 | cross interpolation 需要 prev/curr |
| timer/cross/above | 每步扫描 | min-heap event queue | 减少步数 | Spectre-style event ordering |
| `$bound_step` | 每步遍历模型 | active minimum | 减少 per-step scan | 动态 bound 更新 |
| waveform record | 记录多个 signal | sparse / required-signal trace | 降低记录/输出 | checker/debug 需要信号 |
| CSV/checker | Python | 先保留 Python | 降低迁移风险 | 不是第一优先级 |
| Rust 接口 | 无 | PyO3 或 FFI/subprocess | 接入现有 CLI | ABI/构建复杂度 |

## 建议改造顺序

| 阶段 | 内容 | 验收 |
|---|---|---|
| 1 | 新增 node/state indexed 数据结构，不接管主路径 | 单元测试通过，不改变现有行为 |
| 2 | Python indexed backend prototype | 与 Python dict backend 在小模型上 parity |
| 3 | Python event queue prototype for PFD/PLL | 步数下降，Spectre-equivalence 不退化 |
| 4 | Rust measurement replay | `gain_estimator` 类 row per-step cost 下降 |
| 5 | Rust event queue replay | PFD/PLL 类 row 步数和 runtime 下降 |
| 6 | PyO3 接入 EVAS CLI | opt-in backend 可跑 release smoke |
| 7 | 全量 EVAS/Spectre rerun | 原 EVAS 可仿真内容无功能删减 |

## 最小内核形态

```text
Python front-end
  -> parse Verilog-A / SCS
  -> lower supported subset into EVAS IR
  -> assign node/state/event integer IDs
  -> call Rust kernel

Rust kernel
  -> Vec<f64> voltage_prev
  -> Vec<f64> voltage_curr
  -> Vec<f64> model_state
  -> BinaryHeap or calendar queue for events
  -> transient loop
  -> sparse traces / final metrics
```

## 两类优化问题

### 1. 每步成本过高

代表 row：

- `vbr1_l1_gain_estimator`
- `vbr1_l2_gain_extraction_convergence_measurement_flow`

主要改动：

- 把 `dict[str, float]` 节点电压表换成 `Vec<f64>`。
- 把 `V(node)` 编译成整数 index。
- 把 Python model object/state dict 换成紧凑 state array。
- 用 double buffer 或 array snapshot 替代每步 `dict` copy。

验收标准：

- measurement-heavy row 的 EVAS per-step cost 现在大约是 `87-89 us/step`。
- 第一目标是接近或低于 AX 的 `45-54 us/step` 区间。

### 2. 步数/事件调度过保守

代表 row：

- `vbr1_l1_pfd_small_phase_error_response`
- `vbr1_l1_pfd_up_dn_logic`
- `vbr1_l2_cppll_tracking_and_frequency_step_reacquire_flow`
- `vbr1_l2_adpll_lock_ratio_hop_timer_flow`

主要改动：

- 用 min-heap event queue 替代每步 timer/breakpoint 扫描。
- `$bound_step` 维护 active minimum，而不是每步遍历所有模型。
- 把 continuous evaluation 和 event callback evaluation 拆开。
- 避免对数字/事件主导区间做无意义 oversampling。

验收标准：

- PFD-style row 的 step count 至少下降一个数量级。
- 同时保持 Spectre-equivalence gate 不退化。

## Prototype 阶梯

1. **Smoke microbenchmark**
   - 对比 Python dict loop 和 Rust indexed-array loop。
   - 目的：验证 native indexed data structure 值不值得继续。
   - 2026-06-02 初始结果：
     - Python measurement dict：median `16.590893 s`。
     - Python measurement indexed：median `9.852070 s`。
     - Rust measurement indexed：median `0.135588 s`。
     - 两边 `events`、`checksum`、`err_acc` 一致。
     - 解释：纯 Python 里 `dict -> list/index` 有约 `1.684x`，有价值但不是终点；Rust indexed 比 Python indexed 仍有约 `72.662x`，所以 native hot-loop 仍然必要。

2. **PFD event-queue toy**
   - Python fixed-step toy：`60062` processed steps，median `0.732351 s`。
   - Python event-queue toy：`4719` processed steps，median `0.060759 s`。
   - Rust fixed-step toy：`60062` processed steps，median `0.004671 s`。
   - Rust event-queue toy：`4719` processed steps，median `0.000474 s`。
   - 两边事件数都是 `31`。
   - 解释：Python 里 event queue 已经有约 `12.053x`，Rust 中 event queue 比 fixed-step 也有约 `9.854x`；PFD/PLL 类 row 的关键不是只换语言，而是要减少无意义固定步扫描。

3. **r14 targeted projection**
   - 当前 r14 EVAS subprocess：`100.368435 s`。
   - 当前 AX/EVAS aggregate speedup：`2.073x`。
   - 只优化 8 个已知热点 row 的估算：
     - 保守场景：EVAS `80.039626 s`，AX/EVAS `2.600x`。
     - 平衡场景：EVAS `73.615546 s`，AX/EVAS `2.826x`。
     - 激进场景：EVAS `65.779426 s`，AX/EVAS `3.163x`。
   - 解释：这是 planning estimate，不是 paper-facing claim。

4. **Real-row replay prototype**
   - 手写一个 Rust kernel replay `gain_estimator` 类 row。
   - 目的：估计 measurement-heavy row 上的真实收益。

5. **Event queue prototype**
   - 为 PFD-like row 实现 timer/cross/breakpoint heap。
   - 目的：验证是否能减少步数。

6. **IR boundary**
   - 定义 Python 输出、Rust 消费的小型 EVAS IR。
   - 目的：避免每个 Verilog-A model 都手写 Rust。

7. **PyO3 integration**
   - 把 Rust kernel 暴露成 Python module。
   - 目的：保留现有 EVAS CLI、tests 和 benchmark harness。

## Rust 学习路径

这条路径只学 EVAS kernel 需要的 Rust，不追求一开始把 Rust 全部学完。

1. `Vec<T>`、slice、indexing：理解 Rust 中连续数组怎么读写。
2. ownership/borrowing：理解为什么 Rust 不允许随意同时读写同一块数据。
3. `struct` 和 `impl`：用来表示 kernel state、model state、simulation config。
4. `enum`：用来表示 event kind、IR operation。
5. `BinaryHeap`：用来做 timer/breakpoint event queue。
6. `Result<T, E>`：用来处理 Python/Rust 边界错误。
7. PyO3：等 standalone Rust kernel 有价值后再学。

先不要碰：

- async Rust。
- 复杂 trait 抽象。
- unsafe code。
- full Verilog-A parser rewrite。

## 当前开放问题

- Rust 里应该解释执行 IR，还是让 Python 为每个 benchmark 生成 Rust/C code？
- checker 真正需要记录多少 waveform signal？
- 哪些 event semantics 必须先和 Spectre 完全对齐？
- PFD/PLL row 能否用 event-only scheduling，同时保留 waveform equivalence gate？
