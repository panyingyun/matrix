# matrixmul — 1024×1024 矩阵乘法（Go，五阶段演进）

用 Go 从零实现 1024×1024 双精度矩阵乘法，按**五个阶段**逐步优化：从朴素三重循环
到手写 AVX2/FMA 汇编微内核，最终相对朴素实现加速约 **45×**（1024 规模，峰值 ~250 GF/s），
并且在 96~2048 的全部测试尺寸上都保持相对基线 ≥ 6×（阶段 4 在部分尺寸会塌陷到 1.5×）。
工程按阶段分目录组织，性能测试独立成目录。

- 随机矩阵固定种子（42）生成并落盘为 `input.matrix`，可复现、可复用；
  仓库内的黄金样例在 [e2e/testdata/case01](e2e/testdata/case01/)；
- 计算结果写入 `result.matrix`，由阶段 1 独立重算交叉校验；
- SIMD 用 **Go plan9 汇编手写 AVX2/FMA 微内核**（不使用 cgo，本环境 cgo 二进制无法运行）。

## 目录结构

```
matrix/
├── go.mod                       module matrixmul
├── README.md                    本文件（总览）
├── docs/                        架构图：architecture-code.svg / architecture-business.svg（+ PNG）
├── e2e/                         端到端测试（go test ./e2e）
│   ├── testdata/case01/         用例 1：input.matrix + result.matrix（3 组 1024×1024，种子 42）
│   └── testdata/case02–case11/  用例 2–11：不同维数/组数的 input.matrix + result.matrix
├── cmd/matrix/                  命令行工具：generate / multiply / verify
├── internal/
│   ├── matrixio/                二进制矩阵文件格式读写（共享）
│   └── kernel/                  AVX2/FMA 2×16 微内核（汇编，阶段 3/4/5 共享）
├── stage1/                      阶段 1：朴素三重循环（基线）
├── stage2/                      阶段 2：纯 Go 分块 + goroutine 行并行
├── stage3/                      阶段 3：AVX2/FMA 微内核 + 一维行并行
├── stage4/                      阶段 4：AVX2/FMA + B 打包 + 固定 2D 行列划分
├── stage5/                      阶段 5：AVX2/FMA + 自适应 2D 划分 + 按需打包（默认，最优）
└── benchmark/                   性能测试（独立程序 + 尺寸扫描 + Go 标准基准）
```

每个阶段目录内含实现、测试与 `README.md`（思路、瓶颈、实测数据）。

## 五个阶段与性能

| 阶段 | 目录 | 核心方法 | 单次耗时 | 吞吐 | 相对阶段 1 |
| --- | --- | --- | --- | --- | --- |
| 1 | [stage1](stage1/) | 朴素三重循环（单线程） | ~465 ms | ~4.6 GF/s | 1× |
| 2 | [stage2](stage2/) | 分块 64 + goroutine 行并行（16 行/任务） | ~84.5 ms | ~25 GF/s | **5.5×** |
| 3 | [stage3](stage3/) | AVX2/FMA 汇编微内核 + 一维行并行（64 行/任务） | ~16.4 ms | ~131 GF/s | **28.3×** |
| 4 | [stage4](stage4/) | 同上内核 + B 打包 + 固定 8×16 划分 | ~10.4 ms | ~207 GF/s | **44.8×** |
| 5 | [stage5](stage5/) | 同上内核 + **自适应 2D 划分** + **按需打包**（默认） | ~10.4 ms | ~206 GF/s | **44.6×** |

- 数据来自 `go test ./benchmark -bench . -benchtime=2s -count=3`（1024×1024，GOMAXPROCS=22，取 3 次均值）；
- 阶段 4 与阶段 5 在 1024 上持平（差异 <2%，在测量噪声内）；**阶段 5 的收益在其它尺寸**，见下节；
- 命令行程序（`go run ./benchmark -in e2e/testdata/case01/input.matrix -repeat 3`）3 组合计：
  阶段 1 1337.9 ms → 阶段 5 35.5 ms = **37.7×**（同进程连续测量，受热状态影响，数值系统性偏低）；
- 各阶段均为 `C += A×B` 累加语义，调用前需 `clear(C)`。

## 尺寸扫描：加速比随 n 变化很大

阶段 4 的列任务宽度是 `ceil(n/16)`，只有它是 16（内核宽度）的整数倍时列尾才全走 SIMD；
阶段 5 把任务固定为 32 行 × 64 列一带并按需打包，从而消掉了这个对齐依赖：

```powershell
go run ./benchmark -sizes all -repeat 3      # 预设 16…2048
go run ./benchmark -sizes 64,384,1024        # 自定义尺寸
```

| n | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 4 | 阶段 5 |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.0021 ms | 0.41× | 0.84× | 0.02× | 0.82× |
| 32 | 0.021 ms | 0.81× | 2.49× | 0.19× | 2.61× |
| 64 | 0.145 ms | 1.67× | 5.56× | 0.74× | 6.12× |
| 96 | 0.447 ms | 1.97× | 7.88× | 1.02× | 10.8× |
| 128 | 1.08 ms | 2.76× | 7.46× | 1.48× | 17.1× |
| 192 | 3.25 ms | 2.87× | 13.0× | 1.48× | 23.1× |
| 256 | 7.63 ms | 3.17× | 17.9× | 20.2× | 28.6× |
| 384 | 23.0 ms | 3.02× | 17.9× | 3.79× | 31.5× |
| 512 | 56.5 ms | 3.05× | 24.5× | 32.1× | 27.3× |
| 640 | 108 ms | 3.22× | 24.8× | 6.02× | 34.0× |
| 768 | 197 ms | 3.29× | 28.0× | 35.2× | 35.5× |
| 1024 | 465 ms | 3.32× | 24.4× | 36.1× | 38.7× |
| 2048 | 5488 ms | 8.27× | 21.1× | 75.5× | 70.5× |

（每格是相对同尺寸阶段 1 的加速比；批量重复计时并扣除 `clear` 开销，见 benchmark 的 `timeBatched`。）

- **并行度随 n 增长**：阶段 2/3 每个任务负责 16/64 行，n≲64 时只有 1~2 个任务，几乎拿不到并行收益；
- **阶段 4 的塌陷**：192 只有 1.5×、384 3.8×、640 6.0×，而同为大尺寸的 1024 有 36×；
- **阶段 5 消掉塌陷**：192 → 23.1×、384 → 31.5×、640 → 34.0×，且在 1024/2048 上与阶段 4 持平；
- n≤48 时任何阶段都拿不到加速（矩阵本来就装在 L1/L2 里），此时阶段 3/5 相当。


## 快速开始

```powershell
# 1) 生成随机矩阵（固定种子 42，可复现；写到当前目录的 input.matrix）
go run ./cmd/matrix generate

# 2) 用指定阶段计算：multiply [1|2|3|4|5]，默认 5（最优）
go run ./cmd/matrix multiply 5

# 3) 校验：阶段 1 独立重算校验 result.matrix，并交叉校验全部 5 个阶段
go run ./cmd/matrix verify

# 仓库内黄金样例（e2e 用例 1）可直接校验，无需重新生成
go run ./cmd/matrix verify -in e2e/testdata/case01/input.matrix -result e2e/testdata/case01/result.matrix

# 端到端测试
go test ./e2e
```

也可构建二进制：`go build -o bin/matrix.exe ./cmd/matrix`。

<details>
<summary>输出示例（verify）</summary>

```text
group 1/3: result.matrix          vs stage1: max abs error = 1.705e-13  [OK]
group 1/3: stage2-blocked         vs stage1: max abs error = 0.000e+00  [OK]
group 1/3: stage3-avx2-1d         vs stage1: max abs error = 1.705e-13  [OK]
group 1/3: stage4-avx2-packed-2d  vs stage1: max abs error = 1.705e-13  [OK]
group 1/3: stage5-adaptive        vs stage1: max abs error = 1.705e-13  [OK]
...
verify passed: 3 组 × 5 个阶段全部在容差 1.0e-06 内一致（阶段 1 为参照）
```

</details>

## 性能测试（独立目录）

三种方式，都在 `benchmark/`：

```powershell
# 方式一：命令行对比程序（读矩阵文件，逐组打印表格与加速比）
go run ./benchmark -in e2e/testdata/case01/input.matrix -repeat 3
go build -o bin/benchmark.exe ./benchmark

# 方式二：尺寸扫描（自己生成固定种子矩阵，逐尺寸对比；小尺寸用批量计时避免时钟量化）
go run ./benchmark -sizes all -repeat 3
go run ./benchmark -sizes 64,128,384,640,1024,2048

# 方式三：Go 标准基准（每方法独立测量，最适合横向比较）
go test ./benchmark -bench . -benchtime=2s -count=3
go test ./benchmark -run TestStagesMatchStage1   # 各阶段一致性（多种尺寸）
```

实测（本机 Core Ultra 7 155H，16 核 22 线程），1024×1024、`-repeat 3`：

```text
--- total (3 sets) ---
stage1-naive              1337.88 ms        4.8     1.00x          -
stage2-blocked             411.04 ms       15.7     3.25x   0.00e+00
stage3-avx2-1d              48.54 ms      132.7    27.56x   1.71e-13
stage4-avx2-packed-2d       36.63 ms      175.9    36.52x   1.71e-13
stage5-adaptive             35.54 ms      181.3    37.65x   1.71e-13
```

> 注：命令行程序里各阶段在同一进程、同一热状态下连续测量（阶段 1 先跑 ~1.3 s，
> 影响后续阶段的散热/频率），因此阶段 2 的加速比偏低；Go 基准每个方法独立测量，数值更稳定。
> 绝对加速比还会随机器状态浮动（同一台机器上阶段 1 实测在 440~560 ms 之间），
> 所以比较时请以同一轮测量的相对值为准。


## 数据文件格式（二进制，小端序）

| 偏移 | 大小 | 含义 |
| --- | --- | --- |
| 0 | 8 | 魔数 `MXMATRIX` |
| 8 | 4 | 版本号 = 1 |
| 12 | 4 | 矩阵组数 = 3 |
| 16 | 4 | 矩阵维数 n = 1024 |
| 20 | 8 | 随机种子 = 42 |

随后每组按行主序写入 `float64`：`input.matrix` 每组写 A、B 各 `n*n` 个；
`result.matrix` 每组写 C = A×B。已提交样例见 [e2e/testdata/case01](e2e/testdata/case01/)。
读写实现见 [internal/matrixio](internal/matrixio/matrixio.go)。

## 技术要点

1. **微内核 2 行 × 16 列**（[kernel_amd64.s](internal/kernel/kernel_amd64.s)）：
   每次 k 迭代 4 次 B 载入（16 个 double）+ 2 个 A 广播 + 8 条 FMA，
   累加器 8 个 ymm，广播（port5）与 FMA（2 端口）压力均衡；
   不用 4×16 是因为 AVX2 每个 ymm 只能放 4 个 double，4×16 需要 16 个累加器，超出寄存器上限。
2. **B 打包**（阶段 4/5）：把 B 按 (k 块 × j 块) 重排为连续内存，内核读 B 的步长从 8 KiB
   降到 512 B（128 B 缓存行全用上），消除跨行稀疏访存的缓存/TLB 压力。
3. **2D 任务划分**：阶段 4 用固定 8×16 = 128 个任务，每任务只读自己的 A 行带 + B 列带；
   阶段 5 改为固定 32 行 × 64 列一带（列带宽度取内核宽度 16 的整数倍），
   既保证列尾只出现在最后一个列带，也让任务数随 n 自然伸缩。
4. **自适应而非固定参数**（阶段 5，见 [stage5/README](stage5/)）：阶段 4 有三个固定参数
   在小尺寸/非对齐尺寸上会反过来伤性能——列按 16 等分使 `colsPer=ceil(n/16)` 常不是
   16 的倍数（列尾标量兜底以 64 个 double 的步长跨行扫打包 B，每个元素吃一条 cache line，
   n=384/640 时塌陷 6~10 倍）；`bpack` 恒按 512×64 分配（n=16 时分配 256 KiB 去装 2 KiB 的 B）；
   恒定 128 个任务（n=16 时 128 个 goroutine 算 256 个元素）。阶段 5 分别用
   "固定 32×64 带 + 列尾连续访问"、"按实际 klen 分配"、"n < 720 时免打包直接算" 解决。
5. **不做 cgo**：本环境任何 cgo 二进制都无法启动（最小 hello 程序同样失败），
   所以 SIMD 只能走 Go plan9 汇编；非 amd64 平台提供标量兜底实现保证可移植。
6. **PGO 实测无效**：热点 89.5% 在汇编内核，PGO 只作用于编译器生成的 Go 代码；
   对朴素实现 PGO 反而因把大循环内联进调用方而慢约 6%。

## 正确性

- 各阶段与阶段 1 的最大绝对误差 ≤ 3.5e-13（FMA 与 mul+add 的舍入顺序差异），容差 `1e-6`；
- 阶段 1 自身用 2×2/3×3 手工用例、单位矩阵、累加语义测试保证；
- `cmd/matrix verify` 对 3 组 × 5 个阶段全量交叉校验；
- `benchmark/bench_test.go` 覆盖 1,2,3,7,16,63,64,65,127,128,129,192,240,241,256,384,512,640,767,768,769,1024
  等尺寸（含内核宽度、行带/列带、k 块、打包阈值与阶段 4 的塌陷边界）；
- `stage5/adaptive_test.go` 另外显式校验阈值两侧（719/720/721 走不同分支但结果一致）；
- `go test ./e2e` 走命令行全流程。用例 1 对照 `e2e/testdata/case01`，用例 2–11 对照 `e2e/testdata/case02`–`case11`；其余用例覆盖精确小矩阵、分块边界、默认可复现性与失败路径。

## 参考

- https://github.com/tpoisonooo/how-to-optimize-gemm