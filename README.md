# matrixmul — 1024×1024 矩阵乘法（Go，四阶段演进）

用 Go 从零实现 1024×1024 双精度矩阵乘法，按**四个阶段**逐步优化，最终相对朴素实现
加速约 **50×**（峰值 ~286 GF/s）。工程按阶段分目录组织，性能测试独立成目录。

- 随机矩阵固定种子（42）生成并落盘为 `input.matrix`，可复现、可复用；
- 计算结果写入 `result.matrix`，由阶段 1 独立重算交叉校验；
- SIMD 用 **Go plan9 汇编手写 AVX2/FMA 微内核**（不使用 cgo，本环境 cgo 二进制无法运行）。

## 目录结构

```
matrix/
├── go.mod                       module matrixmul
├── README.md                    本文件（总览）
├── input.matrix                 3 组 1024×1024 随机矩阵（48 MiB，种子 42，可复现）
├── result.matrix                计算结果（24 MiB）
├── cmd/matrix/                  命令行工具：generate / multiply / verify
├── internal/
│   ├── matrixio/                二进制矩阵文件格式读写（共享）
│   └── kernel/                  AVX2/FMA 2×16 微内核（汇编，阶段 3/4 共享）
├── stage1/                      阶段 1：朴素三重循环（基线）
├── stage2/                      阶段 2：纯 Go 分块 + goroutine 行并行
├── stage3/                      阶段 3：AVX2/FMA 微内核 + 一维行并行
├── stage4/                      阶段 4：AVX2/FMA + B 打包 + 2D 行列划分（最优）
└── benchmark/                   性能测试（独立程序 + Go 标准基准）
```

每个阶段目录内含实现、测试与 `README.md`（思路、瓶颈、实测数据）。

## 四个阶段与性能

| 阶段 | 目录 | 核心方法 | 单次耗时 | 吞吐 | 相对阶段 1 |
| --- | --- | --- | --- | --- | --- |
| 1 | [stage1](stage1/) | 朴素三重循环（单线程） | ~559 ms | ~3.9 GF/s | 1× |
| 2 | [stage2](stage2/) | 分块 64 + goroutine 行并行 | ~82 ms | ~26 GF/s | **6.8×** |
| 3 | [stage3](stage3/) | AVX2/FMA 汇编微内核 + 一维行并行 | ~15.3 ms | ~140 GF/s | **36.5×** |
| 4 | [stage4](stage4/) | 同上内核 + **B 打包** + **2D 行列划分** | ~10.2 ms | ~210 GF/s | **54.6×** |

- 单次耗时/吞吐来自 `go test ./benchmark -bench .`（1024×1024，GOMAXPROCS=22，取 3 次均值）；
- 命令行性能测试（`go run ./benchmark`）3 组合计：阶段 1 1648.9 ms → 阶段 4 33.0 ms = **49.96×**；
- 阶段 4 单组实测峰值 **285.9 GF/s**（8.51 ms/组）；
- 各阶段均为 `C += A×B` 累加语义，调用前需 `clear(C)`。

## 快速开始

```powershell
# 1) 生成随机矩阵（固定种子，可复现；已存在时可跳过）
go run ./cmd/matrix generate

# 2) 用指定阶段计算：multiply [1|2|3|4]，默认 4（最优）
go run ./cmd/matrix multiply 4

# 3) 校验：阶段 1 独立重算校验 result.matrix，并交叉校验全部 4 个阶段
go run ./cmd/matrix verify
```

也可构建二进制：`go build -o bin/matrix.exe ./cmd/matrix`。

<details>
<summary>输出示例（verify）</summary>

```text
group 1/3: result.matrix          vs stage1: max abs error = 1.705e-13  [OK]
group 1/3: stage2-blocked         vs stage1: max abs error = 0.000e+00  [OK]
group 1/3: stage3-avx2-1d         vs stage1: max abs error = 1.705e-13  [OK]
group 1/3: stage4-avx2-packed-2d  vs stage1: max abs error = 1.705e-13  [OK]
...
verify passed: 3 组 × 4 个阶段全部在容差 1.0e-06 内一致（阶段 1 为参照）
```

</details>

## 性能测试（独立目录）

两种方式，都在 `benchmark/`：

```powershell
# 方式一：命令行对比程序（读 input.matrix，逐组打印表格与加速比）
go run ./benchmark                 # 可选 -in/-repeat，例如 -repeat 3 取最快一次
go build -o bin/benchmark.exe ./benchmark

# 方式二：Go 标准基准（每方法独立测量，更适合横向比较）
go test ./benchmark -bench . -benchtime=2s -count=3
go test ./benchmark -run TestStagesMatchStage1   # 4 阶段一致性（多种尺寸）
```

实测（本机 Core Ultra 7 155H，16 核 22 线程）：

```text
--- total (3 sets) ---
stage1-naive              1648.87 ms        3.9     1.00x          -
stage2-blocked             383.20 ms       16.8     4.30x   0.00e+00
stage3-avx2-1d              57.21 ms      112.6    28.82x   1.71e-13
stage4-avx2-packed-2d       33.01 ms      195.2    49.96x   1.71e-13
```

> 注：命令行程序里各阶段在同一进程、同一热状态下连续测量（阶段 1 会先跑 ~1.6 s，
> 影响后续阶段的散热/频率），因此阶段 2 的加速比偏低；Go 基准每个方法独立测量，数值更稳定。

## 数据文件格式（二进制，小端序）

| 偏移 | 大小 | 含义 |
| --- | --- | --- |
| 0 | 8 | 魔数 `MXMATRIX` |
| 8 | 4 | 版本号 = 1 |
| 12 | 4 | 矩阵组数 = 3 |
| 16 | 4 | 矩阵维数 n = 1024 |
| 20 | 8 | 随机种子 = 42 |

随后每组按行主序写入 `float64`：`input.matrix` 每组写 A、B 各 `n*n` 个；
`result.matrix` 每组写 C = A×B。读写实现见 [internal/matrixio](internal/matrixio/matrixio.go)。

## 技术要点

1. **微内核 2 行 × 16 列**（[kernel_amd64.s](internal/kernel/kernel_amd64.s)）：
   每次 k 迭代 4 次 B 载入（16 个 double）+ 2 个 A 广播 + 8 条 FMA，
   累加器 8 个 ymm，广播（port5）与 FMA（2 端口）压力均衡；
   不用 4×16 是因为 AVX2 每个 ymm 只能放 4 个 double，4×16 需要 16 个累加器，超出寄存器上限。
2. **B 打包**：把 B 按 (k 块 × j 块) 重排为连续内存，内核读 B 的步长从 8 KiB
   降到 512 B（128 B 缓存行全用上），消除跨行稀疏访存的缓存/TLB 压力。
3. **2D 任务划分**：行组 × 列组（8×16 = 128 任务），每任务只读自己的 A 行带 + B 列带，
   总访存量从 ~256 MiB 降到 ~64 MiB。
4. **不做 cgo**：本环境任何 cgo 二进制都无法启动（最小 hello 程序同样失败），
   所以 SIMD 只能走 Go plan9 汇编；非 amd64 平台提供标量兜底实现保证可移植。
5. **PGO 实测无效**：热点 89.5% 在汇编内核，PGO 只作用于编译器生成的 Go 代码；
   对朴素实现 PGO 反而因把大循环内联进调用方而慢约 6%。

## 正确性

- 各阶段与阶段 1 的最大绝对误差 ≤ 1.7e-13（FMA 与 mul+add 的舍入顺序差异），容差 `1e-6`；
- 阶段 1 自身用 2×2/3×3 手工用例、单位矩阵、累加语义测试保证；
- `cmd/matrix verify` 对 3 组 × 4 个阶段全量交叉校验；
- `benchmark/bench_test.go` 覆盖 1,2,3,7,16,63,64,65,127,128,129,256,1024 等尺寸（含任务/分块边界）。
