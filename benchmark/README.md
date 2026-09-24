# 性能测试（独立）

本目录是本工程**独立**的性能测试入口，只负责测量与对比，不包含任何算法实现
（算法在 [stage1](../stage1/) ~ [stage4](../stage4/)）。

## 两种测量方式

### 1. 命令行对比程序 — `main.go`

在同一份 `input.matrix` 上逐组计时四个阶段，打印耗时、GFLOP/s、
相对阶段 1 的加速比与正确性误差：

```powershell
go run ./benchmark
go run ./benchmark -in input.matrix -repeat 3    # -repeat N：每组重复 N 次取最快
go build -o ../bin/benchmark.exe ./benchmark
```

### 2. Go 标准基准 — `bench_test.go`

每个阶段独立测量（进程内互不干扰，数值更稳定）：

```powershell
go test ./benchmark -bench . -benchtime=2s -count=3
go test ./benchmark -run TestStagesMatchStage1    # 阶段一致性（多尺寸）
```

- `TestStagesMatchStage1` 校验阶段 2~4 在 1…1024 多种尺寸下与阶段 1 一致（容差 1e-9）；
- 基准用与 `cmd/matrix generate` 相同的种子在内存中生成矩阵，无需读文件。

## 实测结果

1024×1024 float64，Core Ultra 7 155H（16 核 22 线程，GOMAXPROCS=22）：

**命令行程序（3 组合计）**

```text
stage1-naive              1648.87 ms        3.9     1.00x          -
stage2-blocked             383.20 ms       16.8     4.30x   0.00e+00
stage3-avx2-1d              57.21 ms      112.6    28.82x   1.71e-13
stage4-avx2-packed-2d       33.01 ms      195.2    49.96x   1.71e-13
```

**Go 标准基准（ns/op）**

| 基准 | 单次耗时 | 相对阶段 1 |
| --- | --- | --- |
| `BenchmarkStage1_Naive` | ~559 ms | 1× |
| `BenchmarkStage2_Blocked` | ~81.7 ms | 6.8× |
| `BenchmarkStage3_AVX2_1D` | ~15.3 ms | 36.5× |
| `BenchmarkStage4_AVX2_Packed_2D` | ~10.2 ms | **54.6×** |

> 两种方式的差异主要来自测量环境：命令行程序在同一进程内连续测量各阶段
> （阶段 1 先跑约 1.6 s，影响 CPU 热状态与频率），Go 基准则每个方法独立测量。

## 测量约定

- 所有阶段都是 `C += A×B` 累加语义，计时前对 `C` 执行 `clear(C)`，
  与算法正常使用方式一致（清零不计入耗时）；
- `GF/s = 2·n³ / 耗时`；
- 正确性误差以阶段 1 的结果为参照（阶段 2 与阶段 1 逐位一致，误差 0；
  阶段 3/4 因 FMA 舍入顺序差异约 1.7e-13）。
