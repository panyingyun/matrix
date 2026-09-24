# 阶段 3：AVX2/FMA 汇编微内核 + 一维行并行

## 思路

把阶段 2 的**内层标量循环**换成手写 AVX2 + FMA 汇编微内核
（[internal/kernel](../internal/kernel/kernel_amd64.s) 的 `Mul2x16`）：

- 微内核形状 **2 行 × 16 列**，累加器 8 个 `ymm`；
- 每次 k 迭代：4 次 B 行载入（16 个 double）+ 2 个 A 元素广播 + 8 条 FMA，
  广播（port5）与 FMA（2 个 FMA 端口）压力均衡；
- 为什么不是 4×16：AVX2 每个 `ymm` 只放 4 个 double，4×16 需要 16 个累加器，
  超出 16 个 ymm 寄存器上限（这是实现中踩过的坑）；
- 外层保留 j/k 分块（64/256）与 goroutine 并行（64 行/任务，共 16 个任务）。

## 为什么还不够快

任务按**一维行**划分：每个任务都要以自己的行带去乘整个 B，
B 的读取步长是 8 KiB（每次只取 64B 中的 16 个 double），
16 个任务 × 8 MiB ≈ 128 MiB 的稀疏访存 → 缓存行利用率低、TLB 压力大、
多核扩展性差（22 线程只用到 ~20% 的机器浮点峰值）。

## 实测

| 指标 | 数值 |
| --- | --- |
| 单次耗时 | ~22–25 ms |
| 吞吐 | ~90–96 GF/s |
| 相对阶段 1 | **约 24–27×** |

## 代码

- [avx2.go](avx2.go) — `Mul`（一维行划分 + `kernel.Mul2x16`）
- [avx2_test.go](avx2_test.go) — 与阶段 1 交叉校验
- 微内核：[internal/kernel/kernel_amd64.s](../internal/kernel/kernel_amd64.s)

## 单独运行

```powershell
go test ./stage3
go run ./cmd/matrix multiply 3
```
