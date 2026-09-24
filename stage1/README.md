# 阶段 1：朴素三重循环（基线）

## 思路

最直接的三重循环 `i-k-j`：对每个输出行 `i`，沿 `k` 累加 `A[i][k] * B[k][:]`，
内层 `j` 循环访问行主序的 `B[k][j]`、`C[i][j]`，两行都是连续内存。

## 性能瓶颈

单线程、无分块，每算一行 `C` 都要完整扫一遍 `B`：

```
1024 行 × 8 MiB(B) ≈ 8 GiB 访存 / 次乘法
```

远远超出缓存容量，几乎每次都是内存访问 → **纯内存带宽受限**。

## 实测（本机 Core Ultra 7 155H）

| 指标 | 数值 |
| --- | --- |
| 单次耗时 | ~520–650 ms |
| 吞吐 | ~3.9 GF/s |
| 加速比 | 1×（基准） |

## 代码

- [naive.go](naive.go) — `Mul(A, B, C []float64, n int)`，累加语义（调用前 `clear(C)`）
- [naive_test.go](naive_test.go) — 手工小矩阵 + 单位矩阵校验（基线自身的正确性）

## 单独运行

```powershell
go test ./stage1
go run ./cmd/matrix multiply 1     # 用阶段 1 生成 result.matrix
```
