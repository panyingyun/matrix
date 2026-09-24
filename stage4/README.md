# 阶段 4：AVX2/FMA + B 并行打包 + 2D 行列划分（最优）

## 思路

复用与阶段 3 **完全相同**的汇编微内核（[internal/kernel](../internal/kernel/kernel_amd64.s)），
只改数据布局与任务划分——这正是"同一内核、更好的访存策略"带来的收益：

1. **B 并行打包**：把 B 按 `(k 块 × j 块)` 重排为连续内存
   `bpack[kkIdx][jjIdx][k][j]`。内核读 B 的步长从 8 KiB（每行只用 16/64 个 double）
   降到 `jb = 64` 个 double（512B，连续 128B 全用上），缓存行利用率与 TLB 行为大幅改善；
2. **2D 任务划分**：任务 = 行组 × 列组（默认 8 × 16 = 128 个），
   每个任务只读自己的 A 行带 + B 列带（~4 MiB），
   总访存量从阶段 3 的 ~256 MiB 降到 ~64 MiB。

## 实测

| 指标 | 数值 |
| --- | --- |
| 单次耗时 | ~9.5–12 ms |
| 吞吐 | ~180–226 GF/s |
| 相对阶段 1 | **约 50–68×** |
| 相对阶段 3 | 约 2.3× |

## 代码

- [packed.go](packed.go) — `Mul`（打包 + 2D 划分 + `kernel.Mul2x16`）
- [packed_test.go](packed_test.go) — 与阶段 1 交叉校验（含 1024 规模与边界尺寸）
- 微内核：[internal/kernel/kernel_amd64.s](../internal/kernel/kernel_amd64.s)

## 单独运行

```powershell
go test ./stage4
go run ./cmd/matrix multiply 4     # 默认阶段
```
