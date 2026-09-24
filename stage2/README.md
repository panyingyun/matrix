# 阶段 2：纯 Go 分块并行

## 思路

在阶段 1 的 `i-k-j` 循环上做两件事：

1. **分块（blocked/tiled）**：块边长 64，`64×64×8B = 32KiB` 适配 L1/L2，
   把"每行扫描整个 B（8 MiB）"变成"块内扫描 B（128 KiB）"；
2. **多核并行**：按 16 行/任务派发 goroutine；各任务只写 `C` 的互不重叠行，
   读共享的 `A`/`B`，无需加锁。

## 实测

| 指标 | 数值 |
| --- | --- |
| 单次耗时 | ~85–100 ms |
| 吞吐 | ~21–25 GF/s |
| 相对阶段 1 | **约 6.5×** |

仍然是纯 Go 标量代码：Go 编译器（gc）不会自动向量化，所以内层循环
每个元素仍要一条 `MULSD`+`ADDSD`，单核算力受限——这正是阶段 3 要解决的问题。

## 代码

- [blocked.go](blocked.go) — `Mul`（分块 + 行并行）
- [blocked_test.go](blocked_test.go) — 与阶段 1 交叉校验（含 1024 规模）

## 单独运行

```powershell
go test ./stage2
go run ./cmd/matrix multiply 2
```
