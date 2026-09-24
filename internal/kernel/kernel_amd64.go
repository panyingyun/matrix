//go:build amd64

// Package kernel 提供手写 AVX2/FMA 汇编实现的矩阵乘法微内核，
// 供阶段 3、阶段 4 复用（两个阶段共用同一微内核，差异只在数据划分与打包策略）。
package kernel

// Mul2x16 计算 C[0:2][0:16] += A[0:2][0:k] * B[0:k][0:16]（行主序 float64）。
//
// 参数：
//
//	k   求和长度（A 的列数 / B 的行数）
//	a   A 的起始元素指针（A[i0][k0]）
//	b   B 的起始元素指针（B[k0][j0]）
//	c   C 的起始元素指针（C[i0][j0]）
//	lda A 的行跨度（元素个数，非字节）
//	ldb B 的行跨度
//	ldc C 的行跨度
//
// 由 kernel_amd64.s 用 AVX2+FMA 实现：每次 k 迭代 4 次 B 载入（16 个 double）
// + 2 个 A 元素广播 + 8 条 FMA（每条 4 double × 2 flops），累加器 8 个 ymm，
// 广播(port5)与 FMA(2 端口)压力均衡。
//
//go:noescape
func Mul2x16(k int, a, b, c *float64, lda, ldb, ldc int)
