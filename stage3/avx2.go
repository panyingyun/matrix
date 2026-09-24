// Package stage3 实现阶段 3：AVX2/FMA 汇编微内核 + 按行一维并行。
//
// 相对阶段 2 的核心变化：把内层标量循环换成手写 AVX2+FMA 汇编微内核
// （internal/kernel.Mul2x16，2 行 × 16 列，每次 k 迭代 8 条 FMA），
// 外层仍保持 j/k 分块与 goroutine 并行。
//
// 局限：任务按一维行划分（64 行/任务），每个任务都要以 8KiB 跨行步长
// 流式扫描整个 B（8MiB × 16 任务 ≈ 128MiB），缓存行只用到 16/64 个 double，
// TLB 与缓存压力大，并行扩展性受限。
//
// 实测（1024×1024 float64，单次）：约 22–25 ms，~90–96 GF/s，相对阶段 1 加速约 24–27×。
package stage3

import (
	"sync"

	"matrixmul/internal/kernel"
)

const (
	taskRows = 64  // 每个 goroutine 负责的连续行数（1024/64 = 16 个任务）
	blockJ   = 64  // j 分块边长
	blockK   = 256 // k 分块边长
)

// Mul 计算 dst += A*B（行主序 n×n），AVX2/FMA 微内核 + 一维行并行。
// 调用前 dst 需清零（累加语义）。
//
// 需要 CPU 支持 AVX2+FMA。
func Mul(A, B, dst []float64, n int) {
	var wg sync.WaitGroup
	for i0 := 0; i0 < n; i0 += taskRows {
		i1 := min(i0+taskRows, n)
		wg.Add(1)
		go func(i0, i1 int) {
			defer wg.Done()
			mulRange(A, B, dst, n, i0, i1)
		}(i0, i1)
	}
	wg.Wait()
}

// mulRange 计算 dst[i0:i1][:] += A[i0:i1][:] * B（直接在原矩阵上按块读取 B）。
func mulRange(A, B, dst []float64, n, i0, i1 int) {
	for jj := 0; jj < n; jj += blockJ {
		jEnd := min(jj+blockJ, n)
		for kk := 0; kk < n; kk += blockK {
			kEnd := min(kk+blockK, n)
			klen := kEnd - kk
			for i := i0; i < i1; i += 2 {
				if i+1 >= i1 {
					// 末尾单行：标量兜底
					ai := A[i*n+kk : i*n+kEnd]
					row := dst[i*n+jj : i*n+jEnd]
					for k := 0; k < klen; k++ {
						aik := ai[k]
						if aik == 0 {
							continue
						}
						bk := B[(kk+k)*n+jj : (kk+k)*n+jEnd]
						for j := 0; j < len(bk); j++ {
							row[j] += aik * bk[j]
						}
					}
					continue
				}
				// 2 行 × 16 列 AVX2/FMA 微内核
				j := jj
				for ; j+16 <= jEnd; j += 16 {
					kernel.Mul2x16(klen, &A[i*n+kk], &B[kk*n+j], &dst[i*n+j], n, n, n)
				}
				// 列尾标量兜底（2 行 × 剩余列）
				for ; j < jEnd; j++ {
					for k := 0; k < klen; k++ {
						a0 := A[i*n+kk+k]
						a1 := A[(i+1)*n+kk+k]
						bkj := B[(kk+k)*n+j]
						dst[i*n+j] += a0 * bkj
						dst[(i+1)*n+j] += a1 * bkj
					}
				}
			}
		}
	}
}
