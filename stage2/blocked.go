// Package stage2 实现阶段 2：纯 Go 分块（blocked/tiled）+ goroutine 按行并行。
//
// 相对阶段 1 的两点改进：
//  1. i-k-j 分块，块边长 64（64×64×8B = 32KiB，适配 L1/L2 缓存），
//     把"整行扫描 B"变成"块内扫描 B"，显著提高缓存命中率；
//  2. 按 16 行/任务派发 goroutine，多核并行；各任务只写 C 的不同行，无需加锁。
//
// 实测（1024×1024 float64，单次）：约 85–100 ms，~21–25 GF/s，相对阶段 1 加速约 6.5×。
package stage2

import "sync"

const (
	blockSize   = 64 // 分块边长：64*64*8B = 32KiB
	rowsPerTask = 16 // 每个 goroutine 处理的连续行数
)

// Mul 计算 C += A*B（行主序 n×n），分块 + 行并行。调用前 C 需清零（累加语义）。
func Mul(A, B, C []float64, n int) {
	var wg sync.WaitGroup
	for i0 := 0; i0 < n; i0 += rowsPerTask {
		i1 := min(i0+rowsPerTask, n)
		wg.Add(1)
		go func(i0, i1 int) {
			defer wg.Done()
			mulRows(A, B, C, n, i0, i1)
		}(i0, i1)
	}
	wg.Wait()
}

// mulRows 计算 C[i0:i1][:] += A[i0:i1][:] * B，使用 i-k-j 分块。
func mulRows(A, B, C []float64, n, i0, i1 int) {
	for ii := i0; ii < i1; ii += blockSize {
		iEnd := min(ii+blockSize, i1)
		for k0 := 0; k0 < n; k0 += blockSize {
			kEnd := min(k0+blockSize, n)
			for j0 := 0; j0 < n; j0 += blockSize {
				jEnd := min(j0+blockSize, n)
				for i := ii; i < iEnd; i++ {
					ci := C[i*n : i*n+jEnd]
					ai := A[i*n : i*n+n]
					for k := k0; k < kEnd; k++ {
						aik := ai[k]
						if aik == 0 {
							continue
						}
						bk := B[k*n : k*n+jEnd]
						for j := j0; j < jEnd; j++ {
							ci[j] += aik * bk[j]
						}
					}
				}
			}
		}
	}
}
