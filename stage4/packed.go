// Package stage4 实现阶段 4（当前最优）：AVX2/FMA 微内核 + B 并行打包 + 2D 行列划分。
//
// 复用与阶段 3 相同的汇编微内核（internal/kernel.Mul2x16），
// 但在数据布局与任务划分上做两点改进：
//
//  1. B 按 (k 块 × j 块) 并行打包为连续内存：内核的 B 载入在块内密排
//     （k 行步长仅 jb = 64 个 double，即 512B），消除阶段 3 中 8KiB 跨行步长
//     造成的缓存行浪费与 TLB 压力；
//  2. 任务按**行组 × 列组**二维划分（默认 8 × 16 = 128 个任务）：每个任务只读
//     自己的 A 行带与 B 列带，总访存量从阶段 3 的 ~256MiB 降到 ~64MiB。
//
// 实测（1024×1024 float64，单次）：约 9.5–12 ms，~180–226 GF/s，相对阶段 1 加速约 50–68×。
package stage4

import (
	"sync"

	"matrixmul/internal/kernel"
)

const (
	blockJ    = 64  // 打包 B 的 j 块边长
	blockK    = 512 // k 块边长
	rowGroups = 8   // 行组数（每任务行数 = n/rowGroups）
	colGroups = 16  // 列组数（每任务列数 = n/colGroups）
)

// Mul 计算 dst += A*B（行主序 n×n），AVX2/FMA 微内核 + B 打包 + 2D 任务划分。
// 调用前 dst 需清零（累加语义）。
//
// 需要 CPU 支持 AVX2+FMA。
func Mul(A, B, dst []float64, n int) {
	jb, kb := blockJ, blockK
	nJJ := (n + jb - 1) / jb
	nKK := (n + kb - 1) / kb

	// 1) 并行打包 B -> bpack[kkIdx][jjIdx][k][j]（k 外层、j 内层连续）
	bpack := make([]float64, nKK*nJJ*kb*jb)
	var packWG sync.WaitGroup
	for jjIdx := 0; jjIdx < nJJ; jjIdx++ {
		packWG.Add(1)
		go func(jjIdx int) {
			defer packWG.Done()
			j0 := jjIdx * jb
			j1 := min(j0+jb, n)
			jw := j1 - j0
			for kkIdx := 0; kkIdx < nKK; kkIdx++ {
				k0 := kkIdx * kb
				k1 := min(k0+kb, n)
				blk := bpack[(kkIdx*nJJ+jjIdx)*kb*jb : (kkIdx*nJJ+jjIdx+1)*kb*jb]
				for k := k0; k < k1; k++ {
					copy(blk[(k-k0)*jb:(k-k0)*jb+jw], B[k*n+j0:k*n+j1])
				}
			}
		}(jjIdx)
	}
	packWG.Wait()

	// 2) 2D 任务：行组 × 列组
	rowsPer := (n + rowGroups - 1) / rowGroups
	colsPer := (n + colGroups - 1) / colGroups
	var wg sync.WaitGroup
	for r := 0; r < rowGroups; r++ {
		i0 := r * rowsPer
		i1 := min(i0+rowsPer, n)
		for c := 0; c < colGroups; c++ {
			j0 := c * colsPer
			j1 := min(j0+colsPer, n)
			wg.Add(1)
			go func(i0, i1, j0, j1 int) {
				defer wg.Done()
				mulBlock(A, bpack, dst, n, kb, jb, nJJ, i0, i1, j0, j1)
			}(i0, i1, j0, j1)
		}
	}
	wg.Wait()
}

// mulBlock 计算 dst[i0:i1][j0:j1] += A[i0:i1][*] * B[*][j0:j1]（从打包 B 读取）。
func mulBlock(A, bpack, dst []float64, n, kb, jb, nJJ, i0, i1, j0, j1 int) {
	nKK := (n + kb - 1) / kb
	firstJJ := j0 / jb
	lastJJ := (j1 + jb - 1) / jb
	for kkIdx := 0; kkIdx < nKK; kkIdx++ {
		k0 := kkIdx * kb
		k1 := min(k0+kb, n)
		klen := k1 - k0
		for jjIdx := firstJJ; jjIdx < lastJJ; jjIdx++ {
			jj0 := jjIdx * jb
			jjA := max(jj0, j0) // 该打包块与任务列带 [j0,j1) 的交集
			jjB := min(jj0+jb, j1)
			if jjA >= jjB {
				continue
			}
			bp := bpack[(kkIdx*nJJ+jjIdx)*kb*jb:]
			for i := i0; i < i1; i += 2 {
				if i+1 >= i1 {
					// 末尾单行：标量兜底
					ai := A[i*n+k0 : i*n+k1]
					row := dst[i*n+jjA : i*n+jjB]
					for k := 0; k < klen; k++ {
						aik := ai[k]
						if aik == 0 {
							continue
						}
						brow := bp[k*jb+(jjA-jj0) : k*jb+(jjB-jj0)]
						for j := 0; j < len(brow); j++ {
							row[j] += aik * brow[j]
						}
					}
					continue
				}
				// 2 行 × 16 列 AVX2/FMA 微内核
				j := jjA
				for ; j+16 <= jjB; j += 16 {
					kernel.Mul2x16(klen, &A[i*n+k0], &bp[j-jj0], &dst[i*n+j], n, jb, n)
				}
				// 列尾标量兜底（2 行 × 剩余列）
				for ; j < jjB; j++ {
					for k := 0; k < klen; k++ {
						a0 := A[i*n+k0+k]
						a1 := A[(i+1)*n+k0+k]
						bkj := bp[k*jb+(j-jj0)]
						dst[i*n+j] += a0 * bkj
						dst[(i+1)*n+j] += a1 * bkj
					}
				}
			}
		}
	}
}
