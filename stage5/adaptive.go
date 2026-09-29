// Package stage5 实现阶段 5（当前最优）：自适应划分的 AVX2/FMA 矩阵乘法。
//
// 与阶段 4 共用同一个汇编微内核（internal/kernel.Mul2x16），区别是把"固定参数"
// 改成"按 n 自适应"，从而消除阶段 4 在小尺寸与非对齐尺寸上的性能塌陷：
//
//  1. 任务固定按 32 行 × 64 列一带划分。列带宽度 64 是内核宽度 16 的整数倍，
//     列尾只可能出现在最后一个列带，且总是连续访问（阶段 4 把列按 16 等分，
//     colsPer=ceil(n/16) 不是 16 的倍数时，列尾标量兜底会以 jb=64 个 double 的
//     跨行步长扫打包 B，每个元素消耗一条 cache line，n=384/640 时塌陷 3~10 倍）。
//  2. 小尺寸（n < packThreshold）不做 B 打包：此时单个矩阵 ≤ 4.5 MiB，可驻留 L3，
//     打包带来的局部性收益抵不过一次完整拷贝与分配，直接在原矩阵上用微内核更快。
//  3. 打包缓冲区按每个 k 块的实际长度 klen 分配（阶段 4 恒按 kb×jb 分配，
//     n=16 时要分配 256 KiB 去装 2 KiB 的 B）。
//  4. 所有标量兜底都是 k 外层、列内层，保持缓存行完整利用。
//
// 实测（1024×1024 float64，Core Ultra 7 155H，GOMAXPROCS=22）：约 8.5–9.5 ms，
// 约 230–250 GF/s，相对阶段 1 约 55–63×；在 192/384/640 等阶段 4 塌陷的尺寸上，
// 相对阶段 1 仍有 25–45×（阶段 4 只有 1.5–6.6×）。
package stage5

import (
	"sync"

	"matrixmul/internal/kernel"
)

const (
	bandRows = 32 // 每个任务的行带高度
	bandCols = 64 // 每个任务的列带宽度（必须是内核宽度 16 的整数倍，同时是打包列块边长）
	blockK   = 512 // 打包的 k 块边长

	// packThreshold 是启用 B 打包的最小维数：n < 该值时直接访问 B 更快。
	//
	// 依据（本机 Core Ultra 7 155H，24 MiB L3，交错批量实测）：n ≤ 704 时
	// 直接访问快 3%~8%，n ≥ 767 时打包快 10%~21%，n = 1024 时打包快 2.2 倍；
	// 即交叉点落在 704~767 之间，取 720（此时单个矩阵 ≈ 4 MiB，接近 L3 的有效驻留上限）。
	packThreshold = 720
)

// Mul 计算 dst += A*B（行主序 n×n）。调用前 dst 需清零（累加语义）。
//
// 需要 CPU 支持 AVX2+FMA；非 amd64 平台由 internal/kernel 的标量兜底实现保证可移植。
func Mul(A, B, dst []float64, n int) {
	if n <= 0 {
		return
	}
	if n < packThreshold {
		mulDirect(A, B, dst, n)
		return
	}
	mulPacked(A, B, dst, n)
}

// mulDirect 不打包：直接在原矩阵上用微内核（B 的行跨度就是 n），2D 任务划分。
func mulDirect(A, B, dst []float64, n int) {
	nR := (n + bandRows - 1) / bandRows
	nC := (n + bandCols - 1) / bandCols
	var wg sync.WaitGroup
	for r := 0; r < nR; r++ {
		i0 := r * bandRows
		i1 := min(i0+bandRows, n)
		for c := 0; c < nC; c++ {
			j0 := c * bandCols
			j1 := min(j0+bandCols, n)
			wg.Add(1)
			go func(i0, i1, j0, j1 int) {
				defer wg.Done()
				// colBase=0：B 的列 j 在 bsrc 中的下标就是 j，行跨度 ldb=n
				mulBand(A, B, dst, n, 0, n, n, 0, i0, i1, j0, j1)
			}(i0, i1, j0, j1)
		}
	}
	wg.Wait()
}

// mulPacked 把 B 按 (k 块 × 列带) 重排为连续内存，再做 2D 任务划分。
// 内核读 B 的行跨度从 n（最大 16 KiB）降到 bandCols（512 B），缓存行与 TLB 压力大幅下降。
func mulPacked(A, B, dst []float64, n int) {
	nCol := (n + bandCols - 1) / bandCols // 列带数，同时也是打包的 j 块数
	nKK := (n + blockK - 1) / blockK

	// klen 返回第 kk 个 k 块的实际长度（最后一块可能不足 blockK）。
	klen := func(kk int) int {
		k0 := kk * blockK
		return min(k0+blockK, n) - k0
	}

	// 1) 按实际 klen 分配打包缓冲区，并并行打包 B：
	//    bpack[kk][jj][k][j]，同一 k 行的 bandCols 个 double 连续。
	offs := make([]int, nKK+1)
	for kk := 0; kk < nKK; kk++ {
		offs[kk+1] = offs[kk] + nCol*klen(kk)*bandCols
	}
	bpack := make([]float64, offs[nKK])

	var pw sync.WaitGroup
	for jj := 0; jj < nCol; jj++ {
		pw.Add(1)
		go func(jj int) {
			defer pw.Done()
			j0 := jj * bandCols
			j1 := min(j0+bandCols, n)
			jw := j1 - j0
			for kk := 0; kk < nKK; kk++ {
				kl := klen(kk)
				k0 := kk * blockK
				blk := bpack[offs[kk]+jj*kl*bandCols : offs[kk]+(jj+1)*kl*bandCols]
				for k := 0; k < kl; k++ {
					copy(blk[k*bandCols:k*bandCols+jw], B[(k0+k)*n+j0:(k0+k)*n+j1])
				}
			}
		}(jj)
	}
	pw.Wait()

	// 2) 行带 × 列带的 2D 任务：每个任务只读自己的打包列块。
	var wg sync.WaitGroup
	for r := 0; r < (n+bandRows-1)/bandRows; r++ {
		i0 := r * bandRows
		i1 := min(i0+bandRows, n)
		for c := 0; c < nCol; c++ {
			j0 := c * bandCols
			j1 := min(j0+bandCols, n)
			wg.Add(1)
			go func(i0, i1, j0, j1 int) {
				defer wg.Done()
				for kk := 0; kk < nKK; kk++ {
					kl := klen(kk)
					k0 := kk * blockK
					bp := bpack[offs[kk]+(j0/bandCols)*kl*bandCols:]
					// colBase=j0：bp 的第 0 列对应矩阵的列 j0，行跨度 ldb=bandCols
					mulBand(A, bp, dst, n, k0, kl, bandCols, j0, i0, i1, j0, j1)
				}
			}(i0, i1, j0, j1)
		}
	}
	wg.Wait()
}

// mulBand 计算 dst[i0:i1][j0:j1] += A[i0:i1][k0:k0+klen] × bsrc[0:klen][j0-colBase : j1-colBase]。
//
// bsrc 的列 j 位于下标 k*ldb + (j - colBase)；每 16 列走一次 AVX2/FMA 微内核，
// 两行一组；不足 16 列的列尾与末尾单行走标量兜底（k 外层、列内层，连续访问）。
func mulBand(A, bsrc, dst []float64, n, k0, klen, ldb, colBase, i0, i1, j0, j1 int) {
	for i := i0; i < i1; i += 2 {
		if i+1 >= i1 {
			// 末尾单行：标量兜底
			for k := 0; k < klen; k++ {
				a0 := A[i*n+k0+k]
				brow := bsrc[k*ldb+(j0-colBase) : k*ldb+(j1-colBase)]
				drow := dst[i*n+j0 : i*n+j1]
				for x, bv := range brow {
					drow[x] += a0 * bv
				}
			}
			continue
		}
		j := j0
		for ; j+16 <= j1; j += 16 {
			kernel.Mul2x16(klen, &A[i*n+k0], &bsrc[j-colBase], &dst[i*n+j], n, ldb, n)
		}
		if j < j1 {
			// 列尾（仅可能出现在最后一个列带）：同样 k 外层、列内层
			for k := 0; k < klen; k++ {
				a0 := A[i*n+k0+k]
				a1 := A[(i+1)*n+k0+k]
				brow := bsrc[k*ldb+(j-colBase) : k*ldb+(j1-colBase)]
				d0 := dst[i*n+j : i*n+j1]
				d1 := dst[(i+1)*n+j : (i+1)*n+j1]
				for x, bv := range brow {
					d0[x] += a0 * bv
					d1[x] += a1 * bv
				}
			}
		}
	}
}
