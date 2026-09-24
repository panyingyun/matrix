// Package stage1 实现阶段 1：朴素三重循环矩阵乘法（基线实现）。
//
// 算法：i-k-j 顺序的三重循环，内层对行主序的 B、C 连续访问。
// 特点：单线程、无分块；每计算一行 C 都要完整扫描一遍 B，
// 1024 行 × 8MiB ≈ 8GiB 访存，完全受内存带宽制约，是四个阶段中最慢的参照实现。
//
// 实测（1024×1024 float64，单次）：约 520–650 ms，~3.9 GF/s。
package stage1

// Mul 计算 C += A*B（行主序 n×n）。
//
// 累加语义：调用前必须把 C 清零（clear(C)）。四个阶段保持同一语义，
// 便于横向对比与交叉校验。
func Mul(A, B, C []float64, n int) {
	for i := 0; i < n; i++ {
		ci := C[i*n : i*n+n]
		ai := A[i*n : i*n+n]
		for k := 0; k < n; k++ {
			aik := ai[k]
			if aik == 0 {
				continue
			}
			bk := B[k*n : k*n+n]
			for j := 0; j < n; j++ {
				ci[j] += aik * bk[j]
			}
		}
	}
}
