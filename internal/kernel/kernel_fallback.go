//go:build !amd64

package kernel

import "unsafe"

// Mul2x16 是非 amd64 平台上的标量兜底实现，保证工程可移植（性能不保证）。
// 语义与汇编版本一致：C[0:2][0:16] += A[0:2][0:k] * B[0:k][0:16]。
func Mul2x16(k int, a, b, c *float64, lda, ldb, ldc int) {
	if k <= 0 {
		return
	}
	as := unsafe.Slice(a, lda+k)              // a[1][k-1] 是最大下标
	bs := unsafe.Slice(b, (k-1)*ldb+16)       // b[k-1][15] 是最大下标
	cs := unsafe.Slice(c, ldc+16)             // c[1][15] 是最大下标
	for i := 0; i < 2; i++ {
		for j := 0; j < 16; j++ {
			sum := cs[i*ldc+j]
			for kk := 0; kk < k; kk++ {
				sum += as[i*lda+kk] * bs[kk*ldb+j]
			}
			cs[i*ldc+j] = sum
		}
	}
}
