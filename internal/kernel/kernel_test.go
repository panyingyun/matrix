package kernel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestMul2x16 校验微内核在各种 k、行跨度（lda/ldb/ldc 互不相同）下与标量参考一致，
// 并确认未触及的 C 元素保持不变。
func TestMul2x16(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	cases := []struct{ k, lda, ldb, ldc int }{
		{1, 16, 16, 16},
		{2, 16, 16, 16},
		{3, 24, 20, 28},
		{16, 16, 16, 16},
		{64, 64, 64, 64},
		{257, 257, 300, 280}, // 非整齐跨度
		{512, 512, 512, 512},
		{0, 16, 16, 16}, // k=0：应保持 C 不变
	}
	for _, c := range cases {
		A := make([]float64, 2*c.lda)
		B := make([]float64, (c.k+1)*c.ldb)
		C := make([]float64, 2*c.ldc)
		for i := range A {
			A[i] = rng.Float64()
		}
		for i := range B {
			B[i] = rng.Float64()
		}
		for i := range C {
			C[i] = 0.5 // 哨兵：未被内核写入的位置应保持该值
		}

		// 标量参考
		want := make([]float64, len(C))
		copy(want, C)
		for i := 0; i < 2; i++ {
			for j := 0; j < 16; j++ {
				sum := want[i*c.ldc+j]
				for kk := 0; kk < c.k; kk++ {
					sum += A[i*c.lda+kk] * B[kk*c.ldb+j]
				}
				want[i*c.ldc+j] = sum
			}
		}

		Mul2x16(c.k, &A[0], &B[0], &C[0], c.lda, c.ldb, c.ldc)

		for i := range C {
			if d := math.Abs(C[i] - want[i]); d > 1e-12 {
				t.Fatalf("k=%d lda=%d ldb=%d ldc=%d: C[%d] = %g, want %g",
					c.k, c.lda, c.ldb, c.ldc, i, C[i], want[i])
			}
		}
	}
}
