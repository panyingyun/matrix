package stage3

import (
	"math"
	"math/rand/v2"
	"testing"

	"matrixmul/stage1"
)

// TestMulMatchesStage1 与阶段 1 交叉校验（含 1024 规模、奇数行与列尾边界）。
func TestMulMatchesStage1(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{1, 2, 3, 7, 16, 17, 63, 64, 65, 127, 128, 129, 1024} {
		total := n * n
		A := make([]float64, total)
		B := make([]float64, total)
		for i := range A {
			A[i] = rng.Float64()
			B[i] = rng.Float64()
		}
		want := make([]float64, total)
		stage1.Mul(A, B, want, n)
		got := make([]float64, total)
		Mul(A, B, got, n)
		for i := range got {
			if d := math.Abs(got[i] - want[i]); d > 1e-9 {
				t.Fatalf("n=%d index %d: got %g want %g (diff %g)", n, i, got[i], want[i], d)
			}
		}
	}
}
