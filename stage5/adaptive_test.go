package stage5

import (
	"math"
	"math/rand/v2"
	"testing"

	"matrixmul/stage1"
)

// TestMulMatchesStage1 与阶段 1 交叉校验。
//
// 尺寸覆盖：打包阈值两侧（767/768/769）、列带与 k 块边界（63/64/65、511/512/513）、
// 内核宽度边界（15/16/17）以及阶段 4 会塌陷的非对齐尺寸（192/384/640）。
func TestMulMatchesStage1(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	sizes := []int{
		1, 2, 3, 7, 15, 16, 17, 31, 32, 33,
		63, 64, 65, 127, 128, 129,
		192, 255, 256, 257, 383, 384, 385,
		511, 512, 513, 639, 640, 641,
		767, 768, 769, 1024,
	}
	for _, n := range sizes {
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

// TestAccumulate 验证累加语义：不清零时结果叠加。
func TestAccumulate(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	B := []float64{5, 6, 7, 8}
	C := make([]float64, 4)
	Mul(A, B, C, 2)
	Mul(A, B, C, 2)
	want := []float64{38, 44, 86, 100}
	for i := range C {
		if math.Abs(C[i]-want[i]) > 1e-12 {
			t.Fatalf("第二次累加后 C[%d] = %g, want %g", i, C[i], want[i])
		}
	}
}

// TestThresholdBranches 确认阈值两侧分别走免打包与打包路径，且都能算对。
func TestThresholdBranches(t *testing.T) {
	for _, n := range []int{packThreshold - 1, packThreshold, packThreshold + 1} {
		total := n * n
		A := make([]float64, total)
		B := make([]float64, total)
		for i := range A {
			A[i] = float64(i%7) - 3
			B[i] = float64(i%5) - 2
		}
		want := make([]float64, total)
		stage1.Mul(A, B, want, n)
		got := make([]float64, total)
		Mul(A, B, got, n)
		direct := make([]float64, total)
		mulDirect(A, B, direct, n)
		packed := make([]float64, total)
		mulPacked(A, B, packed, n)
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-9 {
				t.Fatalf("n=%d index %d: Mul got %g want %g", n, i, got[i], want[i])
			}
			if math.Abs(direct[i]-want[i]) > 1e-9 {
				t.Fatalf("n=%d index %d: mulDirect got %g want %g", n, i, direct[i], want[i])
			}
			if math.Abs(packed[i]-want[i]) > 1e-9 {
				t.Fatalf("n=%d index %d: mulPacked got %g want %g", n, i, packed[i], want[i])
			}
		}
	}
}
