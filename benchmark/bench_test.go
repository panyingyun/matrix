// bench_test.go — 独立的 Go 标准基准测试与阶段一致性测试。
//
// 运行：
//
//	go test ./benchmark -bench . -benchtime=2s -count=3
//	go test ./benchmark -run TestStagesMatchStage1
package main

import (
	"math"
	"math/rand/v2"
	"testing"
)

func benchImpl(b *testing.B, mul func(A, B, C []float64, n int)) {
	const n = 1024
	A, B := genMatrices(n)
	C := make([]float64, n*n)
	b.SetBytes(int64(2 * n * n * 8)) // A + B 的字节数
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(C)
		mul(A, B, C, n)
	}
}

// BenchmarkStage1_Naive 阶段 1：朴素三重循环（基线）。
func BenchmarkStage1_Naive(b *testing.B) { benchImpl(b, impls[0].mul) }

// BenchmarkStage2_Blocked 阶段 2：纯 Go 分块并行。
func BenchmarkStage2_Blocked(b *testing.B) { benchImpl(b, impls[1].mul) }

// BenchmarkStage3_AVX2_1D 阶段 3：AVX2/FMA 微内核 + 一维行并行。
func BenchmarkStage3_AVX2_1D(b *testing.B) { benchImpl(b, impls[2].mul) }

// BenchmarkStage4_AVX2_Packed_2D 阶段 4：AVX2/FMA + B 打包 + 固定 2D 划分。
func BenchmarkStage4_AVX2_Packed_2D(b *testing.B) { benchImpl(b, impls[3].mul) }

// BenchmarkStage5_Adaptive 阶段 5：AVX2/FMA + 自适应 2D 划分 + 按需打包（最优）。
func BenchmarkStage5_Adaptive(b *testing.B) { benchImpl(b, impls[4].mul) }

// TestStagesMatchStage1 校验其余阶段在多种尺寸下与阶段 1 一致。
// 尺寸覆盖内核宽度（16）、行带/列带（32/64）、k 块（512）、打包阈值（768）
// 以及阶段 4 会塌陷的非对齐尺寸（192/384/640）。阶段 1 自身的正确性由
// stage1 包的手工用例保证。
func TestStagesMatchStage1(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	sizes := []int{
		1, 2, 3, 7, 16, 63, 64, 65, 127, 128, 129, 256,
		192, 240, 241, 384, 512, 640, 767, 768, 769, 1024,
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
		impls[0].mul(A, B, want, n)

		for i := 1; i < len(impls); i++ {
			got := make([]float64, total)
			impls[i].mul(A, B, got, n)
			for k := range got {
				if d := math.Abs(got[k] - want[k]); d > 1e-9 {
					t.Fatalf("n=%d %s: index %d got %g want %g (diff %g)",
						n, impls[i].name, k, got[k], want[k], d)
				}
			}
		}
	}
}
