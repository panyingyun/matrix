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

	"matrixmul/internal/matrixio"
)

// benchMatrices 用与 cmd/matrix generate 相同的种子生成 n×n 随机矩阵，
// 因此与 input.matrix 的第 1 组完全一致，无需读取文件即可跑基准。
func benchMatrices(n int) ([]float64, []float64) {
	rng := rand.New(rand.NewPCG(matrixio.FixedSeed, matrixio.FixedSeed^0x9e3779b97f4a7c15))
	total := n * n
	A := make([]float64, total)
	B := make([]float64, total)
	for i := range A {
		A[i] = rng.Float64()
		B[i] = rng.Float64()
	}
	return A, B
}

func benchImpl(b *testing.B, mul func(A, B, C []float64, n int)) {
	const n = 1024
	A, B := benchMatrices(n)
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

// BenchmarkStage4_AVX2_Packed_2D 阶段 4：AVX2/FMA + B 打包 + 2D 划分（最优）。
func BenchmarkStage4_AVX2_Packed_2D(b *testing.B) { benchImpl(b, impls[3].mul) }

// TestStagesMatchStage1 校验阶段 2~4 在多种尺寸（含 1024 与各任务/分块边界）
// 下与阶段 1 的结果一致。阶段 1 自身的正确性由 stage1 包的手工用例保证。
func TestStagesMatchStage1(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	sizes := []int{1, 2, 3, 7, 16, 63, 64, 65, 127, 128, 129, 256, 1024}
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
