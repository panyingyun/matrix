package stage1

import (
	"math"
	"testing"
)

// TestMulSmall 手工验证两组小矩阵。
func TestMulSmall(t *testing.T) {
	// A = [[1,2],[3,4]]，B = [[5,6],[7,8]] -> C = [[19,22],[43,50]]
	A := []float64{1, 2, 3, 4}
	B := []float64{5, 6, 7, 8}
	C := make([]float64, 4)
	Mul(A, B, C, 2)
	want := []float64{19, 22, 43, 50}
	for i := range C {
		if math.Abs(C[i]-want[i]) > 1e-12 {
			t.Fatalf("2x2: C[%d] = %g, want %g", i, C[i], want[i])
		}
	}

	// A = [[1,2,3],[4,5,6],[7,8,9]]，B = I -> C = A
	A3 := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	I3 := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	C3 := make([]float64, 9)
	Mul(A3, I3, C3, 3)
	for i := range C3 {
		if math.Abs(C3[i]-A3[i]) > 1e-12 {
			t.Fatalf("3x3*A=I: C[%d] = %g, want %g", i, C3[i], A3[i])
		}
	}
}

// TestIdentity 验证 A*I = I*A = A（n=5）。
func TestIdentity(t *testing.T) {
	const n = 5
	A := make([]float64, n*n)
	I := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			A[i*n+j] = float64(i*10 + j)
			if i == j {
				I[i*n+j] = 1
			}
		}
	}
	for _, tc := range []struct {
		name string
		X, Y []float64
	}{
		{"A*I", A, I},
		{"I*A", I, A},
	} {
		C := make([]float64, n*n)
		Mul(tc.X, tc.Y, C, n)
		for i := range C {
			if math.Abs(C[i]-A[i]) > 1e-12 {
				t.Fatalf("%s: C[%d] = %g, want %g", tc.name, i, C[i], A[i])
			}
		}
	}
}

// TestAccumulate 验证累加语义：不清零时结果会叠加。
func TestAccumulate(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	B := []float64{5, 6, 7, 8}
	C := make([]float64, 4)
	Mul(A, B, C, 2)
	Mul(A, B, C, 2) // 再次调用：C 应为 2 倍
	want := []float64{38, 44, 86, 100}
	for i := range C {
		if math.Abs(C[i]-want[i]) > 1e-12 {
			t.Fatalf("第二次累加后 C[%d] = %g, want %g", i, C[i], want[i])
		}
	}
}
