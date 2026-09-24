package matrixio

import (
	"math/rand/v2"
	"path/filepath"
	"testing"
)

// TestRoundTrip 校验文件头与矩阵数据的写入/读回一致。
func TestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.matrix")
	h := NewHeader(2, 32, 12345)
	const n = 32
	rng := rand.New(rand.NewPCG(7, 8))
	A := make([]float64, n*n)
	B := make([]float64, n*n)
	for i := range A {
		A[i] = rng.Float64()
		B[i] = rng.Float64()
	}

	w, f, err := OpenWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteHeader(w, h); err != nil {
		t.Fatal(err)
	}
	if err := WriteMatrix(w, A); err != nil {
		t.Fatal(err)
	}
	if err := WriteMatrix(w, B); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	r, rf, err := OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()
	h2, err := ReadHeader(r)
	if err != nil {
		t.Fatal(err)
	}
	if h2.Groups != 2 || h2.Dim != 32 || h2.Seed != 12345 || h2.Ver != FileVersion {
		t.Fatalf("header mismatch: %+v", h2)
	}
	A2 := make([]float64, n*n)
	B2 := make([]float64, n*n)
	if err := ReadMatrix(r, A2); err != nil {
		t.Fatal(err)
	}
	if err := ReadMatrix(r, B2); err != nil {
		t.Fatal(err)
	}
	for i := range A {
		if A[i] != A2[i] || B[i] != B2[i] {
			t.Fatalf("round-trip mismatch at %d: A %g/%g B %g/%g", i, A[i], A2[i], B[i], B2[i])
		}
	}
}

// TestBadMagic 校验非本格式文件会被拒绝。
func TestBadMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.matrix")
	w, f, err := OpenWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("NOPE")); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	r, rf, err := OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()
	if _, err := ReadHeader(r); err == nil {
		t.Fatal("expected error for bad magic")
	}
}
