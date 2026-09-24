// Command benchmark 是独立的性能测试程序：在同一份 input.matrix 上
// 横向对比阶段 1~4 的耗时、GFLOP/s、相对阶段 1 的加速比与正确性误差。
//
// 用法：
//
//	go run ./benchmark                      # 默认读 input.matrix，各阶段每组件测 1 次
//	go run ./benchmark -in input.matrix -repeat 3
//
// Go 标准基准测试见 bench_test.go：
//
//	go test ./benchmark -bench . -benchtime=2s -count=3
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"matrixmul/internal/matrixio"
	"matrixmul/stage1"
	"matrixmul/stage2"
	"matrixmul/stage3"
	"matrixmul/stage4"
)

// impl 是一个待测实现。
type impl struct {
	name string
	mul  func(A, B, C []float64, n int)
}

// impls 按阶段顺序列出全部实现，impls[0]（阶段 1）同时作为正确性参照。
var impls = []impl{
	{"stage1-naive", stage1.Mul},
	{"stage2-blocked", stage2.Mul},
	{"stage3-avx2-1d", stage3.Mul},
	{"stage4-avx2-packed-2d", stage4.Mul},
}

// tolerance 与 cmd/matrix verify 保持一致。
const tolerance = 1e-6

func main() {
	in := flag.String("in", matrixio.InputFile, "输入矩阵文件")
	repeat := flag.Int("repeat", 1, "每个阶段每组重复测量次数（取最快一次）")
	flag.Parse()

	if err := run(*in, *repeat); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(path string, repeat int) error {
	r, f, err := matrixio.OpenReader(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h, err := matrixio.ReadHeader(r)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	if repeat < 1 {
		repeat = 1
	}

	n := int(h.Dim)
	total := n * n
	flopsPerSet := 2.0 * float64(total) * float64(n) // 乘 + 加各一次

	A := make([]float64, total)
	B := make([]float64, total)
	C := make([]float64, total)
	ref := make([]float64, total)

	fmt.Printf("benchmark: %d set(s) of %dx%d from %s, GOMAXPROCS=%d, repeat=%d\n",
		h.Groups, n, n, path, runtime.GOMAXPROCS(0), repeat)
	fmt.Printf("%-24s %10s %10s %10s %10s\n", "stage", "per set", "GF/s", "speedup", "max err")

	sums := make([]time.Duration, len(impls))
	errs := make([]float64, len(impls))

	for g := 0; g < int(h.Groups); g++ {
		if err := matrixio.ReadMatrix(r, A); err != nil {
			return fmt.Errorf("group %d: reading A: %w", g+1, err)
		}
		if err := matrixio.ReadMatrix(r, B); err != nil {
			return fmt.Errorf("group %d: reading B: %w", g+1, err)
		}
		fmt.Printf("--- set %d/%d ---\n", g+1, h.Groups)

		var base time.Duration
		for i, im := range impls {
			best := time.Duration(math.MaxInt64)
			for rep := 0; rep < repeat; rep++ {
				clear(C)
				start := time.Now()
				im.mul(A, B, C, n)
				if d := time.Since(start); d < best {
					best = d
				}
			}
			sums[i] += best
			if i == 0 {
				base = best
				copy(ref, C)
			} else {
				if e := maxAbsDiff(ref, C); e > errs[i] {
					errs[i] = e
				}
			}
			e := errs[i]
			speed := float64(base) / float64(best)
			if i == 0 {
				fmt.Printf("%-24s %8.2f ms %10.1f %9s %10s\n",
					im.name, ms(best), gf(flopsPerSet, best), "1.00x", "-")
			} else {
				fmt.Printf("%-24s %8.2f ms %10.1f %9.2fx %10.2e\n",
					im.name, ms(best), gf(flopsPerSet, best), speed, e)
			}
		}
	}

	fmt.Printf("--- total (%d sets) ---\n", h.Groups)
	for i, im := range impls {
		if i == 0 {
			fmt.Printf("%-24s %8.2f ms %10.1f %9s %10s\n",
				im.name, ms(sums[i]), gf(flopsPerSet*float64(h.Groups), sums[i]), "1.00x", "-")
			continue
		}
		fmt.Printf("%-24s %8.2f ms %10.1f %9.2fx %10.2e\n",
			im.name, ms(sums[i]), gf(flopsPerSet*float64(h.Groups), sums[i]),
			float64(sums[0])/float64(sums[i]), errs[i])
	}

	for i := 1; i < len(impls); i++ {
		if errs[i] > tolerance {
			return fmt.Errorf("%s FAILED correctness: max err %.3e > %.1e", impls[i].name, errs[i], tolerance)
		}
	}
	fmt.Printf("correctness: 阶段 2~4 与阶段 1 的最大误差均在容差 %.1e 内（pass）\n", tolerance)
	return nil
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func gf(flops float64, d time.Duration) float64 { return flops / d.Seconds() / 1e9 }

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}
