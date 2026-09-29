// Command benchmark 是独立的性能测试程序。
//
// 两种模式：
//
//  1. 文件模式（默认）：在同一份 input.matrix 上横向对比阶段 1~5 的耗时、
//     GFLOP/s、相对阶段 1 的加速比与正确性误差；
//  2. 尺寸扫描模式（-sizes）：自己生成固定种子的随机矩阵，扫描一组维数，
//     用"批量重复计时"消除小尺寸下的时钟量化误差（详见 timeBatched）。
//
// 用法：
//
//	go run ./benchmark                          # 默认读 input.matrix，各阶段每组件测 1 次
//	go run ./benchmark -in input.matrix -repeat 3
//	go run ./benchmark -sizes all               # 预设尺寸集 16…2048
//	go run ./benchmark -sizes 64,128,384,1024   # 自定义尺寸
//
// Go 标准基准测试见 bench_test.go：
//
//	go test ./benchmark -bench . -benchtime=2s -count=3
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"matrixmul/internal/matrixio"
	"matrixmul/stage1"
	"matrixmul/stage2"
	"matrixmul/stage3"
	"matrixmul/stage4"
	"matrixmul/stage5"
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
	{"stage5-adaptive", stage5.Mul},
}

// tolerance 与 cmd/matrix verify 保持一致。
const tolerance = 1e-6

// sweepPreset 是 -sizes all 使用的预设尺寸集：
// 覆盖内核宽度（16）、分块/行带（32/64）、k 块（512）、打包阈值（768）
// 以及阶段 4 会塌陷的非对齐尺寸（192/384/640）。
var sweepPreset = []int{16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 2048}

func main() {
	in := flag.String("in", matrixio.InputFile, "输入矩阵文件（文件模式）")
	repeat := flag.Int("repeat", 1, "重复测量次数（取最快一次；尺寸模式下为计时轮数）")
	sizes := flag.String("sizes", "", "尺寸扫描模式：逗号分隔的维数，或 all（预设 "+sizesHint()+"）")
	flag.Parse()

	var err error
	if *sizes != "" {
		err = runSizes(*sizes, *repeat)
	} else {
		err = run(*in, *repeat)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func sizesHint() string {
	parts := make([]string, len(sweepPreset))
	for i, n := range sweepPreset {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ",")
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
	fmt.Printf("correctness: 阶段 2~%d 与阶段 1 的最大误差均在容差 %.1e 内（pass）\n", len(impls), tolerance)
	return nil
}

// parseSizes 解析 -sizes：all（预设尺寸集）或逗号分隔的维数列表。
func parseSizes(spec string) ([]int, error) {
	if spec == "all" {
		return append([]int(nil), sweepPreset...), nil
	}
	var sizes []int
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid size %q", part)
		}
		if n < 1 || n > 4096 {
			return nil, fmt.Errorf("size %d out of range [1, 4096]", n)
		}
		sizes = append(sizes, n)
	}
	if len(sizes) == 0 {
		return nil, fmt.Errorf("-sizes: no size given")
	}
	return sizes, nil
}

// genMatrices 用与 cmd/matrix generate 相同的方式生成 n×n 随机矩阵对：
// 同一种子（matrixio.FixedSeed），先填满 A 再填满 B，因此与 generate 写出的
// 第 1 组数据一致，无需读文件。
func genMatrices(n int) (A, B []float64) {
	rng := rand.New(rand.NewPCG(matrixio.FixedSeed, matrixio.FixedSeed^0x9e3779b97f4a7c15))
	A = make([]float64, n*n)
	B = make([]float64, n*n)
	for i := range A {
		A[i] = rng.Float64()
	}
	for i := range B {
		B[i] = rng.Float64()
	}
	return A, B
}

// timeBatched 返回 mul 在给定矩阵上的平均单次耗时，用于尺寸扫描模式。
//
// 小尺寸下单次调用只有几微秒，而 Windows 单调时钟存在量化误差（实测约 0.5ms），
// 单次计时会把 0.02ms 与 0.0001ms 测成同一个数。因此这里先标定内部重复次数 inner
// （使单轮计时跨度 ≥ 25ms），再取 rounds 轮批量计时的最快值，并单独测量 clear(C)
// 的成本予以扣除（清零不计入乘法耗时）。
func timeBatched(mul func(A, B, C []float64, n int), A, B, C []float64, n int, rounds int) time.Duration {
	const target = 25 * time.Millisecond
	inner := 1
	for {
		st := time.Now()
		for i := 0; i < inner; i++ {
			clear(C)
			mul(A, B, C, n)
		}
		if time.Since(st) >= target || inner >= 1<<20 {
			break
		}
		inner *= 2
	}
	if rounds < 1 {
		rounds = 1
	}
	bestClear, bestBoth := time.Duration(math.MaxInt64), time.Duration(math.MaxInt64)
	for r := 0; r < rounds; r++ {
		st := time.Now()
		for i := 0; i < inner; i++ {
			clear(C)
		}
		if d := time.Since(st); d < bestClear {
			bestClear = d
		}
		st = time.Now()
		for i := 0; i < inner; i++ {
			clear(C)
			mul(A, B, C, n)
		}
		if d := time.Since(st); d < bestBoth {
			bestBoth = d
		}
	}
	if bestBoth < bestClear {
		bestBoth = bestClear
	}
	return time.Duration((int64(bestBoth) - int64(bestClear)) / int64(inner))
}

// runSizes 是尺寸扫描模式：逐尺寸测量各阶段耗时、吞吐与相对阶段 1 的加速比，
// 并顺带校验各阶段结果与阶段 1 一致。
func runSizes(spec string, repeat int) error {
	sizes, err := parseSizes(spec)
	if err != nil {
		return err
	}
	if repeat < 1 {
		repeat = 1
	}
	fmt.Printf("size sweep: %d 个尺寸, GOMAXPROCS=%d, 每阶段 %d 轮批量计时（已扣除 clear 开销）\n",
		len(sizes), runtime.GOMAXPROCS(0), repeat)

	var hdr strings.Builder
	fmt.Fprintf(&hdr, "%6s |", "n")
	for _, im := range impls {
		fmt.Fprintf(&hdr, " %11s", shortName(im.name))
	}
	fmt.Fprintf(&hdr, " |")
	for i := 1; i < len(impls); i++ {
		fmt.Fprintf(&hdr, " %6s", fmt.Sprintf("s%dx", i+1))
	}
	fmt.Fprintf(&hdr, " | %9s", "GF/s 最优")
	fmt.Println(hdr.String())

	worstErr := 0.0
	for _, n := range sizes {
		A, B := genMatrices(n)
		total := n * n
		C := make([]float64, total)
		d := make([]time.Duration, len(impls))
		for i, im := range impls {
			d[i] = timeBatched(im.mul, A, B, C, n, repeat)
		}

		// 正确性：阶段 1 为参照，make 出来的切片已清零（累加语义）
		ref := make([]float64, total)
		impls[0].mul(A, B, ref, n)
		for i := 1; i < len(impls); i++ {
			got := make([]float64, total)
			impls[i].mul(A, B, got, n)
			if e := maxAbsDiff(ref, got); e > worstErr {
				worstErr = e
			}
		}

		flops := 2.0 * float64(total) * float64(n)
		var row strings.Builder
		fmt.Fprintf(&row, "%6d |", n)
		for _, x := range d {
			fmt.Fprintf(&row, " %8.4f ms", ms(x))
		}
		fmt.Fprintf(&row, " |")
		for i := 1; i < len(d); i++ {
			fmt.Fprintf(&row, " %5.2fx", float64(d[0])/float64(d[i]))
		}
		fmt.Fprintf(&row, " | %9.1f", gf(flops, d[len(d)-1]))
		fmt.Println(row.String())
	}

	if worstErr > tolerance {
		return fmt.Errorf("sweep correctness FAILED: max err %.3e > %.1e", worstErr, tolerance)
	}
	fmt.Printf("correctness: 阶段 2~%d 与阶段 1 的最大误差 %.3e（容差 %.1e，pass）\n",
		len(impls), worstErr, tolerance)
	return nil
}

// shortName 把 stage3-avx2-1d 压成 stage3 以便表头对齐。
func shortName(name string) string {
	if i := strings.IndexByte(name, '-'); i > 0 {
		return name[:i]
	}
	return name
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
