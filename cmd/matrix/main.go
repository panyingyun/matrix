// Command matrix 是 1024×1024 矩阵乘法工程的命令行工具。
//
// 用法：
//
//	matrix generate                  生成 input.matrix（3 组 1024×1024，种子 42，可复现）
//	matrix multiply [1|2|3|4]        用指定阶段计算 A*B -> result.matrix（默认 4，最优）
//	matrix verify                    用阶段 1 重算校验 result.matrix，并交叉校验全部阶段
//
// 数据文件为自定义二进制格式，见 internal/matrixio。
// 性能测试是独立程序，见 benchmark/（go run ./benchmark）。
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"time"

	"matrixmul/internal/matrixio"
	"matrixmul/stage1"
	"matrixmul/stage2"
	"matrixmul/stage3"
	"matrixmul/stage4"
)

// stage 描述一个可选的实现阶段。
type stage struct {
	id   int
	name string
	mul  func(A, B, C []float64, n int)
}

// stages 按阶段顺序列出全部实现。
var stages = []stage{
	{1, "stage1-naive", stage1.Mul},
	{2, "stage2-blocked", stage2.Mul},
	{3, "stage3-avx2-1d", stage3.Mul},
	{4, "stage4-avx2-packed-2d", stage4.Mul},
}

// verifyTolerance 是各阶段与阶段 1 交叉校验的容差（FMA 与 mul+add 的顺序差异 ~1e-13）。
const verifyTolerance = 1e-6

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "generate":
		err = cmdGenerate(os.Args[2:])
	case "multiply":
		err = cmdMultiply(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	case "help", "-h", "--help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Println(`matrix — 1024x1024 矩阵乘法（阶段 1~4）

usage:
  matrix generate  [-out input.matrix] [-dim 1024] [-sets 3] [-seed 42]
  matrix multiply  [-in input.matrix] [-out result.matrix] [-stage 4] [1|2|3|4]
  matrix verify    [-in input.matrix] [-result result.matrix]

stages:
  1  stage1-naive            朴素三重循环（基线）
  2  stage2-blocked          纯 Go 分块 + goroutine 行并行
  3  stage3-avx2-1d          AVX2/FMA 汇编微内核 + 一维行并行
  4  stage4-avx2-packed-2d   AVX2/FMA + B 打包 + 2D 行列划分（默认，最优）

性能测试是独立程序：go run ./benchmark`)
}

// cmdGenerate 生成 N 组随机矩阵并写入 input.matrix（固定种子，可复现）。
func cmdGenerate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	out := fs.String("out", matrixio.InputFile, "输出文件")
	dim := fs.Int("dim", matrixio.DefaultDim, "矩阵维数")
	sets := fs.Int("sets", matrixio.DefaultSets, "矩阵组数")
	seed := fs.Uint64("seed", matrixio.FixedSeed, "随机种子")
	if err := fs.Parse(args); err != nil {
		return err
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))
	total := *dim * *dim
	h := matrixio.NewHeader(*sets, *dim, *seed)

	w, f, err := matrixio.OpenWriter(*out)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := matrixio.WriteHeader(w, h); err != nil {
		return err
	}

	m := make([]float64, total)
	start := time.Now()
	for g := 0; g < *sets; g++ {
		for i := range m { // A
			m[i] = rng.Float64()
		}
		if err := matrixio.WriteMatrix(w, m); err != nil {
			return err
		}
		for i := range m { // B
			m[i] = rng.Float64()
		}
		if err := matrixio.WriteMatrix(w, m); err != nil {
			return err
		}
		fmt.Printf("group %d/%d generated (%dx%d, values in [0,1))\n", g+1, *sets, *dim, *dim)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("seed=%d, wrote %s (%.1f MiB) in %v\n",
		*seed, *out, float64(fileSize(*out))/(1<<20), time.Since(start))
	return nil
}

// cmdMultiply 读入 input.matrix，用指定阶段逐组计算 A*B 并写入 result.matrix。
func cmdMultiply(args []string) error {
	fs := flag.NewFlagSet("multiply", flag.ExitOnError)
	in := fs.String("in", matrixio.InputFile, "输入矩阵文件")
	out := fs.String("out", matrixio.ResultFile, "输出结果文件")
	stageID := fs.Int("stage", 4, "使用的阶段（1-4）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	// 允许位置参数形式：matrix multiply 3
	if rest := fs.Args(); len(rest) > 0 {
		var v int
		if _, err := fmt.Sscanf(rest[0], "%d", &v); err != nil {
			return fmt.Errorf("invalid stage %q (want 1-4)", rest[0])
		}
		*stageID = v
	}
	st, err := findStage(*stageID)
	if err != nil {
		return err
	}

	r, rf, err := matrixio.OpenReader(*in)
	if err != nil {
		return err
	}
	defer rf.Close()
	h, err := matrixio.ReadHeader(r)
	if err != nil {
		return fmt.Errorf("reading %s: %w", *in, err)
	}

	w, wf, err := matrixio.OpenWriter(*out)
	if err != nil {
		return err
	}
	defer wf.Close()
	if err := matrixio.WriteHeader(w, h); err != nil {
		return err
	}

	n := int(h.Dim)
	total := n * n
	A := make([]float64, total)
	B := make([]float64, total)
	C := make([]float64, total)

	totalStart := time.Now()
	for g := 0; g < int(h.Groups); g++ {
		if err := matrixio.ReadMatrix(r, A); err != nil {
			return fmt.Errorf("group %d: reading A: %w", g+1, err)
		}
		if err := matrixio.ReadMatrix(r, B); err != nil {
			return fmt.Errorf("group %d: reading B: %w", g+1, err)
		}
		clear(C) // 各阶段均为 C += A*B 的累加语义
		start := time.Now()
		st.mul(A, B, C, n)
		elapsed := time.Since(start)
		if err := matrixio.WriteMatrix(w, C); err != nil {
			return err
		}
		fmt.Printf("group %d/%d [%s]: %dx%d * %dx%d -> %dx%d  in %v (%.2f GFLOP/s)\n",
			g+1, h.Groups, st.name, n, n, n, n, n, n, elapsed,
			2*float64(total)*float64(n)/elapsed.Seconds()/1e9)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if err := wf.Close(); err != nil {
		return err
	}
	fmt.Printf("wrote %s (%.1f MiB, %d sets) in %v\n",
		*out, float64(fileSize(*out))/(1<<20), h.Groups, time.Since(totalStart))
	return nil
}

// cmdVerify 用阶段 1 重算并与 result.matrix 比对，同时交叉校验全部阶段。
func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	in := fs.String("in", matrixio.InputFile, "输入矩阵文件")
	result := fs.String("result", matrixio.ResultFile, "待校验的结果文件")
	if err := fs.Parse(args); err != nil {
		return err
	}

	r, rf, err := matrixio.OpenReader(*in)
	if err != nil {
		return err
	}
	defer rf.Close()
	h, err := matrixio.ReadHeader(r)
	if err != nil {
		return fmt.Errorf("reading %s: %w", *in, err)
	}

	rr, rrf, err := matrixio.OpenReader(*result)
	if err != nil {
		return err
	}
	defer rrf.Close()
	hr, err := matrixio.ReadHeader(rr)
	if err != nil {
		return fmt.Errorf("reading %s: %w", *result, err)
	}
	if hr.Groups != h.Groups || hr.Dim != h.Dim {
		return fmt.Errorf("header mismatch: %s has %d groups x %d, %s has %d groups x %d",
			*in, h.Groups, h.Dim, *result, hr.Groups, hr.Dim)
	}

	n := int(h.Dim)
	total := n * n
	A := make([]float64, total)
	B := make([]float64, total)
	ref := make([]float64, total)
	C := make([]float64, total)
	got := make([]float64, total)

	worst := make([]float64, len(stages)) // 各阶段相对阶段 1 的最大误差
	failed := false
	for g := 0; g < int(h.Groups); g++ {
		if err := matrixio.ReadMatrix(r, A); err != nil {
			return fmt.Errorf("group %d: reading A: %w", g+1, err)
		}
		if err := matrixio.ReadMatrix(r, B); err != nil {
			return fmt.Errorf("group %d: reading B: %w", g+1, err)
		}
		clear(ref) // 阶段 1 也是累加语义，先清零
		stages[0].mul(A, B, ref, n)

		if err := matrixio.ReadMatrix(rr, C); err != nil {
			return fmt.Errorf("group %d: reading result: %w", g+1, err)
		}
		e := maxAbsDiff(ref, C)
		worst[0] = math.Max(worst[0], e)
		status := "OK"
		if e > verifyTolerance {
			status = "MISMATCH"
			failed = true
		}
		fmt.Printf("group %d/%d: %-22s vs stage1: max abs error = %.3e  [%s]\n",
			g+1, h.Groups, *result, e, status)

		for i := 1; i < len(stages); i++ {
			clear(got)
			stages[i].mul(A, B, got, n)
			e := maxAbsDiff(ref, got)
			worst[i] = math.Max(worst[i], e)
			status := "OK"
			if e > verifyTolerance {
				status = "MISMATCH"
				failed = true
			}
			fmt.Printf("group %d/%d: %-22s vs stage1: max abs error = %.3e  [%s]\n",
				g+1, h.Groups, stages[i].name, e, status)
		}
	}

	if failed {
		return fmt.Errorf("verify FAILED: some stage exceeds tolerance %.1e", verifyTolerance)
	}
	fmt.Printf("verify passed: %d 组 × %d 个阶段全部在容差 %.1e 内一致（阶段 1 为参照）\n",
		h.Groups, len(stages), verifyTolerance)
	return nil
}

func findStage(id int) (stage, error) {
	for _, s := range stages {
		if s.id == id {
			return s, nil
		}
	}
	return stage{}, fmt.Errorf("unknown stage %d (want 1-4)", id)
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return -1
	}
	return fi.Size()
}
