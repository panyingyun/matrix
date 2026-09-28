// 端到端测试：构建 cmd/matrix，用真实子进程跑 generate / multiply / verify。
//
//	go test ./e2e
//
// 用例 1 使用已提交的黄金文件 testdata/case01（3 组 1024×1024，种子 42）。
// 用例 2–11 使用 testdata/case02–case11 的 input.matrix / result.matrix。
// 其余用例在临时目录生成小矩阵，覆盖精确结果、分块边界、命令行契约和失败路径。
package e2e_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"matrixmul/internal/matrixio"
)

const (
	headerBytes = 28 // 魔数 8 + 版本/组数/维数各 4 + 种子 8
	verifyTol   = 1e-6
)

var binPath string

func TestMain(m *testing.M) {
	td, err := os.MkdirTemp("", "matrix-e2e-bin-*")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	name := "matrix"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binPath = filepath.Join(td, name)

	root, err := filepath.Abs("..")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cmd := exec.Command("go", "build", "-o", binPath, "./cmd/matrix")
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "build cmd/matrix: %v\n%s", err, out)
		os.RemoveAll(td)
		os.Exit(1)
	}

	code := m.Run()
	os.RemoveAll(td)
	os.Exit(code)
}

// TestCase01_Golden1024 用例 1：仓库内 3 组 1024×1024、种子 42 的黄金输入与结果。
func TestCase01_Golden1024(t *testing.T) {
	in := absPath(t, filepath.Join("testdata", "case01", "input.matrix"))
	result := absPath(t, filepath.Join("testdata", "case01", "result.matrix"))

	inH := readHeader(t, in)
	outH := readHeader(t, result)
	if inH.Groups != matrixio.DefaultSets || inH.Dim != matrixio.DefaultDim || inH.Seed != matrixio.FixedSeed || inH.Ver != matrixio.FileVersion {
		t.Fatalf("input header = %+v, want groups=%d dim=%d seed=%d ver=%d",
			inH, matrixio.DefaultSets, matrixio.DefaultDim, matrixio.FixedSeed, matrixio.FileVersion)
	}
	if outH.Groups != inH.Groups || outH.Dim != inH.Dim || outH.Seed != inH.Seed || outH.Ver != inH.Ver {
		t.Fatalf("result header = %+v, want match input %+v", outH, inH)
	}
	assertSize(t, in, fileBytes(int(inH.Dim), int(inH.Groups)*2))
	assertSize(t, result, fileBytes(int(outH.Dim), int(outH.Groups)))

	t.Run("verify", func(t *testing.T) {
		stdout := verifyOK(t, in, result)
		if !strings.Contains(stdout, "3 组") {
			t.Fatalf("verify stdout missing group count:\n%s", stdout)
		}
	})

	t.Run("generate_matches_input", func(t *testing.T) {
		dir := t.TempDir()
		got := filepath.Join(dir, "input.matrix")
		mustRun(t, dir, "generate",
			"-out", got,
			"-dim", strconv.Itoa(matrixio.DefaultDim),
			"-sets", strconv.Itoa(matrixio.DefaultSets),
			"-seed", strconv.FormatUint(matrixio.FixedSeed, 10),
		)
		if sha256File(t, got) != sha256File(t, in) {
			t.Fatalf("generate -dim %d -sets %d -seed %d != testdata/case01/input.matrix",
				matrixio.DefaultDim, matrixio.DefaultSets, matrixio.FixedSeed)
		}
	})

	t.Run("multiply_all_stages", func(t *testing.T) {
		_, want := loadMatrices(t, result, 1)
		dir := t.TempDir()
		for stage := 1; stage <= 4; stage++ {
			out := filepath.Join(dir, fmt.Sprintf("stage%d.matrix", stage))
			mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", strconv.Itoa(stage))
			h, got := loadMatrices(t, out, 1)
			if h.Groups != outH.Groups || h.Dim != outH.Dim || h.Seed != outH.Seed {
				t.Fatalf("stage %d header %+v, want groups=%d dim=%d seed=%d",
					stage, h, outH.Groups, outH.Dim, outH.Seed)
			}
			assertMatricesClose(t, fmt.Sprintf("stage %d", stage), got, want, verifyTol)
		}
	})
}

// fixtureCases 是 testdata/case02–case11。每组含 input.matrix 与 result.matrix
// （generate 固定种子，result 由 multiply -stage 4 写出）。
var fixtureCases = []struct {
	name      string
	dim, sets int
	seed      uint64
}{
	{"case02", 1, 1, 1},
	{"case03", 2, 1, 2},
	{"case04", 3, 2, 3},
	{"case05", 7, 1, 7},
	{"case06", 16, 2, 16},
	{"case07", 17, 1, 17},
	{"case08", 63, 1, 63},
	{"case09", 64, 2, 64},
	{"case10", 65, 1, 65},
	{"case11", 128, 1, 128},
}

// TestFixtureCases 用例 2–11：已落盘的矩阵文件可复现，且与阶段 4 结果一致。
func TestFixtureCases(t *testing.T) {
	for _, tc := range fixtureCases {
		t.Run(tc.name, func(t *testing.T) {
			in := absPath(t, filepath.Join("testdata", tc.name, "input.matrix"))
			result := absPath(t, filepath.Join("testdata", tc.name, "result.matrix"))

			h := readHeader(t, in)
			if int(h.Dim) != tc.dim || int(h.Groups) != tc.sets || h.Seed != tc.seed || h.Ver != matrixio.FileVersion {
				t.Fatalf("input header = %+v, want dim=%d sets=%d seed=%d", h, tc.dim, tc.sets, tc.seed)
			}
			rh := readHeader(t, result)
			if rh.Groups != h.Groups || rh.Dim != h.Dim || rh.Seed != h.Seed || rh.Ver != h.Ver {
				t.Fatalf("result header = %+v, want match input %+v", rh, h)
			}
			assertSize(t, in, fileBytes(tc.dim, tc.sets*2))
			assertSize(t, result, fileBytes(tc.dim, tc.sets))

			dir := t.TempDir()
			regen := filepath.Join(dir, "input.matrix")
			mustRun(t, dir, "generate",
				"-out", regen,
				"-dim", strconv.Itoa(tc.dim),
				"-sets", strconv.Itoa(tc.sets),
				"-seed", strconv.FormatUint(tc.seed, 10),
			)
			if sha256File(t, regen) != sha256File(t, in) {
				t.Fatalf("%s: generate 结果与 input.matrix 不一致", tc.name)
			}

			out := filepath.Join(dir, "result.matrix")
			mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", "4")
			if sha256File(t, out) != sha256File(t, result) {
				t.Fatalf("%s: stage 4 结果与 result.matrix 不一致", tc.name)
			}

			stdout := verifyOK(t, in, result)
			if !strings.Contains(stdout, fmt.Sprintf("%d 组", tc.sets)) {
				t.Fatalf("%s verify stdout:\n%s", tc.name, stdout)
			}
		})
	}
}

// TestCase02_Exact2x2 用例 2：手工 2×2，四个阶段都必须得到精确整数结果。
func TestCase02_Exact2x2(t *testing.T) {
	t.Run("positive", func(t *testing.T) {
		// [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
		dir := t.TempDir()
		in := writeInput(t, filepath.Join(dir, "input.matrix"), 2, 1, 0, [][]float64{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		})
		assertStagesExact(t, in, [][]float64{{19, 22, 43, 50}})
	})
	t.Run("negative", func(t *testing.T) {
		// [[-1,2],[3,-4]] * [[5,-6],[-7,8]] = [[-19,22],[43,-50]]
		dir := t.TempDir()
		in := writeInput(t, filepath.Join(dir, "input.matrix"), 2, 1, 0, [][]float64{
			{-1, 2, 3, -4},
			{5, -6, -7, 8},
		})
		assertStagesExact(t, in, [][]float64{{-19, 22, 43, -50}})
	})
}

// TestCase03_Identity 用例 3：单位矩阵。n=17 带 AVX 列尾，n=64 对齐分块。
func TestCase03_Identity(t *testing.T) {
	for _, n := range []int{1, 17, 64} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			A := sequential(n)
			I := identity(n)
			dir := t.TempDir()
			inAI := writeInput(t, filepath.Join(dir, "ai.matrix"), n, 1, 1, [][]float64{A, I})
			inIA := writeInput(t, filepath.Join(dir, "ia.matrix"), n, 1, 1, [][]float64{I, A})
			assertStagesExact(t, inAI, [][]float64{A})
			assertStagesExact(t, inIA, [][]float64{A})
		})
	}
}

// TestCase04_ShapeBoundaries 用例 4：1、分块 64、微内核 16 附近的维数，经 generate→multiply→verify。
func TestCase04_ShapeBoundaries(t *testing.T) {
	cases := []struct {
		dim, sets int
		seed      uint64
	}{
		{1, 1, 1},
		{1, 3, 42},
		{2, 2, 7},
		{3, 1, 3},
		{7, 2, 11},
		{8, 1, 8},
		{15, 1, 15},
		{16, 2, 16},
		{17, 1, 17},
		{31, 1, 31},
		{32, 2, 32},
		{63, 1, 63},
		{64, 3, 42},
		{65, 2, 65},
		{127, 1, 127},
		{128, 2, 128},
		{129, 1, 129},
		{255, 1, 255},
		{256, 1, 256},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(fmt.Sprintf("dim=%d_sets=%d", tc.dim, tc.sets), func(t *testing.T) {
			generateMultiplyVerify(t, tc.dim, tc.sets, tc.seed, 4)
		})
	}
}

// TestCase05_MultiGroupReproducible 用例 5：多组小矩阵两次 generate 字节相同，且四个阶段互相一致。
func TestCase05_MultiGroupReproducible(t *testing.T) {
	const dim, sets = 48, 4
	const seed = uint64(99)
	dir := t.TempDir()
	a := filepath.Join(dir, "a.matrix")
	b := filepath.Join(dir, "b.matrix")
	gen := func(out string) {
		t.Helper()
		stdout := mustRun(t, dir, "generate",
			"-out", out, "-dim", strconv.Itoa(dim), "-sets", strconv.Itoa(sets), "-seed", strconv.FormatUint(seed, 10))
		if !strings.Contains(stdout, "seed=99") {
			t.Fatalf("generate stdout missing seed:\n%s", stdout)
		}
	}
	gen(a)
	gen(b)
	if sha256File(t, a) != sha256File(t, b) {
		t.Fatal("same seed produced different input files")
	}

	h, mats := loadMatrices(t, a, 2)
	if h.Groups != sets || h.Dim != dim || h.Seed != seed {
		t.Fatalf("header %+v", h)
	}
	if len(mats) != sets*2 {
		t.Fatalf("matrix count %d", len(mats))
	}

	outs := multiplyAllStages(t, a, dir)
	baseH, base := loadMatrices(t, outs[1], 1)
	if baseH.Seed != seed || baseH.Groups != sets || baseH.Dim != dim {
		t.Fatalf("stage1 result header %+v", baseH)
	}
	if len(base) != sets {
		t.Fatalf("stage1 groups %d", len(base))
	}
	for stage := 2; stage <= 4; stage++ {
		_, got := loadMatrices(t, outs[stage], 1)
		assertMatricesClose(t, fmt.Sprintf("stage %d vs stage 1", stage), got, base, verifyTol)
	}
	stdout := verifyOK(t, a, outs[4])
	if !strings.Contains(stdout, "4 组") {
		t.Fatalf("verify stdout:\n%s", stdout)
	}
}

// TestCase06_ZeroAndOnes 用例 6：全零与全一。全一时 C 的每个元素都是 n。
func TestCase06_ZeroAndOnes(t *testing.T) {
	for _, n := range []int{1, 9, 64} {
		t.Run(fmt.Sprintf("zero_n=%d", n), func(t *testing.T) {
			z := make([]float64, n*n)
			dir := t.TempDir()
			in := writeInput(t, filepath.Join(dir, "input.matrix"), n, 2, 5, [][]float64{z, z, z, z})
			assertStagesExact(t, in, [][]float64{z, z})
		})
		t.Run(fmt.Sprintf("ones_n=%d", n), func(t *testing.T) {
			ones := filled(n, 1)
			want := filled(n, float64(n))
			dir := t.TempDir()
			in := writeInput(t, filepath.Join(dir, "input.matrix"), n, 1, 6, [][]float64{ones, ones})
			assertStagesExact(t, in, [][]float64{want})
		})
	}
}

// TestCase07_CLIContract 用例 7：用法、非法阶段、坏文件。
func TestCase07_CLIContract(t *testing.T) {
	dir := t.TempDir()

	t.Run("no_args", func(t *testing.T) {
		stdout, _, code := run(t, dir)
		if code != 2 || !strings.Contains(stdout, "usage:") {
			t.Fatalf("code=%d stdout=%s", code, stdout)
		}
	})
	for _, arg := range []string{"help", "-h", "--help"} {
		arg := arg
		t.Run(arg, func(t *testing.T) {
			stdout, stderr, code := run(t, dir, arg)
			if code != 0 || !strings.Contains(stdout, "usage:") || stderr != "" {
				t.Fatalf("code=%d stderr=%q stdout=%s", code, stderr, stdout)
			}
		})
	}
	t.Run("unknown_command", func(t *testing.T) {
		stdout, _, code := run(t, dir, "nope")
		if code != 2 || !strings.Contains(stdout, "usage:") {
			t.Fatalf("code=%d stdout=%s", code, stdout)
		}
	})
	t.Run("bad_flag", func(t *testing.T) {
		_, stderr, code := run(t, dir, "generate", "-dim")
		if code != 2 || !strings.Contains(stderr, "flag") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	for _, stage := range []string{"0", "5", "99", "abc"} {
		stage := stage
		t.Run("stage_"+stage, func(t *testing.T) {
			_, stderr, code := run(t, dir, "multiply", stage, "-in", "x", "-out", "y")
			if code != 1 || !strings.Contains(stderr, "error:") {
				t.Fatalf("stage %s: code=%d stderr=%s", stage, code, stderr)
			}
		})
	}
	t.Run("stage_negative", func(t *testing.T) {
		_, stderr, code := run(t, dir, "multiply", "--", "-1")
		if code != 1 || !strings.Contains(stderr, "unknown stage") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("missing_default_input", func(t *testing.T) {
		_, stderr, code := run(t, dir, "multiply")
		if code != 1 || !strings.Contains(stderr, "error:") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("bad_magic", func(t *testing.T) {
		path := filepath.Join(dir, "bad.matrix")
		writeRaw(t, path, []byte("NOTAMATRIX_padding_bytes!!!!"))
		_, stderr, code := run(t, dir, "multiply", "-in", path, "-out", filepath.Join(dir, "o.matrix"))
		if code != 1 || !strings.Contains(stderr, "bad magic") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("bad_version", func(t *testing.T) {
		path := filepath.Join(dir, "ver.matrix")
		h := matrixio.NewHeader(1, 4, 1)
		h.Ver = 99
		var buf bytes.Buffer
		if err := binary.Write(&buf, binary.LittleEndian, &h); err != nil {
			t.Fatal(err)
		}
		writeRaw(t, path, buf.Bytes())
		_, stderr, code := run(t, dir, "multiply", "-in", path, "-out", filepath.Join(dir, "o.matrix"))
		if code != 1 || !strings.Contains(stderr, "unsupported file version") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("truncated", func(t *testing.T) {
		path := filepath.Join(dir, "short.matrix")
		writeRaw(t, path, []byte("MX"))
		_, stderr, code := run(t, dir, "verify", "-in", path, "-result", path)
		if code != 1 || !strings.Contains(stderr, "error:") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("header_only", func(t *testing.T) {
		path := filepath.Join(dir, "hdr.matrix")
		in := writeInput(t, filepath.Join(dir, "full.matrix"), 4, 1, 1, [][]float64{
			filled(4, 1),
			filled(4, 1),
		})
		f, err := os.Open(in)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		hdr := make([]byte, headerBytes)
		if _, err := io.ReadFull(f, hdr); err != nil {
			t.Fatal(err)
		}
		writeRaw(t, path, hdr)
		_, stderr, code := run(t, dir, "multiply", "-in", path, "-out", filepath.Join(dir, "o.matrix"))
		if code != 1 || !strings.Contains(stderr, "error:") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
}

// TestCase08_VerifyRejectsBadResult 用例 8：结果被改、维度不一致、或把输入当成结果时，verify 必须失败。
func TestCase08_VerifyRejectsBadResult(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "input.matrix")
	out := filepath.Join(dir, "result.matrix")
	mustRun(t, dir, "generate", "-out", in, "-dim", "8", "-sets", "2", "-seed", "3")
	mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", "1")
	verifyOK(t, in, out)

	t.Run("tampered_value", func(t *testing.T) {
		bad := filepath.Join(dir, "tampered.matrix")
		copyFile(t, out, bad)
		flipFirstValue(t, bad)
		stdout, stderr, code := run(t, dir, "verify", "-in", in, "-result", bad)
		if code != 1 || !strings.Contains(stdout, "MISMATCH") || !strings.Contains(stderr, "verify FAILED") {
			t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr, stdout)
		}
	})
	t.Run("input_as_result", func(t *testing.T) {
		stdout, stderr, code := run(t, dir, "verify", "-in", in, "-result", in)
		if code != 1 || !strings.Contains(stdout, "MISMATCH") || !strings.Contains(stderr, "verify FAILED") {
			t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr, stdout)
		}
	})
	t.Run("dim_mismatch", func(t *testing.T) {
		other := filepath.Join(dir, "dim16.matrix")
		mustRun(t, dir, "generate", "-out", other, "-dim", "16", "-sets", "2", "-seed", "3")
		_, stderr, code := run(t, dir, "verify", "-in", in, "-result", other)
		if code != 1 || !strings.Contains(stderr, "header mismatch") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
	t.Run("groups_mismatch", func(t *testing.T) {
		other := filepath.Join(dir, "sets1.matrix")
		mustRun(t, dir, "generate", "-out", other, "-dim", "8", "-sets", "1", "-seed", "3")
		res := filepath.Join(dir, "sets1-result.matrix")
		mustRun(t, dir, "multiply", "-in", other, "-out", res, "-stage", "4")
		_, stderr, code := run(t, dir, "verify", "-in", in, "-result", res)
		if code != 1 || !strings.Contains(stderr, "header mismatch") {
			t.Fatalf("code=%d stderr=%s", code, stderr)
		}
	})
}

// TestCase09_StageSelection 用例 9：位置参数、-stage、重复运行的确定性。
func TestCase09_StageSelection(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "input.matrix")
	mustRun(t, dir, "generate", "-out", in, "-dim", "65", "-sets", "2", "-seed", "65")

	for stage := 1; stage <= 4; stage++ {
		stage := stage
		t.Run(fmt.Sprintf("stage_%d", stage), func(t *testing.T) {
			byFlag := filepath.Join(dir, fmt.Sprintf("flag%d.matrix", stage))
			byPos := filepath.Join(dir, fmt.Sprintf("pos%d.matrix", stage))
			again := filepath.Join(dir, fmt.Sprintf("again%d.matrix", stage))
			mustRun(t, dir, "multiply", "-in", in, "-out", byFlag, "-stage", strconv.Itoa(stage))
			stdout := mustRun(t, dir, "multiply", "-in", in, "-out", byPos, strconv.Itoa(stage))
			if !strings.Contains(stdout, "stage"+strconv.Itoa(stage)) {
				t.Fatalf("stdout missing stage name:\n%s", stdout)
			}
			mustRun(t, dir, "multiply", "-in", in, "-out", again, "-stage="+strconv.Itoa(stage))
			if sha256File(t, byFlag) != sha256File(t, byPos) || sha256File(t, byFlag) != sha256File(t, again) {
				t.Fatalf("stage %d outputs differ across flag, positional arg, and rerun", stage)
			}
			verifyOK(t, in, byFlag)
		})
	}

	t.Run("positional_overrides_flag", func(t *testing.T) {
		out := filepath.Join(dir, "override.matrix")
		stage1 := filepath.Join(dir, "only1.matrix")
		mustRun(t, dir, "multiply", "-stage", "4", "-in", in, "-out", out, "1")
		mustRun(t, dir, "multiply", "-in", in, "-out", stage1, "-stage", "1")
		if sha256File(t, out) != sha256File(t, stage1) {
			t.Fatal("positional stage did not override -stage")
		}
	})
}

// TestCase10_DefaultFilenames 用例 10：在工作目录使用默认的 input.matrix / result.matrix。
func TestCase10_DefaultFilenames(t *testing.T) {
	dir := t.TempDir()
	stdout := mustRun(t, dir, "generate", "-dim", "12", "-sets", "2", "-seed", "12")
	if !strings.Contains(stdout, "input.matrix") {
		t.Fatalf("stdout:\n%s", stdout)
	}
	stdout = mustRun(t, dir, "multiply", "2")
	if !strings.Contains(stdout, "result.matrix") || !strings.Contains(stdout, "stage2-blocked") {
		t.Fatalf("stdout:\n%s", stdout)
	}
	verifyOK(t, filepath.Join(dir, "input.matrix"), filepath.Join(dir, "result.matrix"))
	if _, err := os.Stat(filepath.Join(dir, matrixio.InputFile)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, matrixio.ResultFile)); err != nil {
		t.Fatal(err)
	}
}

// TestCase11_EmptyShapes 用例 11：0 组与 0 维只写文件头，命令仍成功且不产出矩阵元素。
func TestCase11_EmptyShapes(t *testing.T) {
	for _, tc := range []struct {
		dim, sets int
	}{
		{0, 0},
		{0, 1},
		{8, 0},
	} {
		tc := tc
		t.Run(fmt.Sprintf("dim=%d_sets=%d", tc.dim, tc.sets), func(t *testing.T) {
			dir := t.TempDir()
			in := filepath.Join(dir, "input.matrix")
			out := filepath.Join(dir, "result.matrix")
			mustRun(t, dir, "generate", "-out", in, "-dim", strconv.Itoa(tc.dim), "-sets", strconv.Itoa(tc.sets), "-seed", "1")
			mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", "4")
			stdout := verifyOK(t, in, out)
			if !strings.Contains(stdout, fmt.Sprintf("%d 组", tc.sets)) {
				t.Fatalf("verify stdout:\n%s", stdout)
			}
			assertSize(t, in, fileBytes(tc.dim, tc.sets*2))
			assertSize(t, out, fileBytes(tc.dim, tc.sets))
			h := readHeader(t, out)
			if int(h.Dim) != tc.dim || int(h.Groups) != tc.sets || h.Seed != 1 {
				t.Fatalf("header %+v", h)
			}
		})
	}
}

func generateMultiplyVerify(t *testing.T, dim, sets int, seed uint64, stage int) {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "input.matrix")
	out := filepath.Join(dir, "result.matrix")
	mustRun(t, dir, "generate",
		"-out", in,
		"-dim", strconv.Itoa(dim),
		"-sets", strconv.Itoa(sets),
		"-seed", strconv.FormatUint(seed, 10),
	)
	mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", strconv.Itoa(stage))
	verifyOK(t, in, out)
}

func assertStagesExact(t *testing.T, in string, want [][]float64) {
	t.Helper()
	dir := t.TempDir()
	outs := multiplyAllStages(t, in, dir)
	for stage, path := range outs {
		_, got := loadMatrices(t, path, 1)
		assertMatricesClose(t, fmt.Sprintf("stage %d", stage), got, want, 0)
	}
	verifyOK(t, in, outs[1])
}

func multiplyAllStages(t *testing.T, in, dir string) map[int]string {
	t.Helper()
	outs := make(map[int]string, 4)
	for stage := 1; stage <= 4; stage++ {
		out := filepath.Join(dir, fmt.Sprintf("stage%d.matrix", stage))
		mustRun(t, dir, "multiply", "-in", in, "-out", out, "-stage", strconv.Itoa(stage))
		outs[stage] = out
	}
	return outs
}

func verifyOK(t *testing.T, in, result string) string {
	t.Helper()
	stdout, stderr, code := run(t, t.TempDir(), "verify", "-in", in, "-result", result)
	if code != 0 || !strings.Contains(stdout, "verify passed") {
		t.Fatalf("verify failed: code=%d\nstderr: %s\nstdout: %s", code, stderr, stdout)
	}
	return stdout
}

func mustRun(t *testing.T, dir string, args ...string) string {
	t.Helper()
	stdout, stderr, code := run(t, dir, args...)
	if code != 0 {
		t.Fatalf("%s\nexit %d\nstderr: %s\nstdout: %s", strings.Join(args, " "), code, stderr, stdout)
	}
	return stdout
}

func run(t *testing.T, dir string, args ...string) (string, string, int) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err == nil {
		return stdout.String(), stderr.String(), 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return stdout.String(), stderr.String(), ee.ExitCode()
	}
	t.Fatalf("run %v: %v", args, err)
	return "", "", -1
}

func writeInput(t *testing.T, path string, dim, groups int, seed uint64, mats [][]float64) string {
	t.Helper()
	if len(mats) != groups*2 {
		t.Fatalf("want %d matrices, got %d", groups*2, len(mats))
	}
	w, f, err := matrixio.OpenWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := matrixio.WriteHeader(w, matrixio.NewHeader(groups, dim, seed)); err != nil {
		t.Fatal(err)
	}
	for _, m := range mats {
		if len(m) != dim*dim {
			t.Fatalf("matrix len %d, want %d", len(m), dim*dim)
		}
		if err := matrixio.WriteMatrix(w, m); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func loadMatrices(t *testing.T, path string, perGroup int) (matrixio.Header, [][]float64) {
	t.Helper()
	r, f, err := matrixio.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h, err := matrixio.ReadHeader(r)
	if err != nil {
		t.Fatal(err)
	}
	n := int(h.Dim)
	out := make([][]float64, 0, int(h.Groups)*perGroup)
	for g := 0; g < int(h.Groups); g++ {
		for k := 0; k < perGroup; k++ {
			m := make([]float64, n*n)
			if err := matrixio.ReadMatrix(r, m); err != nil {
				t.Fatalf("%s group %d matrix %d: %v", path, g, k, err)
			}
			out = append(out, m)
		}
	}
	return h, out
}

func absPath(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

func readHeader(t *testing.T, path string) matrixio.Header {
	t.Helper()
	r, f, err := matrixio.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h, err := matrixio.ReadHeader(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(h.Magic[:]) != matrixio.FileMagic {
		t.Fatalf("magic %q", h.Magic)
	}
	return h
}

func assertMatricesClose(t *testing.T, label string, got, want [][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d matrices, want %d", label, len(got), len(want))
	}
	for g := range want {
		if len(got[g]) != len(want[g]) {
			t.Fatalf("%s group %d: len %d want %d", label, g, len(got[g]), len(want[g]))
		}
		idx := -1
		max := 0.0
		for i := range want[g] {
			d := math.Abs(got[g][i] - want[g][i])
			if d > max {
				max = d
				idx = i
			}
		}
		if max > tol {
			t.Fatalf("%s group %d: max abs err %g at %d (got %g want %g, tol %g)",
				label, g, max, idx, got[g][idx], want[g][idx], tol)
		}
	}
}

func assertSize(t *testing.T, path string, want int64) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() != want {
		t.Fatalf("%s size %d, want %d", path, fi.Size(), want)
	}
}

func fileBytes(dim, matrices int) int64 {
	return int64(headerBytes + matrices*dim*dim*8)
}

func sha256File(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeRaw(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func copyFile(t *testing.T, src, dst string) {
	t.Helper()
	in, err := os.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
}

func flipFirstValue(t *testing.T, path string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.Seek(headerBytes, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	var v float64
	if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
		t.Fatal(err)
	}
	v += 1
	if _, err := f.Seek(headerBytes, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}

func identity(n int) []float64 {
	m := make([]float64, n*n)
	for i := 0; i < n; i++ {
		m[i*n+i] = 1
	}
	return m
}

func sequential(n int) []float64 {
	m := make([]float64, n*n)
	for i := range m {
		m[i] = float64(i + 1)
	}
	return m
}

func filled(n int, v float64) []float64 {
	m := make([]float64, n*n)
	for i := range m {
		m[i] = v
	}
	return m
}
