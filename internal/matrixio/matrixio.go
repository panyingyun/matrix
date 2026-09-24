// Package matrixio 提供矩阵运算工程的自定义二进制文件格式读写。
//
// 文件格式（全部小端序 little-endian）：
//
//	[8]byte   魔数 "MXMATRIX"
//	uint32    版本号 = 1
//	uint32    矩阵组数（默认 3）
//	uint32    矩阵维数 n（默认 1024）
//	uint64    随机数种子（默认 42）
//	随后每组按行主序（row-major）写入 float64 值：
//	  input.matrix : 每组写 A (n*n)，再写 B (n*n)
//	  result.matrix: 每组写 C = A*B (n*n)
//
// 随机数使用固定种子（math/rand/v2 PCG），因此 input.matrix 可完全复现，
// 也可以直接保留文件供下次测试复用同一矩阵。
package matrixio

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	// FileMagic 是文件魔数。
	FileMagic = "MXMATRIX"
	// FileVersion 是当前格式版本号。
	FileVersion = 1

	// InputFile / ResultFile 是默认的数据文件名。
	InputFile  = "input.matrix"
	ResultFile = "result.matrix"

	// DefaultDim / DefaultSets 是默认矩阵规模与组数。
	DefaultDim  = 1024
	DefaultSets = 3
	// FixedSeed 是默认随机种子（保证矩阵可复现）。
	FixedSeed = 42
)

// Header 是二进制文件的定长头部（35 字节，按字段自然对齐写入）。
type Header struct {
	Magic  [8]byte
	Ver    uint32
	Groups uint32
	Dim    uint32
	Seed   uint64
}

var byteOrder = binary.LittleEndian

// NewHeader 构造文件头。
func NewHeader(groups, dim int, seed uint64) Header {
	var h Header
	copy(h.Magic[:], FileMagic)
	h.Ver = FileVersion
	h.Groups = uint32(groups)
	h.Dim = uint32(dim)
	h.Seed = seed
	return h
}

// WriteHeader 写入文件头。
func WriteHeader(w io.Writer, h Header) error {
	return binary.Write(w, byteOrder, &h)
}

// ReadHeader 读取并校验文件头。
func ReadHeader(r io.Reader) (Header, error) {
	var h Header
	if err := binary.Read(r, byteOrder, &h); err != nil {
		return h, err
	}
	if string(h.Magic[:]) != FileMagic {
		return h, errors.New("not a matrix file: bad magic")
	}
	if h.Ver != FileVersion {
		return h, fmt.Errorf("unsupported file version %d", h.Ver)
	}
	return h, nil
}

// OpenWriter 创建文件并返回 1MiB 缓冲的写入器。
func OpenWriter(path string) (*bufio.Writer, *os.File, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, nil, err
	}
	return bufio.NewWriterSize(f, 1<<20), f, nil
}

// OpenReader 打开文件并返回 1MiB 缓冲的读取器。
func OpenReader(path string) (*bufio.Reader, *os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	return bufio.NewReaderSize(f, 1<<20), f, nil
}

// WriteMatrix 写入一个行主序矩阵。
func WriteMatrix(w io.Writer, m []float64) error {
	return binary.Write(w, byteOrder, m)
}

// ReadMatrix 读入一个行主序矩阵（按 m 的长度读取）。
func ReadMatrix(r io.Reader, m []float64) error {
	return binary.Read(r, byteOrder, m)
}
