// kernel_amd64.s — AVX2/FMA 2x16 微内核（Go plan9 汇编）。
//
// Mul2x16 计算 C[0:2][0:16] += A[0:2][0:k] * B[0:k][0:16]（行主序 float64）。
//
// ABI0 参数（编译器自动生成 ABIInternal->ABI0 包装）：
//
//	k   +0(FP)   int      求和长度
//	a   +8(FP)   *float64 &A[i0][k0]
//	b   +16(FP)  *float64 &B[k0][j0]
//	c   +24(FP)  *float64 &C[i0][j0]
//	lda +32(FP)  int      A 行跨度（元素数）
//	ldb +40(FP)  int      B 行跨度（元素数）
//	ldc +48(FP)  int      C 行跨度（元素数）
//
// 寄存器：Y0-Y7 累加器（2 行 × 16 列），Y8-Y11 B 行（16 个 double），
// Y12/Y13 A 元素广播；R10 为 A 列指针，R11 为 k 计数器。

#include "textflag.h"

TEXT ·Mul2x16(SB), NOSPLIT, $0-56
	MOVQ k+0(FP), AX
	MOVQ a+8(FP), BX
	MOVQ b+16(FP), CX
	MOVQ c+24(FP), DI
	MOVQ lda+32(FP), SI
	MOVQ ldb+40(FP), R8
	MOVQ ldc+48(FP), R9

	// 载入 C 的 2 行 × 16 列累加器
	VMOVUPD 0(DI), Y0          // 行 0，列 0-3
	VMOVUPD 32(DI), Y1         // 行 0，列 4-7
	VMOVUPD 64(DI), Y2         // 行 0，列 8-11
	VMOVUPD 96(DI), Y3         // 行 0，列 12-15
	VMOVUPD 0(DI)(R9*8), Y4    // 行 1，列 0-3
	VMOVUPD 32(DI)(R9*8), Y5   // 行 1，列 4-7
	VMOVUPD 64(DI)(R9*8), Y6   // 行 1，列 8-11
	VMOVUPD 96(DI)(R9*8), Y7   // 行 1，列 12-15

	SHLQ $3, R8                // ldb：元素 -> 字节
	MOVQ BX, R10               // A 列指针
	XORQ R11, R11              // kk = 0
	TESTQ AX, AX
	JZ   store                 // k == 0 直接返回

loop:
	// 载入 B 行（16 个 double）
	VMOVUPD 0(CX), Y8          // B[k][0-3]
	VMOVUPD 32(CX), Y9         // B[k][4-7]
	VMOVUPD 64(CX), Y10        // B[k][8-11]
	VMOVUPD 96(CX), Y11        // B[k][12-15]
	// 广播 A 的两行元素
	VBROADCASTSD (R10), Y12          // A[0][k]
	VBROADCASTSD (R10)(SI*8), Y13    // A[1][k]
	// 行 0 累加
	VFMADD231PD Y12, Y8, Y0
	VFMADD231PD Y12, Y9, Y1
	VFMADD231PD Y12, Y10, Y2
	VFMADD231PD Y12, Y11, Y3
	// 行 1 累加
	VFMADD231PD Y13, Y8, Y4
	VFMADD231PD Y13, Y9, Y5
	VFMADD231PD Y13, Y10, Y6
	VFMADD231PD Y13, Y11, Y7
	// 推进：A 下一列，B 下一行
	ADDQ $8, R10
	ADDQ R8, CX
	INCQ R11
	CMPQ R11, AX
	JLT  loop

store:
	VMOVUPD Y0, 0(DI)
	VMOVUPD Y1, 32(DI)
	VMOVUPD Y2, 64(DI)
	VMOVUPD Y3, 96(DI)
	VMOVUPD Y4, 0(DI)(R9*8)
	VMOVUPD Y5, 32(DI)(R9*8)
	VMOVUPD Y6, 64(DI)(R9*8)
	VMOVUPD Y7, 96(DI)(R9*8)
	RET
