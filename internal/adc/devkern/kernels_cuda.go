//go:build cuda && !hip

// kernels_cuda.go -- the launchers of adc4_kernels.cu and adc2dip_kernels.cu, compiled by
// nvcc into objects beside this file and linked here. They must be built before
// `go build -tags cuda`:
//
//go:generate nvcc -O3 -std=c++14 -c adc4_kernels.cu -o adc4_kernels.o
//go:generate nvcc -O3 -std=c++14 -c adc2dip_kernels.cu -o adc2dip_kernels.o
//
// (scripts/helix/build_adcgo_cuda_helix does it, with goHPC's own two.) See
// docs/adc4_matfree_gpu.md.

package devkern

/*
#cgo LDFLAGS: ${SRCDIR}/adc4_kernels.o ${SRCDIR}/adc2dip_kernels.o -L/usr/local/cuda/lib64 -lcudart -lstdc++

// Launchers defined in adc4_kernels.cu (extern "C").
int adc4_set_coeff1(const double* h_coeff1);
int adc4_wert2_apply(int n2,int n3,int b,int ldIn,int ldOut,int mainOff,int off3,int norb,int nocc,
    const int* rVir,const int* rK,const int* rL,const int* rTyp,
    const int* cI,const int* cJ,const int* cK,const int* cL,const int* cM,const int* cSpin,
    const double* eri,const double* xin,double* yout);
int adc4_c22_apply(int n2,int b,int ldIn,int ldOut,int mainOff,int norb,int nocc,
    const int* K,const int* L,const int* Vir,const int* Typ,
    const double* eri,const double* eps,const double* xin,double* yout);

// Launcher defined in adc2dip_kernels.cu (extern "C").
int adc2_dip_fill_sat(int nslot,int spin,int norb,int parts,int maxElems,
    const int* kind,
    const int* rowO0,const int* rowO1,const int* rowO2,
    const int* colO0,const int* colO1,const int* colO2,
    const int* rowVOff,const int* rowNv,const int* colVOff,const int* colNv,
    const int* bufOff,const int* virs,
    const double* eri,const double* eps,const int* osym,double* buf);
int adc2_dip_sat_apply(int nsat,int njii,int nijk,int b,int ldIn,int ldOut,
    int mainOff,int norb,int parts,int spin,
    int rowLo,int rowHi,int outRowOff,
    const int* rTyp,const int* rGrp,const int* rPart,const int* rVir,
    const int* jO0,const int* jO1,const int* jSt,const int* jVoff,const int* jNv,const int* jVir,
    const int* iO0,const int* iO1,const int* iO2,const int* iSt,const int* iVoff,const int* iNv,const int* iVir,
    const double* eri,const double* eps,const int* osym,
    const double* xin,double* yout);

#include <cuda_profiler_api.h>
static int adc_prof_start(void) { return (int)cudaProfilerStart(); }
static int adc_prof_stop(void)  { return (int)cudaProfilerStop(); }
*/
import "C"

import (
	"unsafe"

	"github.com/leiaSQ/goHPC/backend"
)

// SetCoeff1 uploads the flattened [3][13][30] spin table to constant memory (once).
func SetCoeff1(dev backend.DeviceAccess, coeff1 []float64) {
	dev.Launch("adc4_set_coeff1", func() int {
		return int(C.adc4_set_coeff1((*C.double)(unsafe.Pointer(&coeff1[0]))))
	})
}

// Wert2Apply launches the matrix-free 2h1p x 3h2p coupling apply (forward + transpose) on
// the device-owning thread, accumulating into a.Out.
func Wert2Apply(dev backend.DeviceAccess, a Wert2Args) {
	xin := (*C.double)(dev.DevPtr(a.In))
	yout := (*C.double)(dev.DevPtr(a.Out))
	dev.Launch("adc4_wert2_apply", func() int {
		return int(C.adc4_wert2_apply(C.int(a.N2), C.int(a.N3), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Off3), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.RVir), (*C.int)(a.RK), (*C.int)(a.RL), (*C.int)(a.RTyp),
			(*C.int)(a.CI), (*C.int)(a.CJ), (*C.int)(a.CK), (*C.int)(a.CL), (*C.int)(a.CM), (*C.int)(a.CSpin),
			(*C.double)(a.ERI), xin, yout))
	})
}

// C22Apply launches the matrix-free order-3 2h1p x 2h1p satellite apply (single symmetric
// pass) on the device-owning thread, accumulating into a.Out.
func C22Apply(dev backend.DeviceAccess, a C22Args) {
	xin := (*C.double)(dev.DevPtr(a.In))
	yout := (*C.double)(dev.DevPtr(a.Out))
	dev.Launch("adc4_c22_apply", func() int {
		return int(C.adc4_c22_apply(C.int(a.N2), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.K), (*C.int)(a.L), (*C.int)(a.Vir), (*C.int)(a.Typ),
			(*C.double)(a.ERI), (*C.double)(a.Eps), xin, yout))
	})
}

// DipSatApply launches the matrix-free DIP 3h1p <-> 3h1p satellite apply (one thread per
// output 3h1p row) on the device-owning thread, accumulating into a.Out.
func DipSatApply(dev backend.DeviceAccess, a DipSatArgs) {
	xin := (*C.double)(dev.DevPtr(a.In))
	yout := (*C.double)(dev.DevPtr(a.Out))
	dev.Launch("adc2_dip_sat_apply", func() int {
		return int(C.adc2_dip_sat_apply(C.int(a.Nsat), C.int(a.Njii), C.int(a.Nijk), C.int(a.B),
			C.int(a.LdIn), C.int(a.LdOut), C.int(a.MainOff), C.int(a.Norb), C.int(a.Parts), C.int(a.Spin),
			C.int(a.RowLo), C.int(a.RowHi), C.int(a.OutRowOff),
			(*C.int)(a.RTyp), (*C.int)(a.RGrp), (*C.int)(a.RPart), (*C.int)(a.RVir),
			(*C.int)(a.JO0), (*C.int)(a.JO1), (*C.int)(a.JSt), (*C.int)(a.JVoff), (*C.int)(a.JNv), (*C.int)(a.JVir),
			(*C.int)(a.IO0), (*C.int)(a.IO1), (*C.int)(a.IO2), (*C.int)(a.ISt), (*C.int)(a.IVoff), (*C.int)(a.INv), (*C.int)(a.IVir),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym), xin, yout))
	})
}

// DipSatFillJII materializes a batch of DIP satellite blocks into the backend's device
// scratch and returns one DeviceMat handle per slot, pointing into it.
//
// The scratch is the backend's (DeviceAccess.Scratch): grown on demand and reused across
// applies, because this runs once per mat-vec and a cudaMalloc/cudaFree pair per call
// would put allocator churn on the hot path. Handles are invalidated by the next call,
// which is why the caller consumes them immediately via GemmMatBatched.
func DipSatFillJII(dev backend.DeviceAccess, a DipFillJIIArgs) []backend.DeviceMat {
	lo, hi := a.SlotLo, a.SlotHi
	n := hi - lo
	if n <= 0 {
		return nil
	}
	// The kernel indexes every per-slot array by its y-block, so passing the arrays offset by lo
	// and launching hi-lo y-blocks materializes exactly the global range [lo,hi) -- no kernel-side
	// range parameter needed. Virs/ERI/Eps/OrbSym are shared (indexed by content, not slot) and
	// are NOT offset. BufOff is chunk-local, so its offsets and this call's scratch both start at 0.
	off32 := func(p unsafe.Pointer) *C.int { return (*C.int)(unsafe.Add(p, lo*4)) } // int32 arrays
	buf := dev.Scratch(a.ChunkElems)
	dev.Launch("adc2_dip_fill_sat", func() int {
		return int(C.adc2_dip_fill_sat(C.int(n), C.int(a.Spin), C.int(a.Norb), C.int(a.Parts),
			C.int(a.MaxElems), off32(a.Kind),
			off32(a.RowO0), off32(a.RowO1), off32(a.RowO2),
			off32(a.ColO0), off32(a.ColO1), off32(a.ColO2),
			off32(a.RowVOff), off32(a.RowNv), off32(a.ColVOff), off32(a.ColNv),
			off32(a.BufOff), (*C.int)(a.Virs),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym),
			(*C.double)(buf)))
	})

	// Handles for [lo,hi): handle k is global slot lo+k, placed at the same chunk-local prefix
	// the kernel wrote to (BufOff is that same prefix, reset to 0 at the chunk boundary).
	const elemSize = 8
	out := make([]backend.DeviceMat, n)
	off := 0
	for k := range n {
		i := lo + k
		out[k] = dev.WrapMat(unsafe.Add(buf, off*elemSize), a.Rows[i], a.Cols[i])
		off += a.Rows[i] * a.Cols[i]
	}
	return out
}

// ProfilerStart / ProfilerStop open and close a profiler capture window on one GPU
// (nsys --capture-range=cudaProfilerApi). Launch runs them on the thread that owns the
// GPU, where its CUDA context is current.
func ProfilerStart(dev backend.DeviceAccess) {
	dev.Launch("cudaProfilerStart", func() int { return int(C.adc_prof_start()) })
}

func ProfilerStop(dev backend.DeviceAccess) {
	dev.Launch("cudaProfilerStop", func() int { return int(C.adc_prof_stop()) })
}
