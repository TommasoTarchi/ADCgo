// Package devkern holds ADCgo's own device kernels: the matrix-free appliers whose
// element recompute is a custom CUDA kernel reading device-resident integrals, so that
// the large coupling and satellite blocks never occupy VRAM.
//
// The kernels are ADCgo's and are compiled and linked here (adc4_kernels.cu,
// adc2dip_kernels.cu; kernels_cuda.go). What runs them is goHPC's backend: a launcher is
// handed to backend.DeviceAccess, which calls it on the thread that owns the device and
// gives out the device pointers behind the resident panels. On a build without the cuda
// tag the functions exist and panic; nothing reaches them, because no backend of such a
// build implements DeviceAccess.
package devkern

import (
	"unsafe"

	"github.com/leiaSQ/goHPC/backend"
)

// DipFillJIIArgs carries the per-slot block geometry and the shared integral tensors for
// DipSatFillJII. Every Row*/Col*/BufOff/Virs pointer is an UploadInts result
// indexed by slot (Virs is a flat concatenation of per-group virtual-orbital lists, addressed via
// RowVOff/ColVOff); ERI/Eps/OrbSym are the same device tensors DipSatApply uses.
//
// Rows/Cols mirror RowNv/ColNv on the host so the returned DeviceMat handles can be shaped
// without a device round-trip.
//
// One call materializes the slot half-open range [SlotLo, SlotHi) — a "chunk" — into a scratch
// buffer of ChunkElems doubles, and returns SlotHi-SlotLo handles (handle k is global slot
// SlotLo+k). The caller drives the whole plan by looping chunks and reusing the buffer, so the
// resident scratch is bounded by the chunk budget instead of the full materialized operator,
// which for a large sector is hundreds of GB (job 14026481 OOM'd at 424 GB before this).
// BufOff is therefore CHUNK-LOCAL — it resets to 0 at each chunk boundary — which also keeps it
// inside int32 (the global prefix sum reaches ~5e10 elements and would wrap).
//
// TotalElems is the whole-plan element count (informational); MaxElems (largest single block)
// sizes the grid; ChunkElems sizes the scratch allocation for THIS call.
type DipFillJIIArgs struct {
	NSlot, Spin, Norb, Parts int
	TotalElems, MaxElems     int
	SlotLo, SlotHi           int   // half-open slot range materialized by this call
	ChunkElems               int   // scratch size for this chunk (Σ rows·cols over [SlotLo,SlotHi))
	Rows, Cols               []int // per-slot FULL block dims (parts already folded in)

	// Kind selects the element function and how a flat (r,c) decodes into (spin part, virtual):
	// 0 = jiiLKK, 1 = ijkMLL, 2 = ijkLMN. Mirrors dip.satKind.
	Kind unsafe.Pointer

	RowO0, RowO1, RowO2 unsafe.Pointer // O2 unused (-1) for the 2-occupied JII family
	ColO0, ColO1, ColO2 unsafe.Pointer
	RowVOff, RowNv      unsafe.Pointer // virtual-group offset/size, WITHOUT the parts factor
	ColVOff, ColNv      unsafe.Pointer
	BufOff, Virs        unsafe.Pointer
	ERI, Eps, OrbSym    unsafe.Pointer
}

// Wert2Args carries the device buffers and dimensions for Wert2Apply. The
// R* / C* pointers are UploadInts results (row/col config struct-of-arrays); ERI is a
// UploadFloats result; In/Out are resident panels (their device pointers via DevPtr).
type Wert2Args struct {
	N2, N3, B, LdIn, LdOut, MainOff, Off3, Norb, Nocc int
	RVir, RK, RL, RTyp                                unsafe.Pointer
	CI, CJ, CK, CL, CM, CSpin                         unsafe.Pointer
	ERI                                               unsafe.Pointer
	In, Out                                           backend.Vector
}

// C22Args carries the device buffers and dimensions for C22Apply. K/L/Vir/Typ
// are UploadInts results (the 2h1p config struct-of-arrays, length N2); ERI is an UploadFloats
// result and Eps an UploadFloats result (the norb orbital energies); In/Out are resident
// panels (device pointers via DevPtr). The 2h1p region starts at MainOff within each panel.
type C22Args struct {
	N2, B, LdIn, LdOut, MainOff, Norb, Nocc int
	K, L, Vir, Typ                          unsafe.Pointer
	ERI, Eps                                unsafe.Pointer
	In, Out                                 backend.Vector
}

// DipSatArgs carries the device buffers and dimensions for DipSatApply. The
// R* pointers are the per-row config struct-of-arrays (length Nsat); J*/I* are the JII/IJK
// per-group struct-of-arrays with JVir/IVir the flat concatenated absolute virtual orbitals;
// ERI is a UploadFloats result, Eps and OrbSym UploadInts/UploadFloats results. In/Out are
// resident panels (device pointers via DevPtr). The 3h1p region starts at MainOff. Spin is
// 0 (singlet) or 1 (triplet); Parts is 2 or 3.
type DipSatArgs struct {
	Nsat, Njii, Nijk, B, LdIn, LdOut, MainOff, Norb, Parts, Spin int

	// RowLo/RowHi are the half-open 3h1p row band this launch owns and OutRowOff the global
	// row offset of the output panel, for the per-device (-mgpu) apply: device d passes its own
	// band and OutRowOff = bounds[d]. In must be FULL HEIGHT regardless of the band — a
	// candidate column may live in any partition. A whole-band single-device call passes
	// RowLo=0, RowHi=Nsat, OutRowOff=0. Callers must skip the call entirely when the band is
	// empty (RowHi <= RowLo) — that is a bug in the band derivation, not a no-op.
	RowLo, RowHi, OutRowOff int

	RTyp, RGrp, RPart, RVir              unsafe.Pointer
	JO0, JO1, JSt, JVoff, JNv, JVir      unsafe.Pointer
	IO0, IO1, IO2, ISt, IVoff, INv, IVir unsafe.Pointer
	ERI, Eps, OrbSym                     unsafe.Pointer
	In, Out                              backend.Vector
}
