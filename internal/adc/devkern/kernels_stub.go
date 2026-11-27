//go:build !cuda || hip

package devkern

import "github.com/leiaSQ/goHPC/backend"

const noCuda = "devkern: built without the cuda tag; a backend.DeviceAccess cannot exist in this build"

// SetCoeff1 uploads the flattened [3][13][30] spin table to constant memory (once).
func SetCoeff1(backend.DeviceAccess, []float64) { panic(noCuda) }

// Wert2Apply launches the matrix-free 2h1p x 3h2p coupling apply.
func Wert2Apply(backend.DeviceAccess, Wert2Args) { panic(noCuda) }

// C22Apply launches the matrix-free order-3 2h1p x 2h1p satellite apply.
func C22Apply(backend.DeviceAccess, C22Args) { panic(noCuda) }

// DipSatApply launches the matrix-free DIP 3h1p <-> 3h1p satellite apply.
func DipSatApply(backend.DeviceAccess, DipSatArgs) { panic(noCuda) }

// DipSatFillJII materializes a batch of DIP satellite blocks into device scratch.
func DipSatFillJII(backend.DeviceAccess, DipFillJIIArgs) []backend.DeviceMat { panic(noCuda) }

// ProfilerStart / ProfilerStop do nothing in a build without the cuda tag.
func ProfilerStart(backend.DeviceAccess) {}
func ProfilerStop(backend.DeviceAccess)  {}
