package lanczos

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/leiaSQ/goHPC/backend"
)

func TestParseQuantSpec(t *testing.T) {
	cfg, err := ParseQuantSpec("bits=8")
	if err != nil {
		t.Fatalf("bits=8: %v", err)
	}
	if cfg.Bits != 8 || !cfg.In || !cfg.Out || !cfg.PerColumn || cfg.Noise != 0 || cfg.Seed != 1 {
		t.Fatalf("defaults wrong: %+v", cfg)
	}

	cfg, err = ParseQuantSpec("bits=12, noise=1e-3 ,stage=out,scale=panel,seed=7")
	if err != nil {
		t.Fatalf("full spec: %v", err)
	}
	if cfg.Bits != 12 || cfg.Noise != 1e-3 || cfg.In || !cfg.Out || cfg.PerColumn || cfg.Seed != 7 {
		t.Fatalf("full spec wrong: %+v", cfg)
	}

	// A typo must be an error, never a silent full-precision run.
	for _, bad := range []string{"", "bits=1", "bits=53", "bits=x", "noise=-1", "noise=2",
		"bits=8,stage=sideways", "bits=8,scale=rows", "bits=8,bogus=1", "bits=8,junk"} {
		if _, err := ParseQuantSpec(bad); err == nil {
			t.Errorf("ParseQuantSpec(%q) = nil error, want a rejection", bad)
		}
	}
}

func TestQuantizeSliceLevels(t *testing.T) {
	// Every output must sit on one of the 2*(2^(bits-1)-1)+1 representable levels, and the
	// error must be bounded by half a step.
	for _, bits := range []int{2, 4, 8, 16} {
		x := make([]float64, 2000)
		rng := rand.New(rand.NewPCG(1, 2))
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		orig := make([]float64, len(x))
		copy(orig, x)

		amax := 0.0
		for _, v := range x {
			amax = math.Max(amax, math.Abs(v))
		}
		levels := float64(int64(1)<<(bits-1) - 1)
		step := amax / levels

		quantizeSlice(x, bits, 0, nil)

		for i, v := range x {
			l := v / step
			if math.Abs(l-math.Round(l)) > 1e-9 {
				t.Fatalf("bits=%d: x[%d]=%g is not on a level grid (l=%g)", bits, i, v, l)
			}
			if math.Abs(l) > levels+1e-9 {
				t.Fatalf("bits=%d: x[%d]=%g exceeds saturation (|l|=%g > %g)", bits, i, v, math.Abs(l), levels)
			}
			if err := math.Abs(v - orig[i]); err > step/2+1e-12 {
				t.Fatalf("bits=%d: x[%d] error %g exceeds half a step %g", bits, i, err, step/2)
			}
		}
	}
}

func TestQuantizeSliceRelativeError(t *testing.T) {
	// The RMS relative error of an N-bit quantizer should track 2^-N to within a small
	// factor. This is the number the whole experiment is calibrated against, so it is
	// asserted rather than assumed.
	for _, bits := range []int{8, 10, 12} {
		x := make([]float64, 100000)
		rng := rand.New(rand.NewPCG(3, 4))
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		orig := make([]float64, len(x))
		copy(orig, x)
		quantizeSlice(x, bits, 0, nil)

		var sse, ss float64
		for i := range x {
			d := x[i] - orig[i]
			sse += d * d
			ss += orig[i] * orig[i]
		}
		rel := math.Sqrt(sse / ss)
		want := math.Ldexp(1, -bits)
		if rel > 20*want || rel < want/20 {
			t.Errorf("bits=%d: RMS relative error %.3g is not within a factor 20 of 2^-%d = %.3g", bits, rel, bits, want)
		}
	}
}

func TestQuantizeSliceDegenerate(t *testing.T) {
	// An all-zero column has no scale and must pass through untouched rather than
	// producing NaN from a 0/0 step.
	z := make([]float64, 16)
	quantizeSlice(z, 8, 0, nil)
	for i, v := range z {
		if v != 0 {
			t.Fatalf("zero column perturbed at %d: %g", i, v)
		}
	}
	// A single nonzero element is exactly representable (it sets the scale itself).
	one := []float64{0, 0, -2.5, 0}
	quantizeSlice(one, 8, 0, nil)
	if one[2] != -2.5 {
		t.Fatalf("scale-setting element changed: %g, want -2.5", one[2])
	}
	quantizeSlice([]float64{}, 8, 0, nil) // must not panic
}

func TestQuantizeSliceNoiseSaturates(t *testing.T) {
	// With noise the pre-ADC value can exceed the scale; the result must clip, not wrap.
	x := make([]float64, 5000)
	for i := range x {
		x[i] = 1.0
	}
	x[0] = -1.0 // fix the scale at 1
	quantizeSlice(x, 8, 0.5, rand.New(rand.NewPCG(5, 6)))
	for i, v := range x {
		if math.Abs(v) > 1.0+1e-12 {
			t.Fatalf("x[%d]=%g exceeded the saturation limit 1.0", i, v)
		}
	}
}

// quantTestOp is a dense symmetric operator satisfying SatelliteOperator, where the
// "satellite" part is the matrix with its leading main row/column band zeroed — the same
// relationship dip.Matrix has.
type quantTestOp struct {
	m    backend.Mat
	main int
	be   backend.Backend
	nSat int // ApplyBlockSatellite calls, to prove the gate survives wrapping
}

func (o *quantTestOp) Size() int          { return o.m.Rows }
func (o *quantTestOp) MainBlockSize() int { return o.main }

func (o *quantTestOp) ApplyFull(out, in backend.Vector) {
	o.be.Zero(out)
	o.be.GemvN(1, o.be.UploadMat(o.m), in, out)
}

func (o *quantTestOp) ApplyBlock(out, in backend.BlockView) {
	o.be.Zero(out.V)
	o.be.GemmMat(false, 1, o.be.UploadMat(o.m), in, 1, out)
}

func (o *quantTestOp) ApplyBlockSatellite(out, in backend.BlockView) {
	o.nSat++
	s := backend.NewMat(o.m.Rows, o.m.Cols)
	copy(s.Data, o.m.Data)
	for i := range s.Rows {
		for j := range s.Cols {
			if i < o.main || j < o.main {
				s.Set(i, j, 0)
			}
		}
	}
	o.be.Zero(out.V)
	o.be.GemmMat(false, 1, o.be.UploadMat(s), in, 1, out)
}

func quantTestMatrix(n int) backend.Mat {
	m := backend.NewMat(n, n)
	rng := rand.New(rand.NewPCG(11, 12))
	for i := range n {
		for j := i; j < n; j++ {
			v := rng.NormFloat64()
			if i == j {
				v += float64(i) + 10 // a clean, well-separated spectrum
			} else {
				v *= 0.1
			}
			m.Set(i, j, v)
			m.Set(j, i, v)
		}
	}
	return m
}

func TestQuantOpPreservesSatelliteGate(t *testing.T) {
	be := backend.Gonum{}
	op := &quantTestOp{m: quantTestMatrix(24), main: 6, be: be}
	q := &quantOp{Operator: op, be: be, cfg: QuantConfig{Bits: 8, In: true, Out: true, PerColumn: true, Seed: 1}}
	sat := &quantSatOp{quantOp: q, sat: op}

	// The whole point of WrapQuant's type switch: SolveLowMem must still see a
	// SatelliteOperator, or Mode B silently drops its Tarantelli gate.
	var asOp Operator = sat
	so, ok := asOp.(SatelliteOperator)
	if !ok {
		t.Fatal("quantSatOp does not satisfy SatelliteOperator")
	}

	in := backend.BlockView{V: be.Alloc(24 * 3), Rows: 24, Cols: 3, Ld: 24}
	out := backend.BlockView{V: be.Alloc(24 * 3), Rows: 24, Cols: 3, Ld: 24}
	hs := be.HostSlice(in.V)
	for i := range hs {
		hs[i] = float64(i%7) - 3
	}
	so.ApplyBlockSatellite(out, in)
	if op.nSat != 1 {
		t.Fatalf("ApplyBlockSatellite reached the inner operator %d times, want 1", op.nSat)
	}

	// The satellite gate must leave the main rows untouched when the input has no
	// satellite weight: a quantized apply must not leak weight into the main block.
	for i := range hs {
		hs[i] = 0
	}
	for c := range 3 {
		be.HostSlice(in.V)[c*24] = 1 // main-space unit vectors only
	}
	so.ApplyBlockSatellite(out, in)
	for c := range 3 {
		for r := range op.main {
			if v := be.HostSlice(out.V)[c*24+r]; v != 0 {
				t.Fatalf("satellite apply put %g in main row %d of column %d", v, r, c)
			}
		}
	}
}

func TestQuantOpInputIsNotMutated(t *testing.T) {
	// The input panel is the driver's live Krylov block. Quantizing it in place would
	// corrupt the basis instead of modelling a lossy encoder.
	be := backend.Gonum{}
	op := &quantTestOp{m: quantTestMatrix(32), main: 8, be: be}
	q := &quantOp{Operator: op, be: be, cfg: QuantConfig{Bits: 6, In: true, Out: true, PerColumn: true, Seed: 1}}
	defer q.Release()

	in := backend.BlockView{V: be.Alloc(32 * 4), Rows: 32, Cols: 4, Ld: 32}
	out := backend.BlockView{V: be.Alloc(32 * 4), Rows: 32, Cols: 4, Ld: 32}
	rng := rand.New(rand.NewPCG(21, 22))
	hs := be.HostSlice(in.V)
	for i := range hs {
		hs[i] = rng.NormFloat64()
	}
	want := make([]float64, len(hs))
	copy(want, hs)

	q.ApplyBlock(out, in)

	for i := range want {
		if hs[i] != want[i] {
			t.Fatalf("input column data changed at %d: %g != %g", i, hs[i], want[i])
		}
	}
	// And the output must actually have been quantized — a 6-bit readout of a dense
	// product cannot come back bit-identical to the exact one.
	exact := backend.BlockView{V: be.Alloc(32 * 4), Rows: 32, Cols: 4, Ld: 32}
	op.ApplyBlock(exact, in)
	same := true
	for i, v := range be.HostSlice(out.V) {
		if v != be.HostSlice(exact.V)[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("6-bit apply reproduced the float64 apply exactly; the model is not engaged")
	}
}

func TestQuantOpStageSelection(t *testing.T) {
	be := backend.Gonum{}
	m := quantTestMatrix(20)

	run := func(cfg QuantConfig) []float64 {
		op := &quantTestOp{m: m, main: 5, be: be}
		q := &quantOp{Operator: op, be: be, cfg: cfg}
		defer q.Release()
		in := backend.BlockView{V: be.Alloc(20 * 2), Rows: 20, Cols: 2, Ld: 20}
		out := backend.BlockView{V: be.Alloc(20 * 2), Rows: 20, Cols: 2, Ld: 20}
		rng := rand.New(rand.NewPCG(31, 32))
		hs := be.HostSlice(in.V)
		for i := range hs {
			hs[i] = rng.NormFloat64()
		}
		q.ApplyBlock(out, in)
		return be.Download(out.V)
	}

	base := QuantConfig{Bits: 8, PerColumn: true, Seed: 1}
	inOnly := base
	inOnly.In = true
	outOnly := base
	outOnly.Out = true
	both := base
	both.In, both.Out = true, true

	a, b, c := run(inOnly), run(outOnly), run(both)
	// stage=out lands on the readout grid; stage=in does not (it rounds the encoder side
	// and then accumulates in float64). So the three must be mutually distinct.
	if equalVecs(a, b) || equalVecs(a, c) || equalVecs(b, c) {
		t.Fatal("stage=in, stage=out and stage=both produced indistinguishable results")
	}
}

func equalVecs(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestQuantOpNoiseReproducible(t *testing.T) {
	be := backend.Gonum{}
	m := quantTestMatrix(20)
	run := func() []float64 {
		op := &quantTestOp{m: m, main: 5, be: be}
		q := &quantOp{Operator: op, be: be,
			cfg: QuantConfig{Bits: 8, In: true, Out: true, PerColumn: true, Noise: 1e-2, Seed: 99}}
		defer q.Release()
		in := backend.BlockView{V: be.Alloc(20 * 3), Rows: 20, Cols: 3, Ld: 20}
		out := backend.BlockView{V: be.Alloc(20 * 3), Rows: 20, Cols: 3, Ld: 20}
		hs := be.HostSlice(in.V)
		for i := range hs {
			hs[i] = float64(i%5) - 2
		}
		q.ApplyBlock(out, in)
		q.ApplyBlock(out, in) // a second apply must draw DIFFERENT noise
		return be.Download(out.V)
	}
	if !equalVecs(run(), run()) {
		t.Fatal("a seeded noisy run is not reproducible")
	}
}

func TestWrapQuantOffByDefault(t *testing.T) {
	// With ADCGO_QUANT unset (the state every production run is in), WrapQuant must hand
	// back the caller's own operator — not a wrapper that happens to be configured off.
	// quantEnv is read once per process, so this asserts against the process environment;
	// running the suite under ADCGO_QUANT is a deliberate quantized run and skips it.
	if _, on := quantEnv(); on {
		t.Skip("ADCGO_QUANT is set for this process")
	}
	be := backend.Gonum{}
	op := &quantTestOp{m: quantTestMatrix(8), main: 2, be: be}
	if got := WrapQuant(op, be); got != Operator(op) {
		t.Fatalf("WrapQuant wrapped the operator with ADCGO_QUANT unset: %T", got)
	}
}

// releasingOp is an operator with a Release of its own, as dip.Matrix, sip.Matrix and the
// generated sigma operator have.
type releasingOp struct {
	*quantTestOp
	released int
}

func (o *releasingOp) Release() { o.released++ }
func (o *releasingOp) Diagonal(be backend.Backend) backend.Vector {
	d := make(backend.Vec, o.m.Rows)
	for i := range d {
		d[i] = o.m.At(i, i)
	}
	return be.Upload(d)
}

func TestSolversDoNotReleaseTheCallersOperator(t *testing.T) {
	// Every driver defers releaseQuant on the operator it was given. With ADCGO_QUANT
	// unset that is the caller's own operator, and releasing it there freed the blocks a
	// second solve on the same operator needs: the generated sigma operator's triplet
	// solve came back with 10 Krylov vectors of 960 after its singlet solve.
	if _, on := quantEnv(); on {
		t.Skip("ADCGO_QUANT is set for this process")
	}
	be := backend.Gonum{}
	const n, main = 24, 4
	newOp := func() *releasingOp {
		return &releasingOp{quantTestOp: &quantTestOp{m: quantTestMatrix(n), main: main, be: be}}
	}
	for _, tc := range []struct {
		name  string
		solve func(op *releasingOp)
	}{
		{"Solve", func(op *releasingOp) { Solve(op, be, Options{Block: main, MaxDim: n}) }},
		{"SolveLowMem", func(op *releasingOp) { SolveLowMem(op, be, Options{Block: main, MaxDim: n}) }},
		{"SolveDavidson", func(op *releasingOp) { SolveDavidson(op, be, Options{Block: main, NRoots: 2}) }},
	} {
		op := newOp()
		tc.solve(op)
		if op.released != 0 {
			t.Errorf("%s released the caller's operator %d time(s)", tc.name, op.released)
		}
	}
}
