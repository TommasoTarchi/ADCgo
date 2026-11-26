// Reduced-precision operator applies: an effective-precision model of an analog
// matrix-vector unit (photonic or in-memory compute), wrapped around an Operator so
// every σ = M·x the block-Krylov drivers issue is carried out at N bits instead of
// float64.
//
// WHY THIS LEVEL AND NOT THE BACKEND. The obvious place for a precision model is
// backend.GemmMat, where the dense blocks are multiplied. That place is wrong here: the
// satellite region of a large DIP/SIP sector is applied matrix-free (dip.op.mf, the
// DeviceAccess / HostData paths), so it never passes through GemmMat at all. At the
// production DIP shape that region IS the operator — the dense blocks are 8.6 GB of a
// 26.7 GB sector — so a backend-level model would quantize the small part and leave the
// large part in full precision, which is worse than not modelling it. Operator.ApplyBlock
// is the one chokepoint every contribution passes through.
//
// WHAT IS MODELLED. An analog MVM unit has three independent error sources, all of the
// same relative order 2^-N: the DACs that write the operator weights into the mesh, the
// DACs that encode the input vector, and the ADCs that read the result out. Only the last
// two are addressable at this level, because a matrix-free operator has no materialized
// weights to perturb. So `bits` is an EFFECTIVE bit depth: quantizing the apply's input
// and output at N bits reproduces the combined perturbation of all three sources to within
// a factor of order one (√3 for independent contributions), and that factor is far smaller
// than the spread between the bit depths worth testing. Reported results should say
// "effective N-bit applies", never "N-bit hardware".
//
// The quantizer is symmetric mid-tread with a per-column scale — the natural model for a
// unit that normalizes each input vector before encoding it — and saturates at the
// column's own maximum, as a real ADC does. `scale=panel` shares one scale across the
// block instead, which is the pessimistic variant (a small column is then encoded with
// few levels).
//
// This is an experimental knob, off unless ADCGO_QUANT is set, and it changes results by
// construction: it is never on in a production run. When it is on, every driver prints one
// line to stderr naming the configuration, so a log can never be mistaken for a
// full-precision solve.
package lanczos

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
	"github.com/leiaSQ/goHPC/backend"
)

// QuantConfig is the reduced-precision apply model, parsed from ADCGO_QUANT.
type QuantConfig struct {
	// Bits is the effective bit depth of one apply, 2..52. The quantizer resolves
	// 2^(Bits-1)-1 levels either side of zero, so Bits=8 is 127 levels per sign — an
	// 8-bit signed DAC/ADC pair.
	Bits int

	// Noise is the standard deviation of additive Gaussian noise applied before readout
	// quantization, relative to the scale's maximum. 0 (the default) models a noiseless
	// converter chain, which isolates quantization as the only error source.
	Noise float64

	// In and Out select which side of the apply is quantized. Both default to true;
	// stage=in or stage=out isolates one of them, which is how you tell an input-encoding
	// failure from a readout failure.
	In, Out bool

	// PerColumn scales each column of the block independently (the default). False shares
	// one scale across the whole block.
	PerColumn bool

	// Seed makes the noise draw reproducible. The draw depends on (Seed, apply index,
	// column), never on goroutine scheduling, so a run with Noise > 0 repeats exactly.
	Seed uint64
}

// String renders the configuration for the activation notice.
func (c QuantConfig) String() string {
	scale := "panel"
	if c.PerColumn {
		scale = "col"
	}
	stage := "in+out"
	switch {
	case c.In && !c.Out:
		stage = "in"
	case !c.In && c.Out:
		stage = "out"
	}
	return fmt.Sprintf("bits=%d noise=%g stage=%s scale=%s seed=%d", c.Bits, c.Noise, stage, scale, c.Seed)
}

// quantEnv is the parsed ADCGO_QUANT, computed once. A malformed value panics rather than
// being ignored: silently running at full precision would be reported as a quantized
// result, which is the one failure mode this code must not have.
var quantEnv = sync.OnceValues(func() (QuantConfig, bool) {
	spec := strings.TrimSpace(os.Getenv("ADCGO_QUANT"))
	if spec == "" || spec == "off" || spec == "0" {
		return QuantConfig{}, false
	}
	cfg, err := ParseQuantSpec(spec)
	if err != nil {
		panic("lanczos: ADCGO_QUANT: " + err.Error())
	}
	return cfg, true
})

// ParseQuantSpec parses a comma-separated key=value apply-precision spec:
//
//	bits=8                        effective bit depth (required, 2..52)
//	noise=1e-3                    relative Gaussian noise before readout (default 0)
//	stage=in|out|both             which side to quantize (default both)
//	scale=col|panel               per-column or per-block scale (default col)
//	seed=1                        noise seed (default 1)
func ParseQuantSpec(spec string) (QuantConfig, error) {
	cfg := QuantConfig{In: true, Out: true, PerColumn: true, Seed: 1}
	bitsSet := false
	for _, field := range strings.Split(spec, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		key, val, ok := strings.Cut(field, "=")
		if !ok {
			return cfg, fmt.Errorf("field %q is not key=value", field)
		}
		key, val = strings.TrimSpace(key), strings.TrimSpace(val)
		switch key {
		case "bits":
			n, err := strconv.Atoi(val)
			if err != nil {
				return cfg, fmt.Errorf("bits=%q: %w", val, err)
			}
			if n < 2 || n > 52 {
				return cfg, fmt.Errorf("bits=%d out of range (2..52; above 52 a float64 mantissa has nothing left to round)", n)
			}
			cfg.Bits, bitsSet = n, true
		case "noise":
			f, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return cfg, fmt.Errorf("noise=%q: %w", val, err)
			}
			if f < 0 {
				return cfg, fmt.Errorf("noise=%g must be >= 0", f)
			}
			cfg.Noise = f
		case "stage":
			switch val {
			case "in":
				cfg.In, cfg.Out = true, false
			case "out":
				cfg.In, cfg.Out = false, true
			case "both", "in+out":
				cfg.In, cfg.Out = true, true
			default:
				return cfg, fmt.Errorf("stage=%q: want in | out | both", val)
			}
		case "scale":
			switch val {
			case "col":
				cfg.PerColumn = true
			case "panel":
				cfg.PerColumn = false
			default:
				return cfg, fmt.Errorf("scale=%q: want col | panel", val)
			}
		case "seed":
			n, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return cfg, fmt.Errorf("seed=%q: %w", val, err)
			}
			cfg.Seed = n
		default:
			return cfg, fmt.Errorf("unknown key %q (want bits, noise, stage, scale, seed)", key)
		}
	}
	if !bitsSet {
		return cfg, fmt.Errorf("bits= is required")
	}
	return cfg, nil
}

// quantNotice prints the activation line once per process.
var quantNotice sync.Once

// WrapQuant returns op wrapped in the reduced-precision apply model when ADCGO_QUANT is
// set, and op itself otherwise. The wrapper preserves SatelliteOperator, which SolveLowMem
// type-asserts for its Tarantelli gate — losing it would silently drop Mode B to the
// ungated path and change the pole strengths for a reason having nothing to do with
// precision.
func WrapQuant(op Operator, be backend.Backend) Operator {
	cfg, on := quantEnv()
	if !on {
		return op
	}
	q := newQuantOp(op, be, cfg)
	if sat, ok := op.(SatelliteOperator); ok {
		return &quantSatOp{quantOp: q, sat: sat}
	}
	return q
}

// WrapQuantPrecon is WrapQuant for the Davidson driver, preserving Diagonal. The
// preconditioner diagonal is NOT quantized: it is assembled once from the element
// functions, not applied per iteration, so it is not work an accelerator would carry.
func WrapQuantPrecon(op PreconOperator, be backend.Backend) PreconOperator {
	cfg, on := quantEnv()
	if !on {
		return op
	}
	return &quantPreconOp{quantOp: newQuantOp(op, be, cfg), precon: op}
}

// quantOp applies the inner operator at reduced effective precision.
//
// The input panel is copied before quantization — it is the driver's live Krylov block,
// and rounding it in place would corrupt the basis rather than model a lossy encoder. The
// output panel is quantized in place, which is safe because ApplyBlock's contract already
// requires a dedicated buffer with Ld == Rows that it zeroes and overwrites.
type quantOp struct {
	Operator // ApplyFull, Size, MainBlockSize; the drivers call only ApplyBlock

	be  backend.Backend
	cfg QuantConfig

	applies atomic.Uint64 // apply index, so each call draws fresh noise

	mu   sync.Mutex
	host []float64      // staging for one compacted panel
	scr  backend.Vector // quantized copy of the input panel
	scrN int
}

func newQuantOp(op Operator, be backend.Backend, cfg QuantConfig) *quantOp {
	quantNotice.Do(func() {
		fmt.Fprintf(os.Stderr, "adcgo: REDUCED-PRECISION APPLIES ACTIVE (ADCGO_QUANT: %s) -- results are a hardware model, not a converged ADC spectrum\n", cfg)
	})
	return &quantOp{Operator: op, be: be, cfg: cfg}
}

// Release frees the scratch panel. Safe to call more than once.
func (q *quantOp) Release() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.scr != nil {
		q.be.Free(q.scr)
		q.scr, q.scrN = nil, 0
	}
	q.host = nil
}

func (q *quantOp) ApplyBlock(out, in backend.BlockView) {
	n := q.applies.Add(1)
	q.Operator.ApplyBlock(out, q.encode(in, n))
	q.readout(out, n)
}

func (q *quantOp) applyBlockSat(sat SatelliteOperator, out, in backend.BlockView) {
	n := q.applies.Add(1)
	sat.ApplyBlockSatellite(out, q.encode(in, n))
	q.readout(out, n)
}

// encode returns the input panel as the operator should see it: a quantized copy when the
// input stage is enabled, the original otherwise.
func (q *quantOp) encode(in backend.BlockView, call uint64) backend.BlockView {
	if !q.cfg.In || in.Rows == 0 || in.Cols == 0 {
		return in
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	nelem := in.Rows * in.Cols
	buf := q.stage(nelem)
	q.readPanel(in, buf)
	q.quantize(buf, in.Rows, in.Cols, call, false)

	if q.scrN < nelem {
		if q.scr != nil {
			q.be.Free(q.scr)
		}
		q.scr, q.scrN = q.be.Alloc(nelem), nelem
	}
	dst := q.scr.Slice(0, nelem)
	q.writeVec(dst, buf)
	// Compacted by construction: the staging buffer is column-major with no padding.
	return backend.BlockView{V: dst, Rows: in.Rows, Cols: in.Cols, Ld: in.Rows}
}

// readout quantizes the apply's result in place, modelling the ADC that digitizes it.
func (q *quantOp) readout(out backend.BlockView, call uint64) {
	if !q.cfg.Out || out.Rows == 0 || out.Cols == 0 {
		return
	}
	if out.Ld != out.Rows {
		panic(fmt.Sprintf("lanczos: quantized readout needs a compacted output panel (Ld %d != Rows %d); ApplyBlock's contract guarantees one", out.Ld, out.Rows))
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	nelem := out.Rows * out.Cols
	buf := q.stage(nelem)
	q.readPanel(out, buf)
	q.quantize(buf, out.Rows, out.Cols, call, true)
	q.writeVec(out.V.Slice(0, nelem), buf)
}

// stage returns the host staging buffer grown to at least n elements. Held across applies
// so a 300-column production panel is not reallocated thousands of times.
func (q *quantOp) stage(n int) []float64 {
	if cap(q.host) < n {
		q.host = make([]float64, n)
	}
	return q.host[:n]
}

// readPanel gathers b into the compacted column-major dst, taking the cheapest route the
// backend offers: a host slice (no transfer), a strided 2-D copy, or a whole-vector
// download gathered by stride.
func (q *quantOp) readPanel(b backend.BlockView, dst []float64) {
	if hd, ok := q.be.(backend.HostData); ok {
		s := hd.HostSlice(b.V)
		for c := range b.Cols {
			copy(dst[c*b.Rows:(c+1)*b.Rows], s[c*b.Ld:c*b.Ld+b.Rows])
		}
		return
	}
	// A compacted panel is one contiguous download, which BufferedDownloader does without
	// allocating; Download2D would allocate a fresh multi-GB slice on every apply.
	if bd, ok := q.be.(backend.BufferedDownloader); ok && b.Ld == b.Rows {
		bd.DownloadInto(dst, b.V.Slice(0, b.Rows*b.Cols))
		return
	}
	if sd, ok := q.be.(backend.StridedDownloader); ok {
		copy(dst, sd.Download2D(b.V, b.Rows, b.Cols, b.Ld))
		return
	}
	full := q.be.Download(b.V)
	for c := range b.Cols {
		copy(dst[c*b.Rows:(c+1)*b.Rows], full[c*b.Ld:c*b.Ld+b.Rows])
	}
}

// writeVec copies the compacted host buffer into the backend-resident dst.
func (q *quantOp) writeVec(dst backend.Vector, src []float64) {
	if hd, ok := q.be.(backend.HostData); ok {
		copy(hd.HostSlice(dst), src)
		return
	}
	tmp := q.be.Upload(src)
	q.be.Copy(dst, tmp)
	q.be.Free(tmp)
}

// quantize rounds the compacted rows×cols panel to the configured precision. readout
// selects whether noise is injected (it models the analog stage ahead of the ADC, so it
// belongs to the output side only).
func (q *quantOp) quantize(x []float64, rows, cols int, call uint64, readout bool) {
	noise := 0.0
	if readout {
		noise = q.cfg.Noise
	}
	if !q.cfg.PerColumn {
		quantizeSlice(x, q.cfg.Bits, noise, q.rng(call, 0))
		return
	}
	// One column is rows elements — hundreds of thousands at production scale — so a
	// column is heavy work and HeavyRows is the right pool: Rows' small-count fallback
	// would run a 300-column panel serially.
	parallel.HeavyRows(cols, func(c int) {
		quantizeSlice(x[c*rows:(c+1)*rows], q.cfg.Bits, noise, q.rng(call, c))
	})
}

// rng returns the generator for one (apply, column) pair. Seeding from the pair rather than
// from shared state keeps a noisy run reproducible whatever order the workers run in, and
// needs no lock on the hot path.
func (q *quantOp) rng(call uint64, col int) *rand.Rand {
	if q.cfg.Noise == 0 {
		return nil
	}
	return rand.New(rand.NewPCG(q.cfg.Seed^call, uint64(col)+0x9e3779b97f4a7c15))
}

// quantizeSlice applies symmetric mid-tread quantization to x in place.
//
// The scale comes from the slice's own maximum magnitude, which is what a unit that
// normalizes each vector before encoding it does, and it is also the only choice that
// needs no knowledge of the operator's spectrum. Values are saturated at that maximum
// after noise, as a converter clips rather than wrapping.
func quantizeSlice(x []float64, bits int, noise float64, rng *rand.Rand) {
	amax := 0.0
	for _, v := range x {
		if a := math.Abs(v); a > amax {
			amax = a
		}
	}
	if amax == 0 || math.IsInf(amax, 0) || math.IsNaN(amax) {
		return // an all-zero column carries no information to lose; a broken one is not ours to hide
	}
	levels := float64(int64(1)<<(bits-1) - 1)
	step := amax / levels
	for i, v := range x {
		if rng != nil {
			v += noise * amax * rng.NormFloat64()
		}
		l := math.RoundToEven(v / step)
		if l > levels {
			l = levels
		} else if l < -levels {
			l = -levels
		}
		x[i] = l * step
	}
}

// quantSatOp adds the Tarantelli satellite gate, quantized the same way as a full apply.
type quantSatOp struct {
	*quantOp
	sat SatelliteOperator
}

func (q *quantSatOp) ApplyBlockSatellite(out, in backend.BlockView) {
	q.applyBlockSat(q.sat, out, in)
}

// quantPreconOp adds the Davidson preconditioner diagonal, unquantized (see WrapQuantPrecon).
type quantPreconOp struct {
	*quantOp
	precon PreconOperator
}

func (q *quantPreconOp) Diagonal(be backend.Backend) backend.Vector { return q.precon.Diagonal(be) }

// releaseQuant frees a quantizing wrapper's scratch panel at the end of a solve. A no-op
// when ADCGO_QUANT is unset, since op is then the caller's own operator.
func releaseQuant(op any) {
	if r, ok := op.(interface{ Release() }); ok {
		r.Release()
	}
}
