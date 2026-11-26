package main

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
	"github.com/leiaSQ/goHPC/backend"
)

// waterBlocks builds water's SIP ADC(2)x space and, per particle orbital, the block's
// rows, adapted vectors and determinant expansions.
func waterBlocks(t *testing.T) (nocc int, blocks []struct {
	rows []int
	U    backend.Mat
	dets [][]sip.DetTerm
}) {
	t.Helper()
	d := testFCIDUMP(t)
	nocc = mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	sp := sip.NewSpace(nocc, d.NORB, nil, 0)
	groups := map[int][]int{}
	for r := sp.BeginSat; r < len(sp.Configs); r++ {
		groups[sp.Configs[r].Vir] = append(groups[sp.Configs[r].Vir], r)
	}
	for p := range d.NORB - nocc {
		rows := groups[p]
		if len(rows) == 0 {
			continue
		}
		mx := sip.New(sp.Restrict(rows), ints, eps, 2, backend.Gonum{})
		_, U := backend.Gonum{}.SymEig(mx.BuildMatrix())
		dets := make([][]sip.DetTerm, len(rows))
		for i, r := range rows {
			var err error
			if dets[i], err = sp.DetExpansion(r); err != nil {
				t.Fatal(err)
			}
		}
		blocks = append(blocks, struct {
			rows []int
			U    backend.Mat
			dets [][]sip.DetTerm
		}{rows, U, dets})
	}
	return nocc, blocks
}

func randomOrthogonal(n int, seed int64) *mat.Dense {
	rng := rand.New(rand.NewSource(seed))
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(A)
	var Q mat.Dense
	qr.QTo(&Q)
	return &Q
}

// TestTwoHoleCharacterInvariants: one site owns everything -> every adapted state is fully
// one-site; and for any orthogonal L the block totals of the one-site and inner weights
// are the same over the adapted states as over the configurations they rotate (a trace
// is basis independent), which pins the rotation and the normalization.
func TestTwoHoleCharacterInvariants(t *testing.T) {
	nocc, blocks := waterBlocks(t)
	if len(blocks) == 0 {
		t.Fatal("no particle blocks")
	}
	inner := make([]bool, nocc)
	inner[1] = true
	L := randomOrthogonal(nocc, 7)
	split := make([]int, nocc)
	for k := range nocc {
		split[k] = k % 2
	}
	one := make([]int, nocc)
	for _, bl := range blocks {
		m := len(bl.rows)
		u := make([]float64, m)
		var sumA, sumC, innA, innC float64
		for j := range m {
			for i := range m {
				u[i] = bl.U.At(i, j)
			}
			iw, os := twoHoleCharacter(u, bl.dets, nocc, L, split, inner)
			if os < -1e-12 || os > 1+1e-12 || iw < -1e-12 || iw > 1+1e-12 {
				t.Fatalf("weights outside [0,1]: inner %g, one-site %g", iw, os)
			}
			sumA += os
			innA += iw
			if _, all := twoHoleCharacter(u, bl.dets, nocc, L, one, inner); math.Abs(all-1) > 1e-12 {
				t.Fatalf("one site owns every orbital but the one-site weight is %g", all)
			}
			clear(u)
			u[j] = 1 // the configuration itself
			iwC, osC := twoHoleCharacter(u, bl.dets, nocc, L, split, inner)
			sumC += osC
			innC += iwC
		}
		if math.Abs(sumA-sumC) > 1e-10 || math.Abs(innA-innC) > 1e-10 {
			t.Errorf("block of %d: one-site total %.12f (adapted) vs %.12f (configurations); "+
				"inner %.12f vs %.12f", m, sumA, sumC, innA, innC)
		}
	}
}

// TestSchemeBRotatedBlocksAreDiagonal: with the particle index rotated (compact | free),
// the adapted operator T^T M T must be exactly diagonal on every particle block, with the
// block eigenvalues on its diagonal, and its spectrum must be M's (T is orthogonal). This
// pins the assembly of the rotated blocks from the dense 2h1p block, which nothing else
// checks. The hydrogens are treated as non-compact centres (charge 0) so that the compact
// span — the oxygen's AOs projected onto the virtuals — is a genuine rotation of them.
func TestSchemeBRotatedBlocksAreDiagonal(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	ssp := sip.NewSpace(nocc, d.NORB, nil, 0)
	pmx := sip.New(ssp, ints, eps, 2, backend.Gonum{})
	md, err := mo.ReadCanonical("../../testdata/h2o_dzp.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	md.AtomCharges[1], md.AtomCharges[2] = 0, 0
	sel, err := fano.ParseClassRule("q:0:1", "")
	if err != nil {
		t.Fatal(err)
	}
	base := fano.NewPartition(ssp, sel)
	cfg := fanoConfig{sip: sipConfig{order: 2}, bInner: "e<-1.0", bCut: 0.5, bOneSite: "x",
		bParticle: "free", bCompactThresh: 0.02,
		sites: []spectrum.Site{{Name: "W", Members: []string{"O", "H1", "H2"}}}}
	f := sipFamily{parent: ssp, ints: ints, eps: eps, cfg: cfg.sip}
	part, fam, st, err := buildSchemeBWith(md, cfg, ssp, f, pmx, base, eps, nocc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if st.CompactVirtuals == 0 || st.CompactVirtuals == d.NORB-nocc {
		t.Fatalf("%d of %d virtuals compact: the rotation would be trivial", st.CompactVirtuals, d.NORB-nocc)
	}
	A := fam.full.BuildMatrix()
	var maxOff, maxDiag float64
	for _, bl := range fam.full.Transform().Blocks() {
		for i, ri := range bl.Rows {
			maxDiag = math.Max(maxDiag, math.Abs(A.At(ri, ri)-bl.Values[i]))
			for j, rj := range bl.Rows {
				if i != j {
					maxOff = math.Max(maxOff, math.Abs(A.At(ri, rj)))
				}
			}
		}
	}
	if maxOff > 1e-10 || maxDiag > 1e-10 {
		t.Errorf("rotated particle blocks: max off-diagonal %.3g, max |diag - eigenvalue| %.3g", maxOff, maxDiag)
	}
	ea, _ := backend.Gonum{}.SymEig(A)
	em, _ := backend.Gonum{}.SymEig(pmx.BuildMatrix())
	var de float64
	for i := range ea {
		de = math.Max(de, math.Abs(ea[i]-em[i]))
	}
	if de > 1e-10 {
		t.Errorf("T^T M T and M have different spectra (max diff %.3g): T is not orthogonal", de)
	}
	t.Logf("%d compact of %d virtuals; %d blocks; adapted Q %d P %d X %d (bound %d); %s",
		st.CompactVirtuals, d.NORB-nocc, st.Blocks, st.AdaptedQ, st.AdaptedP, st.AdaptedX, st.AdaptedXBound,
		part.CensusString())
}

// TestSchemeBPMPOnItsSupport: on an ADC(2,2) parent the P adapted states live in the
// 2h1p class, so adaptedFamily.matrix assembles M there alone (bCut above 1 sends every
// adapted state to P, since water's holes all localize on O). Its products must equal
// the full-parent restriction (embed, T^T M T, gather) to rounding.
func TestSchemeBPMPOnItsSupport(t *testing.T) {
	for _, particle := range []string{"any", "free"} {
		t.Run(particle, func(t *testing.T) { schemeBPMPOnItsSupport(t, particle) })
	}
}

// schemeBPMPOnItsSupport runs it for one -fano-b-particle; "free" composes the virtual
// compact | free rotation (hydrogens as non-compact centres) with the particle blocks.
func schemeBPMPOnItsSupport(t *testing.T, particle string) {
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	ssp := sip.NewSpace22(nocc, d.NORB, nil, 0)
	pmx := sip.New(ssp, ints, eps, sip.Order22, backend.Gonum{})
	md, err := mo.ReadCanonical("../../testdata/h2o_dzp.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	if particle == "free" {
		md.AtomCharges[1], md.AtomCharges[2] = 0, 0
	}
	sel, err := fano.ParseClassRule("q:1/0-3:0;q:3/0-3:0", "")
	if err != nil {
		t.Fatal(err)
	}
	base := fano.NewPartition(ssp, sel)
	cfg := fanoConfig{sip: sipConfig{order: sip.Order22}, bInner: "e<-1.0", bCut: 1.01, bOneSite: "q",
		bParticle: particle, bCompactThresh: 0.02,
		sites: []spectrum.Site{{Name: "W", Members: []string{"O"}}, {Name: "H", Members: []string{"H1", "H2"}}}}
	f := sipFamily{parent: ssp, ints: ints, eps: eps, cfg: cfg.sip}
	part, fam, _, err := buildSchemeBWith(md, cfg, ssp, f, pmx, base, eps, nocc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if part.PSize() == 0 {
		t.Fatalf("empty P: %s", part.CensusString())
	}
	psp, _ := fam.restrict(part.P)
	ch := &chooser{cands: []candidate{{name: "gonum", be: backend.Gonum{}}}}
	red, err := fam.matrix(ch, "test PMP", psp)
	if err != nil {
		t.Fatal(err)
	}
	defer red.Release()
	if _, ok := red.(ownedSub); !ok {
		t.Fatalf("PMP over %d of %d rows took the full-parent path", part.PSize(), ssp.Size())
	}
	s := psp.(*schemeBSpace)
	full := fano.NewSubOperator(fam.full, fam.full.Backend(), s.rows, s.main, fam.full.DiagonalHost())
	be := backend.Gonum{}
	n := psp.Size()
	rng := rand.New(rand.NewSource(7))
	var worst, scale float64
	for range 3 {
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		ya, yb := be.Alloc(n), be.Alloc(n)
		red.ApplyFull(ya, be.Upload(x))
		full.ApplyFull(yb, be.Upload(x))
		a, b := be.Download(ya), be.Download(yb)
		for i := range a {
			worst = math.Max(worst, math.Abs(a[i]-b[i]))
			scale = math.Max(scale, math.Abs(b[i]))
		}
	}
	if worst > 1e-10*scale {
		t.Errorf("reduced PMP differs from the full restriction by %.3g (scale %.3g)", worst, scale)
	}
	t.Logf("P %d of %d rows; max |diff| %.3g of %.3g", n, ssp.Size(), worst, scale)
}

// TestSchemeCChannels: scheme C's channels, built in sip's spin functions from each 2h1p
// row's determinant expansion, must have the spectrum of the textbook spin-adapted
// first-order two-hole matrix (-eps_i - eps_j + J +/- K, singlet and triplet), cut at
// every position; P must hold exactly open x virtuals per particle irrep; and the
// rotation must be orthogonal (T^T M T has M's spectrum). Water, C1 and C2v.
func TestSchemeCChannels(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	sel, err := fano.ParseClassRule("q:0-3:1", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, sym := range []bool{false, true} {
		var orbSym []int
		nirr := 1
		if sym {
			orbSym = d.OrbSym
			nirr = slices.Max(orbSym)
		}
		irr := func(o int) int {
			if orbSym == nil {
				return 0
			}
			return orbSym[o] - 1
		}
		ref := spinAdaptedChannels(d, eps, nocc, irr, nirr)
		ssp := sip.NewSpace(nocc, d.NORB, orbSym, 0)
		pmx := sip.New(ssp, ints, eps, 2, backend.Gonum{})
		base := fano.NewPartition(ssp, sel)
		f := sipFamily{parent: ssp, ints: ints, eps: eps, cfg: sipConfig{order: 2}}
		// Every cut position of every irrep: LastOpenEV is then each reference eigenvalue.
		maxOpen := 0
		for _, r := range ref {
			maxOpen = max(maxOpen, len(r))
		}
		for k := 1; k <= maxOpen; k++ {
			open := make([]int, nirr)
			for h := range nirr {
				open[h] = min(k, len(ref[h]))
			}
			cfg := fanoConfig{sip: sipConfig{order: 2}, cOpen: open}
			part, fam, st, err := buildSchemeC(cfg, ssp, f, pmx, base, eps, nocc, orbSym)
			if err != nil {
				t.Fatal(err)
			}
			wantP := 0
			for _, c := range st.Channels {
				h := c.Irrep - 1
				if c.Channels != len(ref[h]) {
					t.Fatalf("sym %v: two-hole irrep %d has %d channels, reference %d", sym, c.Irrep, c.Channels, len(ref[h]))
				}
				if c.Open > 0 {
					if got, want := c.LastOpenEV, ref[h][c.Open-1]*hartreeToEV; math.Abs(got-want) > 1e-8 {
						t.Errorf("sym %v irrep %d channel %d: %.10f eV, reference %.10f", sym, c.Irrep, c.Open, got, want)
					}
				}
			}
			for a := range d.NORB - nocc {
				h := irr(nocc + a) // target irrep 0: the two-hole irrep is the particle irrep
				wantP += open[h]
			}
			if part.PSize() != wantP {
				t.Errorf("sym %v open %v: P %d, want %d", sym, open, part.PSize(), wantP)
			}
			if k == 2 {
				ea, _ := backend.Gonum{}.SymEig(fam.full.BuildMatrix())
				em, _ := backend.Gonum{}.SymEig(pmx.BuildMatrix())
				var de float64
				for i := range ea {
					de = math.Max(de, math.Abs(ea[i]-em[i]))
				}
				if de > 1e-10 {
					t.Errorf("sym %v: T^T M T and M spectra differ by %.3g", sym, de)
				}
			}
		}
		t.Logf("sym %v: %d two-hole irreps, channels per irrep %v", sym, nirr, func() []int {
			n := make([]int, nirr)
			for h := range nirr {
				n[h] = len(ref[h])
			}
			return n
		}())
	}
}

// spinAdaptedChannels is the textbook first-order two-hole spectrum per two-hole irrep,
// singlet (i <= j: N[(ik|jl) + (il|jk)]) and triplet (i < j: (ik|jl) - (il|jk)) merged
// and sorted, in hartree.
func spinAdaptedChannels(d *fcidump.Data, eps []float64, nocc int, irr func(int) int, nirr int) [][]float64 {
	out := make([][]float64, nirr)
	for _, triplet := range []bool{false, true} {
		for h := range nirr {
			var pr [][2]int
			for j := range nocc {
				for i := range j + 1 {
					if (triplet && i == j) || irr(i)^irr(j) != h {
						continue
					}
					pr = append(pr, [2]int{i, j})
				}
			}
			n := len(pr)
			if n == 0 {
				continue
			}
			M := mat.NewSymDense(n, nil)
			for a, ij := range pr {
				for b, kl := range pr {
					i, j, k, l := ij[0], ij[1], kl[0], kl[1]
					var v float64
					if triplet {
						v = d.TwoE(i, k, j, l) - d.TwoE(i, l, j, k)
					} else {
						nf := 1.0
						if i == j {
							nf /= math.Sqrt2
						}
						if k == l {
							nf /= math.Sqrt2
						}
						v = nf * (d.TwoE(i, k, j, l) + d.TwoE(i, l, j, k))
					}
					if a == b {
						v -= eps[i] + eps[j]
					}
					M.SetSym(a, b, v)
				}
			}
			var es mat.EigenSym
			es.Factorize(M, false)
			out[h] = append(out[h], es.Values(nil)...)
		}
	}
	for h := range out {
		slices.Sort(out[h])
	}
	return out
}
