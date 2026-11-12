# scripts

| Folder | Contents |
|---|---|
| `fcidump/` | Integral generation: `dump_fcidump.py` (sectioned input, see `adcgo_input.py`), the GAMESS-UK readers, orbital selection, the localized-orbital basis (`orbitals.py`) and KBJ ghost bases (`kbj.py`), `threads.py` (import before numpy on the login node) |
| `fixtures/` | Regenerate the committed test fixtures in `testdata/` |
| `codegen/` | Code generators: adcgen ISR elements (`generate_adc.py`, `isrgen.sbatch`, see `internal/adc/isrgen/README.md`), the ADC(4) spin tables (`gen_coeff4.py`) |
| `helix/` | bwForCluster Helix: `HELIX.md`, the dump-and-run driver (`adcgo_run.sh`, `runADCgo*`), the CUDA build, workspaces, production and benchmark jobs |
| `gpu/` | GPU diagnostics and parity jobs (CUDA smoke, compute-sanitizer, multi-GPU parity, int32 addressing, timing A/B) |
| `analysis/` | Spectrum comparison and plotting |

Scripts find the repository root from their own location, so run them from any directory:

```sh
python scripts/fcidump/dump_fcidump.py --input examples/DIP_h2o/dump.in
sbatch scripts/helix/runADCgo examples/DIP_h2o/dump.in
```

## Auger-Meitner difference spectrum (dimer minus 2 x monomer)

Isolates what a dimer does that two isolated monomers cannot. Needs SIP **and** DIP, whole band, for
both systems; the monomer must be a **counterpoise** fragment (partner ghosted) or the difference
carries basis-set superposition error instead of physics.

```sh
# 1. integrals: dimer (exists) + counterpoise monomer
SYS=naph2 DECK=naph2_cp_mono.in sbatch scripts/dump_melanin.sbatch

# 2. whole-band solves. PREFIX picks the file set, SYS the workspace, so the fragment
#    lives beside its dimer. EIG_B2/EIG_DEVICE are the eigensolve accelerations.
SYS=naph2 PREFIX=naph2_cp_mono EIG_B2=128 EIG_DEVICE=1 sbatch scripts/melanin_dip_mgpu.sbatch
SYS=naph2 PREFIX=naph2_cp_mono sbatch scripts/melanin_sip.sbatch

# 3. solver documents -> stick spectra
adcgo -convert naph2.dip.json -dip -bare -out naph2.dip.spec.json      # and .sip

# 4. the difference. -xrange is REQUIRED (common electron grid) and 2 is exact only for a
#    homodimer whose halves are congruent.
plotspec -mode ees-diff \
  -sip naph2.sip.spec.json         -dip naph2.dip.spec.json \
  -ref-sip naph2_cp_mono.sip.spec.json -ref-dip naph2_cp_mono.dip.spec.json \
  -ref-scale 2 -xrange 0-16 -points 600 -fwhm 1.0 \
  -out naph2_am_diff.png -data naph2_am_diff.tsv
```

- `-xrange` upper bound = `max(SIP energy) - min(DIP energy)`; above it the electron spectrum is zero.
  naph2: 33.17 - 17.09 = **16.1 eV**.
- Intensities are always absolute in `ees-diff`; peak normalisation would make the factor 2 meaningless.
- Channels are summed before subtracting: ghosting breaks the dimer's C2, so the fragment has fewer
  irreps and the decompositions do not correspond.
### Which reference, and why the residual means different things

A dimer's non-interacting double-ionization manifold has **three** families: both holes on A, both on
B, and **one hole on each**. A monomer DIP contains only the one-site kind, so `2 x monomer`
reproduces two families and omits the third entirely.

| reference | flags | residual contains |
|---|---|---|
| one-site only | (default) | the missing two-site family **+** interaction effects, conflated |
| one- and two-site | `-ref-twosite` | interaction effects alone (shifts, splittings, intensity transfer) |

`-ref-twosite` completes the reference by self-convolving `-ref-sip`: each ordered pair of monomer
ionizations (i on A, j on B) becomes a final state at `E_i + E_j + shift` with the independent-hole
weight `s_i * s_j`. Run both and compare — the first says how much of the spectrum the monomers cannot
account for at all, the second how much of *that* is interaction rather than bookkeeping.

```sh
plotspec -mode ees-diff ... -ref-twosite -twosite-shift 4.2
```

- `-twosite-shift` is the hole-hole Coulomb repulsion `e^2/R`, which the strict R->inf reference omits
  but a dimer at contact does not: `14.4/R(Angstrom)` eV. Use the monomer **centroid** separation, not
  the interplanar gap: naph2's stack is slipped, so R = **3.62 A** (-> 3.98 eV), not 3.43 A (4.20 eV).
  Default 0 = the strict non-interacting limit.
- `-twosite-weight` defaults to 0.5, the non-interacting state count, chosen so `-ref-scale 2` stays
  correct: an overall factor on the final-state spectrum cancels between sigma's numerator and its
  `N(E)` denominator, so only the one-site:two-site ratio and the initial-state scale matter.
- It is an **independent-hole construction**, not an ab-initio two-site DIP, and it applies no spin
  multiplicity (each spatial two-site configuration supports a singlet and a triplet). Use
  `-twosite-weight` if a different convention is wanted.
- Cost: `N^2` synthetic states. 573 SIP lines -> 328k states, convolved in ~8 s.

### Decay-channel spectra: define the SITES or the channels are meaningless

`-mode panel` colours panel (b) by decay channel, which needs a DIP spectrum converted with
`-spectrum -mo` rather than `-bare` (a `-bare` spectrum has one `states` channel and nothing to group
by). Two flags decide whether the result means anything.

```sh
# WRONG: no -group, so every ATOM is its own site
adcgo -convert naph2.dip.json -dip -spectrum -mo naph2.mo.json -init-atom C1 -out bad.spec.json
#   -> 38 channels (Auger@C1, ICD:C1->C2 ... ETMD(3)); ETMD(3) takes 91% of the weight

# RIGHT: fold the atoms into the chemical units the channels are about
A=C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,H1,H2,H3,H4,H5,H6,H7,H8
B=C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,H9,H10,H11,H12,H13,H14,H15,H16
adcgo -convert naph2.dip.json -dip -spectrum -mo naph2.mo.json \
      -group="A=$A" -group="B=$B" -init-atom A -out naph2.dip.frag.spec.json
#   -> Auger@A / ICD:A->B / ETMD(2); naph2: 22% / 56% / 22% in the EES window
```

The classifier routes each state by **where its two holes sit relative to the initially ionized
site**: both on A = Auger, one on A one on B = ICD, both on one site != A = ETMD(2), two sites != A =
ETMD(3). The default is *each population column its own site*, so with 36 atomic sites "two different
sites" is satisfied by almost every state and ETMD(3) swallows the spectrum. That 91% is an artefact
of the site definition, not physics. With two sites ETMD(3) is structurally impossible.

- `-init-atom` defaults to **"O"** and fails outright on a hydrocarbon ("initial site \"O\" is not one
  of the defined sites"). With `-group` it takes a SITE name.
- Free correctness check on a homodimer: `Auger@A` and `ETMD(2)` must be equal by the A<->B symmetry.
  naph2 gives 420.149 each.
- `-point-group C2` records the point group in the spectrum meta, so `plotspec -panel-sym` can label
  irreps without `-irrep-labels`.
- `plotspec -dip-family` collapses the channels onto Auger / ICD / ETMD for the figure, keeping three
  canonical hues; without it the shared key grows one row per atom pair and buries the panels.

**Panel (c)'s one-site family is Auger + ETMD(2), not "Auger" alone.** "Both holes on one monomer" is
Auger when they are on the initially ionized monomer and ETMD(2) when they are on the other, which is
exactly what the factor 2 on `-ref-scale` counts. The split is 1:1 and exact for a homodimer, so
`-mode panel` with `-ref-sip/-ref-dip` draws them as separate curves (ETMD(2) dashed, because it
coincides with Auger).
