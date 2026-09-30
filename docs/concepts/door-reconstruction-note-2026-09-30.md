> **Working note, versed to the register as a declaration (Rev77, 2026-09-30), as written that day.** Nothing here is a result. The door it opens is §5d, item 7, of the whitepaper; the design rules it names are §3.5; the reading it names is §4.7c. The physics it reads is the author's private thermodynamics program (six papers, a theorem register, follow up notes and scripts), of which only Paper I is public in a July 2026 snapshot marked not yet submitted.

# Door note: the reconstruction theorem and the four part partition

*Written 2026-09-30 for the quantum-cortex register, door after whitepaper v1 (the §5d doors). Advisory reading of the private thermodynamics repository (six papers, THEOREMS.md, notes 1 to 9, scripts). Nothing here enters v1. Script `reconstruction_theorem.py` was rerun on 2026-09-30 and reproduces every stated identity (symbolic zeros; converse residuals 6.6e-19 and 3.2e-30).*

## 1. What was asked

Whether the thermodynamics program supplies, for an observer of episodes, the three ingredients of Theorem X1 (Paper V): a mass line, an invariant fixed outside the observed variable, and a closure. And, for each level, what is stored, what is derived, and which of the four words it may carry: theorem, reconstructible at measured defect, backup, declared loss.

## 2. Verdict

The theorems do not transfer as theorems. All six papers concern horizon polynomials of the no cubic quartic class and its neighbours; none speaks of episodes. What transfers is a template, an exact instance of the four part partition, and a precise statement of why the many episode case is open.

## 3. X1 in the four part partition

Data: two entropies along a mass line, S± = πa(M)², πb(M)², with first law temperatures; nondegenerate (a ≠ b, a', b' ≠ 0, ab nonconstant). No metric, no polynomial, no charge assumed.

Axioms on the observed pair only. (S1) the signed defect of the visible ledger is the pair monomial c(a − b)²(a + b). (S2) the completed product ab[λ − ab + (a + b)²] = k is conserved.

| Part | Content in X1 | Count |
|---|---|---|
| Invariant | c, λ, k; read off the axioms at two states with distinct ab, then fixed | three constants |
| Stored carrier | a(M), b(M), equivalently (a + b, ab) | two functions |
| Derived, discarded | the virtual pair v + v̄ = −(a + b), v v̄ = λ − ab + (a + b)²; the quartic w²r⁴ + βr² − 2Mr + γ with (w², β, γ) = 2c(1, λ, k); the flow a', b'; the mass itself, Φ/2 with dΦ/dM = 2; the mass law; the residue closure | everything else, by Vieta |
| Residual | the mass origin M₀ | one number |

Word carried: theorem. The rule is Vieta and two algebraic axioms; it passes through nothing that changes with a version.

Three facts about the residual and the carrier that the physics states exactly:

1. Relative order is derivable from the carrier; the absolute origin is not. M is reconstructed up to a shift; no law along the line fixes the shift. It is closed only by a transverse axiom, a second dimension of the state space: the Coulomb axiom (X3) leaves the origin rigid across charge slices, the angular first law axiom (X4) annihilates it.
2. Two states suffice to calibrate the invariant. λ and k are linear in (S2); two states with distinct product determine them.
3. Neither axiom forces alone (Proposition 1, minimality). A defect law alone leaves one functional freedom; a conservation law alone leaves free level flows.

## 4. The three ingredients, sharpened by the papers

**Mass line.** A continuous parameter that orders states, with a first law, and with movement: the nondegeneracy condition (ab nonconstant) means a family that does not move cannot be reconstructed.

**Invariant.** Constants fixed outside the observed variable. Proposition 2 (the fixed J foliation exits the class) adds the sharp condition: a constant that co-varies with the order parameter turns the family into a composed flow, outside the class, and reconstruction fails. An invariant that changes is a change of slice; slices are reconnected only by a transverse axiom.

**Closure.** Hidden in the axioms: e₁ = 0 (the sum of all roots, visible and hidden, vanishes) and e₂ = λ, e₄ = k constant. This is what lets Vieta return the hidden pair from the visible one.

## 5. Why the many episode case is open, in the physics itself

X7 (one root collapse): with one physical root, one conservation axiom suffices. X1: two physical roots, two axioms. Paper V, open problem (ii): polynomials of degree n > 4 carrying two physical roots and n − 2 virtual ones, where the count of transverse axioms should match the count of free Vieta functions. The forcing depth grows with the number of observed things, and the general count is the program's own frontier. An observer of N episodes is the general degree case. There is no Vieta for episodes in these papers, and the papers say where it would have to be proved.

## 6. What transfers by design

These are design principles for the write path, to be built rather than discovered, then verified by measurement. Each carries its lens of origin so a reader can return to the source.

1. **The origin is stored, never derived** (X1, X3, X4). Relative order can be reconstructed from the carrier; the absolute anchor of an episode to the world cannot. It is captured at write time or fixed by a second axis (the domain, the identity). This is the precise reason the "where" channel must enter the write path: no local law returns it later.
2. **Two laws, not one** (Proposition 1). A ledger of defects alone, or a conserved budget alone, leaves freedom. Reconstruction needs a law for the shape of the defect as a function of the visible state and a conserved completed quantity.
3. **A drifting invariant breaks reconstruction** (Proposition 2). Identity, preferences, standing decisions are the invariant; when one changes, the change is a new slice, explicit, with provenance, connected to the old one by a declared transverse rule. Silent drift is a composed flow and reconstruction fails.
4. **A collision is a crossing, not a wall** (X5). At a = b the forced flow is regular and transversal; the pair turns complex conjugate and continues with the completed product conserved. Two episodes that become indistinguishable are moved to the hidden sector as a pair with their invariant kept, not deleted one against the other.
5. **The defect sizes the hidden share but does not name it** (Theorem L2, Paper III). The visible ledger balances exactly when nothing is hidden; the defect is exactly the hidden share; and the defect monomial is blind to the hidden content. From the visible alone one knows how much is omitted, never what. This bounds what any ledger of forgettings can promise.

## 7. What stays a lens

- The beta lens (Paper I §9, Brick 0): apparent volatility is omitted context, R² = 1 under full accounting. The founding move; the origin of "any inconsistency is an omitted variable until proved otherwise".
- The obstruction and its damping (Paper II, Theorems 2 and 3): the flow is generically not isomonodromic, the obstruction is the price of non integrability, and the discriminant both detects the collision and damps the obstruction. Reading: edits near a collision of two memories are where integrity is most at risk. Measured as the obstruction of each edit, never assumed.
- The ledger split second law (Y2): under accretion both shares grow, the hidden one strictly, and the visible share of the inflow is bounded (one third in RN AdS, a number of that polynomial). Reading only.
- The Davies decomposition (Y3): the sign change is a kinematic balance of growth rates of the complete root system. In the organ the knee is measured on a generating probe set, not calculated.

## 8. Attribution and citation status

- Lemma 0 (two horizon equivalence) is Chen, Liu and Zhang, arXiv:1206.2015, on Castro and Rodriguez, arXiv:1204.1284 (inner first law and products). The forcing (X1) is the author's, candidate new, gate Q46 executed 2026-07-13.
- Papers II to VI are established results, every statement an exact symbolic identity or an audited high precision evaluation with a named reproduction script, rerun on 2026-09-30. They are not yet submitted: the pipeline is gated by the Paper I arXiv endorsement (pre submission checklist, item 0), and the reproduction scripts of the follow up notes are not yet in the public repository (the public snapshot is Paper I of July 2026, marked "not yet submitted, do not cite without contacting the author").
- Consequence for the cortex whitepaper: the door states the idea with the attributed lineage and names the forcing theorem as the author's result with its reproduction script. Until that script is public a reader cannot verify it, so the idea must stand in the door on its own terms; the day the follow up payload is pushed, the citation becomes verifiable in the program's own currency, a script rather than an identifier.
- The physics carries its own honesty flags (X4 dictionary conditional; inner horizon instability caveats inherited from the literature). The door inherits them and claims nothing the physics does not.

## 9. Effect on the plan

Nothing for v1. The door after v1 takes the five design principles of section 6, each declared with its lens, and the open problem of section 5 stated as such. The Monday grid, four parts and four words, is answered for the physics: X1 is a theorem; the episode case is not yet drawn.

---

*Addendum at versing (2026-09-30, afternoon).* Section 9 was written before the founder's decision of the same day that the thermodynamics aspects identified for the organ enter v1 in two registers, by design verified on the code and by reading on graved numbers (register Rev73). What entered: the five principles of section 6 as §3.5 of the whitepaper, each with its word, satisfied or declared, after the code was read; the reading of §4.7c; the door of §5d, item 7, with the open problem of section 5; the three regimes of the retention law in §3.5; the lenses of section 7 in the register only, with their origin, unmeasured. The reconstruction test as a threshold protocol is declared in the door and not run in this edition. The note itself is kept as written.
