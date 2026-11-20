import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Abel

/-!
Core certificates for the UQ manuscript, Lean 4.35.0 / Mathlib v4.35.0.
Scope: transformed-jet algebra; actual one-variable derivatives of the gain
primitive, sensitivity, energy and moment ratios; scalar uniform integrals;
complex-Hilbert fidelity projection. PDE existence, integration by parts on R,
the infinite-dimensional Taylor remainder, and numerical accuracy are NOT
asserted to have been formalized here. See verification.md for the claim map.
-/
noncomputable section
namespace SolitonUQ
open Real Complex
open scoped BigOperators

def gainStar (b bp n np w : ℝ) : ℝ := bp/b - np/n - b/w
def defect (b bp n np w g : ℝ) : ℝ := g - gainStar b bp n np w

theorem residual_identity (b w ell g t : ℝ) (hw : w ≠ 0)
    (q qz qtt qt qn : ℂ) :
    Complex.I * (b/w^2 : ℝ) * qz + (b/(2*w^2) : ℝ)*qtt
      + (b/w^2 : ℝ)*qn
      + Complex.I * (-b*t/w^2 + b*(t/w)/w : ℝ)*qt
      + ((b*t^2/(2*w^2) - b/2*(t/w)^2 : ℝ)
      + Complex.I * (ell/2-b/w+b/(2*w)-g/2 : ℝ))*q
      = (b/w^2 : ℝ)*(Complex.I*qz+qtt/2+qn)
      - Complex.I*((g-(ell-b/w))/2 : ℝ)*q := by
  have hwc : (w : ℂ) ≠ 0 := by exact_mod_cast hw
  push_cast
  field_simp
  ring

theorem gain_primitive_derivative {b n w : ℝ → ℝ} {bp np wp z : ℝ}
    (hb : HasDerivAt b bp z) (hn : HasDerivAt n np z)
    (hw : HasDerivAt w wp z) (hb0 : b z ≠ 0) (hn0 : n z ≠ 0)
    (hw0 : w z ≠ 0) (hwp : wp = b z) :
    HasDerivAt (fun x => Real.log (b x)-Real.log (n x)-Real.log (w x))
      (gainStar (b z) bp (n z) np (w z)) z := by
  subst wp
  exact ((hb.log hb0).sub (hn.log hn0)).sub (hw.log hw0)

theorem defect_derivative (b bp n np w g db dbp dn dnp dw dg : ℝ)
    (hb : b ≠ 0) (hn : n ≠ 0) (hw : w ≠ 0) :
    HasDerivAt
      (fun e => defect (b+e*db) (bp+e*dbp) (n+e*dn) (np+e*dnp) (w+e*dw) (g+e*dg))
      (dg-dbp/b+bp*db/b^2+dnp/n-np*dn/n^2+db/w-b*dw/w^2) 0 := by
  have affine (a d : ℝ) : HasDerivAt (fun e : ℝ => a+e*d) d 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const d).const_add a
  have hb0 : b+(0 : ℝ)*db ≠ 0 := by simpa using hb
  have hn0 : n+(0 : ℝ)*dn ≠ 0 := by simpa using hn
  have hw0 : w+(0 : ℝ)*dw ≠ 0 := by simpa using hw
  have h := (affine g dg).sub
    ((((affine bp dbp).div (affine b db) hb0).sub
      ((affine np dnp).div (affine n dn) hn0)).sub
      ((affine b db).div (affine w dw) hw0))
  convert h using 1
  · rfl
  · simp only [zero_mul, add_zero]
    field_simp
    ring

theorem multiplicative_defect_slope (b bp n np w fb fbp fn fnp dw h : ℝ)
    (hb : b ≠ 0) (hn : n ≠ 0) (hw : w ≠ 0) :
    h-(bp*fb+b*fbp)/b+bp*(b*fb)/b^2
      +(np*fn+n*fnp)/n-np*(n*fn)/n^2+(b*fb)/w-b*dw/w^2
      = h-fbp+fnp+b/w*(fb-dw/w) := by
  field_simp
  ring

theorem amplitude_gauge (a ap b n g : ℝ) (ha : a ≠ 0) (u uz utt nonlinear : ℂ) :
    Complex.I*((ap : ℂ)*u+(a : ℂ)*uz)
      + (b/2 : ℝ)*(a : ℂ)*utt
      + (n/a^2 : ℝ)*(a^3 : ℝ)*nonlinear
      - Complex.I*((g+2*ap/a)/2 : ℝ)*(a : ℂ)*u
      = (a : ℂ)*(Complex.I*uz+(b/2 : ℝ)*utt+(n : ℂ)*nonlinear
          -Complex.I*(g/2 : ℝ)*u) := by
  have hac : (a : ℂ) ≠ 0 := by exact_mod_cast ha
  push_cast
  field_simp
  ring

theorem gain_free_clock_residual (b n r : ℝ) (hb : b ≠ 0) (vS vtt nonlinear : ℂ) :
    Complex.I * (b : ℂ) * vS + (b/2 : ℝ)*vtt + (n*r : ℝ)*nonlinear =
      (b : ℂ)*(Complex.I*vS + vtt/2 + (n*r/b : ℝ)*nonlinear) := by
  have hbc : (b : ℂ) ≠ 0 := by exact_mod_cast hb
  push_cast
  field_simp

-- Complex jets of u and its conjugate are kept as separate variables here.
-- The density and flux are the corresponding polynomial expressions.
theorem local_energy_jet (b n g : ℝ) (u ub utt ubtt : ℂ) :
    (Complex.I*(b/2 : ℝ)*utt + Complex.I*(n : ℂ)*u^2*ub + (g/2 : ℝ)*u)*ub
    + u*(-Complex.I*(b/2 : ℝ)*ubtt - Complex.I*(n : ℂ)*ub^2*u + (g/2 : ℝ)*ub)
    + (b : ℂ)*(ub*utt-u*ubtt)/(2*Complex.I) = (g : ℂ)*(u*ub) := by
  push_cast
  field_simp
  ring_nf
  rw [Complex.I_sq]
  ring

private def flowJet (b n g : ℝ) (I' u ub utt : ℂ) : ℂ :=
  I'*(b/2 : ℝ)*utt + I'*(n : ℂ)*u^2*ub + (g/2 : ℝ)*u

private def flowTimeJet (b n g : ℝ) (I' u ub ut ubt uttt : ℂ) : ℂ :=
  I'*(b/2 : ℝ)*uttt + I'*(n : ℂ)*(2*u*ut*ub+u^2*ubt) + (g/2 : ℝ)*ut

theorem local_momentum_jet (b n g : ℝ) (u ub ut ubt utt ubtt uttt ubttt : ℂ) :
    (flowJet b n g (-Complex.I) ub u ubtt*ut
      + ub*flowTimeJet b n g Complex.I u ub ut ubt uttt
      - flowJet b n g Complex.I u ub utt*ubt
      - u*flowTimeJet b n g (-Complex.I) ub u ubt ut ubttt)/(2*Complex.I)
    + (b/4 : ℝ)*(utt*ubt+ut*ubtt-ub*uttt-u*ubttt)
    - (n : ℂ)*(u*ub)*(ut*ub+u*ubt)
    = (g : ℂ)*(ub*ut-u*ubt)/(2*Complex.I) := by
  unfold flowJet flowTimeJet
  push_cast
  field_simp
  ring_nf

theorem cubic_directional_expansion (u v : ℂ) (e : ℝ) :
    (u+(e : ℂ)*v)^2*star (u+(e : ℂ)*v)-u^2*star u
      -(e : ℂ)*(2*u*star u*v+u^2*star v)
      = (e : ℂ)^2*(v^2*star u+2*u*v*star v)+(e : ℂ)^3*v^2*star v := by
  simp only [star_add, star_mul, Complex.star_def,
    Complex.conj_ofReal]
  ring

theorem exponential_energy_derivative {G : ℝ → ℝ} {g z E0 : ℝ}
    (hG : HasDerivAt G g z) :
    HasDerivAt (fun x => E0*Real.exp (G x))
      (g*(E0*Real.exp (G z))) z := by
  convert hG.exp.const_mul E0 using 1
  ring

theorem energy_integrating_factor_derivative {E G : ℝ → ℝ} {g z : ℝ}
    (hE : HasDerivAt E (g*E z) z) (hG : HasDerivAt G g z) :
    HasDerivAt (fun x => E x/Real.exp (G x)) 0 z := by
  convert hE.div hG.exp (Real.exp_ne_zero _) using 1
  ring_nf

theorem momentum_ratio_derivative {P E : ℝ → ℝ} {g z : ℝ}
    (hP : HasDerivAt P (g*P z) z) (hE : HasDerivAt E (g*E z) z) (hE0 : E z ≠ 0) :
    HasDerivAt (fun x => P x/E x) 0 z := by
  convert hP.div hE hE0 using 1
  ring_nf

theorem centroid_derivative {M P E : ℝ → ℝ} {b g z : ℝ}
    (hM : HasDerivAt M (b*P z+g*M z) z) (hE : HasDerivAt E (g*E z) z)
    (hE0 : E z ≠ 0) :
    HasDerivAt (fun x => M x/E x) (b*(P z/E z)) z := by
  convert hM.div hE hE0 using 1
  field_simp
  ring

def uniformMGF (a : ℝ) : ℝ := if a = 0 then 1 else (Real.exp a-Real.exp (-a))/(2*a)

theorem uniform_mgf_integral (a : ℝ) :
    (∫ x : ℝ in (-1)..1, Real.exp (a*x)/2) = uniformMGF a := by
  by_cases ha : a = 0
  · subst a
    norm_num [uniformMGF]
  · have hd : ∀ x : ℝ, HasDerivAt (fun y : ℝ => Real.exp (a*y)/(2*a))
        (Real.exp (a*x)/2) x := by
      intro x
      convert (((hasDerivAt_id x).const_mul a).exp.div_const (2*a)) using 1
      · funext y
        simp only [id_eq]
      · simp only [id_eq]
        field_simp
    have hint : IntervalIntegrable (fun x : ℝ => Real.exp (a*x)/2) MeasureTheory.volume (-1) 1 := by
      apply Continuous.intervalIntegrable
      fun_prop
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x) hint]
    simp only [mul_one, mul_neg, uniformMGF, ha, ↓reduceIte]
    ring

theorem uniform_second_moment : (∫ x : ℝ in (-1)..1, x^2/2) = 1/3 := by
  have hd : ∀ x : ℝ, HasDerivAt (fun y : ℝ => y^3/6) (x^2/2) x := by
    intro x
    convert ((hasDerivAt_id x).pow 3).div_const 6 using 1
    all_goals norm_num [id_eq, Pi.pow_apply]
    all_goals ring
  have hint : IntervalIntegrable (fun x : ℝ => x^2/2) MeasureTheory.volume (-1) 1 := by
    apply Continuous.intervalIntegrable
    fun_prop
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x) hint]
  norm_num

theorem uniform_fourth_moment : (∫ x : ℝ in (-1)..1, x^4/2) = 1/5 := by
  have hd : ∀ x : ℝ, HasDerivAt (fun y : ℝ => y^5/10) (x^4/2) x := by
    intro x
    convert ((hasDerivAt_id x).pow 5).div_const 10 using 1
    all_goals norm_num [id_eq, Pi.pow_apply]
    all_goals ring
  have hint : IntervalIntegrable (fun x : ℝ => x^4/2) MeasureTheory.volume (-1) 1 := by
    apply Continuous.intervalIntegrable
    fun_prop
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x) hint]
  norm_num

theorem uniform_energy_mean (a E0 : ℝ) :
    (∫ x : ℝ in (-1)..1, E0*Real.exp (a*x)/2) = E0*uniformMGF a := by
  calc
    _ = ∫ x : ℝ in (-1)..1, E0*(Real.exp (a*x)/2) := by congr 1; funext x; ring
    _ = E0*(∫ x : ℝ in (-1)..1, Real.exp (a*x)/2) := intervalIntegral.integral_const_mul _ _
    _ = _ := by rw [uniform_mgf_integral]

theorem uniform_energy_second_moment (a E0 : ℝ) :
    (∫ x : ℝ in (-1)..1, (E0*Real.exp (a*x))^2/2) = E0^2*uniformMGF (2*a) := by
  have hexp (x : ℝ) : Real.exp (a*x)^2 = Real.exp ((2*a)*x) := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  calc
    _ = ∫ x : ℝ in (-1)..1, E0^2*Real.exp ((2*a)*x)/2 := by
      congr 1
      funext x
      rw [mul_pow, hexp]
    _ = _ := uniform_energy_mean (2*a) (E0^2)

theorem weighted_quadratic_contraction {Ω ι : Type*} [Fintype Ω] [Fintype ι]
    (p : Ω → ℝ) (xi : Ω → ι → ℝ) (H : ι → ι → ℝ) :
    (∑ w, p w*(∑ i, ∑ j, H i j*xi w i*xi w j))
      = ∑ i, ∑ j, H i j*(∑ w, p w*xi w i*xi w j) := by
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  apply Finset.sum_congr rfl
  intro w hw
  ring

section Fidelity
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℂ V]

def orthResidual (s u : V) : V := u - (inner ℂ s u / inner ℂ s s) • s

theorem orthResidual_orthogonal (s u : V) (hs : s ≠ 0) :
    inner ℂ s (orthResidual s u) = 0 := by
  unfold orthResidual
  rw [inner_sub_right, inner_smul_right]
  rw [div_mul_cancel₀ _ (inner_self_ne_zero.mpr hs)]
  exact sub_self _

theorem orthResidual_affine (s v : V) (e : ℂ) (hs : s ≠ 0) :
    orthResidual s (s+e • v) = e • orthResidual s v := by
  unfold orthResidual
  rw [inner_add_right, inner_smul_right, add_div, div_self (inner_self_ne_zero.mpr hs),
    mul_div_assoc, add_smul, one_smul, smul_sub, smul_smul]
  abel

theorem fidelity_projection_identity (s u : V) (hs : s ≠ 0) (hu : u ≠ 0) :
    1 - ‖inner ℂ s u‖^2/(‖s‖^2*‖u‖^2)
      = ‖orthResidual s u‖^2/‖u‖^2 := by
  let c := inner ℂ s u / inner ℂ s s
  have horth := orthResidual_orthogonal s u hs
  have hdecomp : u = c • s + orthResidual s u := by
    dsimp [c, orthResidual]
    abel
  have hnorm : ‖u‖^2 = ‖c • s‖^2+‖orthResidual s u‖^2 := by
    conv_lhs => rw [hdecomp, norm_add_sq (𝕜 := ℂ)]
    rw [inner_smul_left, horth]
    simp
  have hip : inner ℂ s u = c * inner ℂ s s := by
    dsimp [c]
    rw [div_mul_cancel₀ _ (inner_self_ne_zero.mpr hs)]
  have hprod : ‖inner ℂ s u‖^2 = ‖c • s‖^2 * ‖s‖^2 := by
    rw [hip, norm_mul, inner_self_eq_norm_sq_to_K, norm_pow, norm_smul]
    change (‖c‖ * ‖(‖s‖ : ℂ)‖^2)^2 = (‖c‖*‖s‖)^2 * ‖s‖^2
    rw [Complex.norm_real, Real.norm_eq_abs, abs_norm]
    ring
  have hs0 : ‖s‖ ≠ 0 := norm_ne_zero_iff.mpr hs
  have hu0 : ‖u‖ ≠ 0 := norm_ne_zero_iff.mpr hu
  rw [hprod]
  field_simp
  nlinarith [hnorm]

theorem quadratic_response_nonnegative (s v : V) (hs : s ≠ 0) :
    0 ≤ ‖orthResidual s v‖^2 / ‖s‖^2 := by positivity

theorem affine_fidelity_exact (s v : V) (e : ℝ) (hs : s ≠ 0)
    (h : s+(e : ℂ) • v ≠ 0) :
    1 - ‖inner ℂ s (s+(e : ℂ) • v)‖^2/(‖s‖^2*‖s+(e : ℂ) • v‖^2)
      = e^2*‖orthResidual s v‖^2/‖s+(e : ℂ) • v‖^2 := by
  rw [fidelity_projection_identity s _ hs h, orthResidual_affine s v _ hs,
    norm_smul, mul_pow, Complex.norm_real, Real.norm_eq_abs, sq_abs]

end Fidelity
end SolitonUQ

#print axioms SolitonUQ.residual_identity
#print axioms SolitonUQ.gain_primitive_derivative
#print axioms SolitonUQ.defect_derivative
#print axioms SolitonUQ.multiplicative_defect_slope
#print axioms SolitonUQ.amplitude_gauge
#print axioms SolitonUQ.gain_free_clock_residual
#print axioms SolitonUQ.local_energy_jet
#print axioms SolitonUQ.local_momentum_jet
#print axioms SolitonUQ.cubic_directional_expansion
#print axioms SolitonUQ.exponential_energy_derivative
#print axioms SolitonUQ.energy_integrating_factor_derivative
#print axioms SolitonUQ.momentum_ratio_derivative
#print axioms SolitonUQ.centroid_derivative
#print axioms SolitonUQ.uniform_mgf_integral
#print axioms SolitonUQ.uniform_second_moment
#print axioms SolitonUQ.uniform_fourth_moment
#print axioms SolitonUQ.uniform_energy_mean
#print axioms SolitonUQ.uniform_energy_second_moment
#print axioms SolitonUQ.weighted_quadratic_contraction
#print axioms SolitonUQ.orthResidual_orthogonal
#print axioms SolitonUQ.orthResidual_affine
#print axioms SolitonUQ.fidelity_projection_identity
#print axioms SolitonUQ.quadratic_response_nonnegative
#print axioms SolitonUQ.affine_fidelity_exact
