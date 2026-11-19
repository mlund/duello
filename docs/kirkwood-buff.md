# Kirkwood–Buff analysis of g(r)

`duello nbody` writes the mass-centre _g(r)_ with its Kirkwood–Buff integral _G₂₂_ and the structure factor it predicts, _S(0)_ = 1 + ρ_G₂₂_.
The same numbers appear in the web app and in the `manybody/rdf.csv` extract of a `.duello.zip`.

## Why not integrate g directly

_G₂₂_ = 4π∫(_g_ − 1)_r_²d_r_ is defined for an open, infinite system.
In a closed box of _N_ bodies, bodies gathered near one are missing far from it, so the running integral drifts instead of reaching a plateau.
For hard spheres at packing fraction 0.2 in a 51 Å box, the running integral at _L_/2 is half the converged value.

## Estimator

Following the radial route reviewed by [Simon et al. 2022](https://doi.org/10.1063/5.0106162):

1. **Closed-box correction** of _g_ ([Ganguly & van der Vegt 2013](https://doi.org/10.1021/ct301017q), Eq. 5, p. 1352), with Δ_N_(_r_) the excess number of bodies within _r_.
   duello normalises pair counts by _N_(_N_ − 1), so _g_ is first rescaled by (_N_ − 1)/_N_ to the ρ = _N_/_V_ convention the correction assumes.
2. **Finite-volume integral** _G^V_(_L_) over spheres of diameter _L_, weight 4π_r_²(1 − 3_x_/2 + _x_³/2), _x_ = _r_/_L_ ([Krüger et al. 2013](https://doi.org/10.1021/jz301992u), Table 1, p. 236).
   _G^V_ − _G_∞ falls off as 1/_L_ (p. 236).
3. **Extrapolation**: _G₂₂_ is the slope of _L_·_G^V_(_L_) against _L_ (Simon et al. 2022, pp. 130901-3–4), fitted over the outer half of _L_ up to _L_/2 of the box.
   The fit counts as converged when that range is at least 10 Å wide and _R_² ≥ 0.99, the thresholds of [Peroutka et al. 2026](https://doi.org/10.1021/acs.jcim.6c00344) (p. 6825).
   Here _R_² is the share of Σ(_L_·_G^V_)² the line explains, taken about zero rather than the mean: near the θ point _G₂₂_ ≈ 0 leaves _L_·_G^V_ level at _F_∞, which the usual _R_² would reject as no line at all.
   Otherwise _G₂₂_ and _S(0)_ are left empty. Passing is "not a guarantee" of accuracy (same page): check against a larger box or a longer run.

The integrals are bin-wise, with _g_ constant within each 0.5 Å bin and the weights integrated exactly.

## Views along r

| Column        | Quantity |
|---------------|----------|
| `w/kT`        | −ln _g_, empty where _g_ = 0 |
| `h·r²/Å²`     | (_g_ − 1)_r_², whose area is _G₂₂_/4π |
| `N`           | bodies within _r_ of one body, (_N_ − 1)/_V_ ∫4π_r_²_g_ d_r_ |
| `G22_raw/Å³`  | running 4π∫₀^_r_ (_g_ − 1)_r_²d_r_, uncorrected |
| `G22_V/Å³`    | corrected _G^V_(_L_) with _L_ = _r_ |

## Interpretation

Solvent is implicit (McMillan–Mayer), so _G₂₂_ is the solvent-averaged integral at fixed solvent chemical potential.
In the dilute limit _G₂₂_ → −2_B₂₂_ ([Hill 1955](https://doi.org/10.1063/1.1742068), Eq. 89, p. 634), which links it to `duello scan`.

## Checks

`cargo test --release -p duello --lib g22 -- --ignored` runs two simulations with analytic answers:

| System | Result | Reference |
|--------|--------|-----------|
| Hard spheres, φ = 0.2, 400 bodies | _S(0)_ = 0.213 | Percus–Yevick 0.209 |
| Square well, ρ∫_f_ = 0.03 | _G₂₂_ = 5550 Å³ | −2_B₂_ = 5120 Å³ (+8 %, mostly second-order terms) |

Without the correction and extrapolation, both are off by about a factor of two.
