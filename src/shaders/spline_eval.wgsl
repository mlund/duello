// Spline lookup shared by kernels that bind `spline_coeffs` and `spline_params`.
// Prepended: `const OVERLAP_ENERGY` (energy::OVERLAP_ENERGY).

// Below its table start a pair's model is undefined, so the pose is an
// overlap: its energy is OVERLAP_ENERGY. Without this, exact Yukawa
// outweighs the linearly extrapolated wall and opposite charges collapse.
fn inside_table(params: SplineParams, r: f32) -> bool {
    return r < params.r_min;
}

fn spline_energy(pair_idx: u32, r: f32) -> f32 {
    return spline_energy_at(spline_params[pair_idx], r);
}

// Evaluate spline energy using Horner's method
fn spline_energy_at(params: SplineParams, r: f32) -> f32 {

    if r >= params.r_max {
        return 0.0;
    }

    // Linear extrapolation below r_min (branchless)
    let extrap_dist = max(params.r_min - r, 0.0);

    // Grid-type-specific index computation (clamped to [r_min, r_max])
    let ie = spline_index_eps(params, r);
    let idx = u32(ie.x);
    let frac = ie.y;

    let c = spline_coeffs[params.coeff_offset + idx].u;

    // Horner's method: a0 + frac*(a1 + frac*(a2 + frac*a3))
    let u_spline = c.x + frac * (c.y + frac * (c.z + frac * c.w));

    // Slope = 2·r_min·f_at_rmin (derivative of U w.r.t. r, from -dU/d(r²) convention)
    return u_spline + 2.0 * params.r_min * params.f_at_rmin * extrap_dist;
}

