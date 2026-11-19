// S(r): the share of a pair's energy summed exactly; the far map holds the
// rest. Quintic smoothstep from 1 at r1 to 0 at r2, as `Switch::near_weight`,
// so S′ and S″ vanish at both ends and the far part has no kink.
fn near_weight(r: f32) -> f32 {
    let t = clamp((r - grid.near_r1) / (grid.near_r2 - grid.near_r1), 0.0, 1.0);
    return 1.0 - t * t * t * (t * (6.0 * t - 15.0) + 10.0);
}
