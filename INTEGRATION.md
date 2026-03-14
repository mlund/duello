# Integrate Table6DAdaptive into Duello and Faunus

## Overview

Three repos involved: **icotable** (library), **duello** (table generator), **faunus-rs** (MC lookup).
Both duello and faunus depend on icotable via git.

## Resolution boundary handling

Each (R, ω) slab has its own angular resolution. Lookup uses nearest-bin rounding for R and ω
(same as `Table6DFlat`), so at the boundary where resolution drops (e.g. n_div=2 → n_div=1)
the transition is a hard switch. This is acceptable because the resolution only drops when the
angular gradient is already below threshold — the coarser mesh captures the smooth surface well.

Possible mitigations if needed:
- **Linear R blending**: at transition bins, interpolate between fine and coarse slab results
- **Overlap validation**: evaluate transition slabs at both resolutions, assert small difference

## Step 1: icotable — Add metadata + tail_energy to Table6DAdaptive

`Table6DAdaptive` currently lacks `metadata`, `tail_energy()`, and `validate_metadata()` that
`Table6DFlat` has and Faunus requires.

- **`src/adaptive.rs`** — Add `pub metadata: Option<TableMetadata>` field to `Table6DAdaptive<T>`.
  Add `tail_energy(&self, r: f64) -> f64` and `validate_metadata(&self) -> Result<()>` methods
  (same logic as `Table6DFlat`, reuse `TableMetadata` and `TailCorrectionTerm` from `flat.rs`).
  Update `AdaptiveBuilder::build()` to set `metadata: None`.
- **`src/lib.rs`** — Already re-exports `Table6DAdaptive`; no changes needed.

## Step 2: duello — Use AdaptiveBuilder in icoscan

**`src/icoscan.rs`** — Replace the current `Table6D` → `Table6DFlat::<f16>` workflow with `AdaptiveBuilder`:

1. Add fields to `ScanConfig`: `max_n_div: usize` (default 3), `gradient_threshold: f64` (default 10.0).
2. Replace `Table6D::from_resolution(...)` with `AdaptiveBuilder::new(rmin, rmax, dr, omega_step, max_n_div, gradient_threshold)`.
   Compute `omega_step` from `angle_resolution` the same way dihedral_angles are built now.
3. Outer loop over distances (short→long R): use `builder.current_n_vertices()` per R slice.
   Get vertex directions from `builder.vertex_directions(builder.current_level())`.
4. GPU batch path: for each R, for each omega, build `PoseParams` from vertex directions + `icotable::orient()`, collect into batch, call `backend.compute_energies()`, feed flat n_v×n_v array into `builder.set_slab(ri, oi, &energies)`.
5. After all omega slabs for one R: call `builder.finish_r_slice(ri)`.
6. After all R: `let table = builder.build()`. Attach metadata + tail correction. Call `table.save(path)`.
7. PMF/partition-function calculation: iterate over `table` slabs instead of `Table6D` accessors — or keep the existing PMF code alongside by also computing partition function from the slab data.

**`src/main.rs`** — Add `--max-ndiv` and `--gradient-threshold` CLI args to the `Scan` command, pass to `ScanConfig`.

**`src/lib.rs`** — Update re-exports if needed (add `AdaptiveBuilder`, `Table6DAdaptive`).

**`Cargo.toml`** — Point icotable to branch/commit with Step 1 changes (or use path dependency during dev).

## Step 3: faunus-rs — Support loading Table6DAdaptive

**`faunus/src/energy/tabulated6d.rs`**:

1. Replace `Entry.table: Arc<Table6DFlat<f16>>` with an enum:
   ```rust
   enum TableKind {
       Flat(Table6DFlat<f16>),
       Adaptive(Table6DAdaptive<f32>),
   }
   ```
   Implement shared accessors: `rmin`, `rmax`, `tail_energy(r)`, `lookup_boltzmann(r, omega, dir_a, dir_b, beta)`, `validate_metadata()`.
2. Loading: try `Table6DAdaptive::<f32>::load()` first; if deserialization fails, fall back to `Table6DFlat::<f16>::load()`. Or use file extension / config field to distinguish.
3. `pair_energy()` hot path is unchanged — it calls `lookup_boltzmann()` on whichever variant.

**`faunus/Cargo.toml`** — Point icotable to branch/commit with Step 1 changes.

## Step 4: CPU backend path in duello (optional)

The CPU (rayon) path currently uses `Table6D::get_icospheres()` to iterate vertex pairs.
Replace with explicit loop over `builder.vertex_directions()` × `builder.vertex_directions()`,
computing energy per pose via `backend.compute_energy()`, then `builder.set_slab()`.

## Files summary

| Repo | File | Change |
|------|------|--------|
| icotable | `src/adaptive.rs` | Add `metadata`, `tail_energy()`, `validate_metadata()` |
| duello | `src/icoscan.rs` | Replace Table6D with AdaptiveBuilder |
| duello | `src/main.rs` | Add `--max-ndiv`, `--gradient-threshold` CLI args |
| duello | `Cargo.toml` | Update icotable dependency |
| faunus-rs | `faunus/src/energy/tabulated6d.rs` | Add `TableKind` enum, auto-detect format |
| faunus-rs | `faunus/Cargo.toml` | Update icotable dependency |

## Verification

```
# icotable
cd ~/github/icotable && cargo test

# duello (after icotable changes)
cd ~/github/duello && cargo test

# faunus-rs (after icotable changes)
cd ~/github/faunus-rs && cargo test
```
