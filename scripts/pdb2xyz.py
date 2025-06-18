#!/usr/bin/env python

# This coarse grains a PDB file to a XYZ file with one bead per amino acid.

import mdtraj as md


def convert_pdb(pdb_file, output_xyz_file):
    """Convert PDB to coarse grained XYZ file; one bead per amino acid"""
    traj = md.load_pdb(pdb_file, frame=0)
    residues = []
    for res in traj.topology.residues:
        if not res.is_protein:
            continue

        cm = [0.0, 0.0, 0.0]  # residue mass center
        mw = 0.0  # residue weight
        for a in res.atoms:
            # Add N-terminal
            if res.index == 0 and a.index == 0 and a.name == "N":
                print("Adding N-terminal bead")
                residues.append(dict(name="NTR", cm=traj.xyz[0][a.index] * 10))

            # Add C-terminal
            if a.name == "OXT":
                print("Adding C-terminal bead")
                residues.append(dict(name="CTR", cm=traj.xyz[0][a.index] * 10))

            # Add coarse grained bead
            cm = cm + a.element.mass * traj.xyz[0][a.index]
            mw = mw + a.element.mass

        residues.append(dict(name=res.name, cm=cm / mw * 10))

        if "sidechains" in sys.argv:
            side_chain = add_sidechains(traj, res)
            if side_chain is not None:
                residues.append(side_chain)

    with open(output_xyz_file, "w") as f:
        f.write(f"{len(residues)}\n")
        f.write(f"Converted with Duello pdb2xyz.py using input file {pdb_file}\n")
        for i in residues:
            f.write(f"{i['name']} {i['cm'][0]:.3f} {i['cm'][1]:.3f} {i['cm'][2]:.3f}\n")
    print(f"Converted {pdb_file} -> {output_xyz_file} with {len(residues)} residues.")


def add_sidechains(traj, res):
    for atom in res.atoms:
        if res.name == "ASP" and atom.name == "OD1":
            return dict(name="COO-", cm=traj.xyz[0][atom.index] * 10)
        elif res.name == "GLU" and atom.name == "OE1":
            return dict(name="COO-", cm=traj.xyz[0][atom.index] * 10)
        elif res.name == "ARG" and atom.name == "CZ":
            return dict(name="GDN+", cm=traj.xyz[0][atom.index] * 10)
        elif res.name == "LYS" and atom.name == "NZ":
            return dict(name="NH3+", cm=traj.xyz[0][atom.index] * 10)
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pdb2xyz.py input.pdb output.xyz [--sidechains]")
        sys.exit(1)

    sidechains = "--sidechains" in sys.argv
    pdb_file = sys.argv[1]
    output_xyz_file = sys.argv[2]

    # Pass the sidechains flag via sys.argv for compatibility with existing code
    if sidechains and "sidechains" not in sys.argv:
        sys.argv.append("sidechains")

    convert_pdb(pdb_file, output_xyz_file)
