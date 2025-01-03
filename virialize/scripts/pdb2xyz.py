#!/usr/bin/env python

# This coarse grains a PDB file to a XYZ file with one bead per amino acid.

import mdtraj as md

def convert_pdb(pdb_file, output_xyz_file):
    ''' Convert PDB to coarse grained XYZ file; one bead per amino acid '''
    traj = md.load_pdb(pdb_file, frame=0)
    residues = []
    for res in traj.topology.residues:
        if not res.is_protein:
            continue
        cm = [0.0, 0.0, 0.0]  # residue mass center
        mw = 0.0 # residue weight
        for a in res.atoms:
            cm = cm + a.element.mass * traj.xyz[0][a.index]
            mw = mw + a.element.mass
        cm = cm / mw * 10.0
        residues.append(dict(name=res.name, cm=cm))
    with open(output_xyz_file, "w") as f:
        f.write(f'{len(residues)}\n')
        for i in residues:
            f.write(f"{i['name']} {i['cm'][0]} {i['cm'][1]} {i['cm'][2]}\n")
    print(f"Converted {pdb_file} -> {output_xyz_file} with {len(residues)} residues.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pdb2xyz.py input.pdb output.xyz")
        sys.exit(1)
    convert_pdb(sys.argv[1], sys.argv[2])