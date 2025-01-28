// Example how we can use lazy_static to create a static IcoSphere at runtime
// Useful for globally sharing immutable data. Data is computed at runtime and stored in a static variable
// upon first access.

use duello::{make_vertices, IcoSphere, Vertex};
use lazy_static::lazy_static;

type Vertices = Vec<Vertex>;

lazy_static! {
    /// IcoSphere with 0 subdivisions -> icosahedron
    static ref ICOSAHEDRON: IcoSphere = IcoSphere::new(0, |_| ());
    /// IcoSpheres with increasing subdivisions starting from 0
    static ref VERTICES: Vec<Vertices> = make_subdivided_spheres(9);
}

fn make_subdivided_spheres(max_subdivisions: usize) -> Vec<Vertices> {
    Vec::from_iter((0..=max_subdivisions).map(|i| make_vertices(&IcoSphere::new(i, |_| ()))))
}

fn main() {
    for (i, subdivided) in VERTICES.iter().enumerate() {
        println!("Subdivisions: {}, Vertices: {}", i, subdivided.len());
    }
}
