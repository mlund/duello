// Example how we can use lazy_static to create a static IcoSphere at run-time

use duello::IcoSphere;
use lazy_static::lazy_static;

lazy_static! {
    /// IcoSphere with 0 subdivisions -> icosahedron
    static ref ICOSAHEDRON: IcoSphere = IcoSphere::new(0, |_| ());
    /// Array of IcoSpheres with different subdivisions
    static ref SUBDIVIDED: [IcoSphere; 6] = [
        IcoSphere::new(0, |_| ()),
        IcoSphere::new(1, |_| ()),
        IcoSphere::new(2, |_| ()),
        IcoSphere::new(3, |_| ()),
        IcoSphere::new(4, |_| ()),
        IcoSphere::new(5, |_| ()),
    ];
}

fn main() {
    // Print number of vertices for all elements in SUBDIVIDED
    for (i, icosphere) in SUBDIVIDED.iter().enumerate() {
        println!(
            "Subdivisions: {}, Vertices: {}",
            i,
            icosphere.raw_points().len()
        );
    }
}
