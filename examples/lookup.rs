use duello::IcoSphere;
use hexasphere::AdjacencyBuilder;
fn main() {
    let n_divisions = 1;
    let icosphere = IcoSphere::new(n_divisions, |_| ());

    let indices = icosphere.get_all_indices();
    let vertices = icosphere.raw_points();

    println!("Indices: {:?}", indices);

    println!("\nVertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        println!("{} [{}, {}, {}]", i, vertex.x, vertex.y, vertex.z);
    }

    println!("\nFaces by index:");
    for (i, triangle) in indices.chunks(3).enumerate() {
        println!("{} [{}, {}, {}]", i, triangle[0], triangle[1], triangle[2],);
    }

    println!("\nFaces areas:");
    for (n, triangle) in indices.chunks(3).enumerate() {
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let ab = b - a;
        let ac = c - a;
        println!("Area of face {}: {}", n, 0.5 * ab.cross(ac).length());
    }

    let mut ab = AdjacencyBuilder::new(vertices.len());
    ab.add_indices(&indices);
    let adjency = ab.finish();
    println!("\nVertex neighborlist:\n(The result preserves winding: the resulting array is wound around the center vertex in the same way that the source triangles were wound.):\n {:?}", adjency);
}
