use duello::IcoSphere;
use hexasphere::AdjacencyBuilder;
use std::io::Write;
fn main() {
    let n_divisions = 10;
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

    let face_area = |a: usize, b: usize, c: usize| {
        let a = vertices[a];
        let b = vertices[b];
        let c = vertices[c];
        let ab = b - a;
        let ac = c - a;
        0.5 * ab.cross(ac).length()
    };

    let mut weights = Vec::with_capacity(vertices.len());

    for (i, nb) in adjency.iter().enumerate() {
        let mut areas = Vec::with_capacity(nb.len());
        areas.push(face_area(i, *nb.first().unwrap(), *nb.last().unwrap()) as f64);
        println!("Vertex {}: {} neighbors", i, nb.len());
        for j in 0..nb.len() - 1 {
            let area = face_area(i, nb[j], nb[j + 1]);
            areas.push(area as f64);
        }
        let avg_area = areas.iter().sum::<f64>() / areas.len() as f64;
        weights.push(avg_area);
        // println!("Average area: {}", avg_area);
    }
    let scale = vertices.len() as f64 / weights.iter().sum::<f64>();
    weights.iter_mut().for_each(|w| *w *= scale);

    let mut weight_file =
        std::fs::File::create(format!("weights-n{n_divisions}")).expect("Failed to create file");

    for weight in weights.iter() {
        writeln!(weight_file, "{weight}").expect("Failed to write to file");
    }

    println!("\nVertex weights:\n {:?}", weights);
}
