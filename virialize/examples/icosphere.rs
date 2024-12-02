fn main() {
    let divisions = 0;
    let icosphere = hexasphere::shapes::IcoSphere::new(divisions, |_| ());

    let indices = icosphere.get_all_indices();
    let vertices = icosphere.raw_points();

    println!("Indices: {:?}", indices);

    println!("Vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        println!("{} [{}, {}, {}]", i, vertex.x, vertex.y, vertex.z);
    }

    println!("Faces by index:");
    for (i, triangle) in indices.chunks(3).enumerate() {
        println!("{} [{}, {}, {}]", i, triangle[0], triangle[1], triangle[2],);
    }
    hexasphere::AdjacencyBuilder::new(1).add_indices(&indices);
    //     .build()
    //     .unwrap()
    //     .print();
}
