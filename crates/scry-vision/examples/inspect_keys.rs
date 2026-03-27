// Quick utility to inspect safetensors key names and shapes.
// cargo run -p scry-vision --features safetensors --example inspect_keys -- <path>

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect_keys <path.safetensors>");
    let data = std::fs::read(&path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

    let mut names: Vec<_> = tensors.names().into_iter().collect();
    names.sort();
    for name in &names {
        let info = tensors.tensor(name).unwrap();
        println!("{:50} {:?}  {:?}", name, info.shape(), info.dtype());
    }
    println!("\nTotal: {} tensors", names.len());
}
