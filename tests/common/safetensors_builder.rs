use std::io::Write;
use tempfile::NamedTempFile;

pub fn build_safetensors_file(
    tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
) -> NamedTempFile {
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let serialized = safetensors::tensor::serialize(views, &None).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();
    file
}
