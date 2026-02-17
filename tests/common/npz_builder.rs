use std::io::Write;
use tempfile::NamedTempFile;

/// Build a .npy file in memory: magic + version + header + raw data.
pub fn build_npy(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {} }}",
        descr, shape_str
    );
    // Pad header to 64-byte alignment (v1: 10 bytes preamble)
    let preamble_len = 10;
    let total = preamble_len + header_dict.len() + 1; // +1 for newline
    let pad = ((total + 63) / 64) * 64 - total;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(pad));

    let mut npy = Vec::new();
    npy.extend_from_slice(b"\x93NUMPY");
    npy.push(1); // major version
    npy.push(0); // minor version
    let header_len = padded_header.len() as u16;
    npy.extend_from_slice(&header_len.to_le_bytes());
    npy.extend_from_slice(padded_header.as_bytes());
    npy.extend_from_slice(data);
    npy
}

/// Build a Fortran-order .npy file in memory.
pub fn build_npy_fortran(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': True, 'shape': {} }}",
        descr, shape_str
    );
    let preamble_len = 10;
    let total = preamble_len + header_dict.len() + 1;
    let pad = ((total + 63) / 64) * 64 - total;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(pad));

    let mut npy = Vec::new();
    npy.extend_from_slice(b"\x93NUMPY");
    npy.push(1);
    npy.push(0);
    let header_len = padded_header.len() as u16;
    npy.extend_from_slice(&header_len.to_le_bytes());
    npy.extend_from_slice(padded_header.as_bytes());
    npy.extend_from_slice(data);
    npy
}

/// Build an .npz file (ZIP of .npy entries) with STORED compression.
pub fn build_npz_file(entries: Vec<(&str, &str, &[usize], &[u8])>) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".npz").unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        for (name, descr, shape, data) in &entries {
            let npy_data = build_npy(descr, shape, data);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file(format!("{}.npy", name), options).unwrap();
            std::io::Write::write_all(&mut zip, &npy_data).unwrap();
        }
        zip.finish().unwrap();
    }
    file.flush().unwrap();
    file
}
