// Utility functions for printing tensor metadata and tables
use comfy_table::{Table, Row, Cell, presets::UTF8_FULL, ContentArrangement};
use humansize::{format_size, DECIMAL};
use ztensor::TensorMetadata;

pub fn print_tensor_metadata(meta: &TensorMetadata) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(&meta.name)]));
    table.add_row(Row::from(vec![Cell::new("DType"), Cell::new(format!("{:?}", meta.dtype))]));
    table.add_row(Row::from(vec![Cell::new("Encoding"), Cell::new(format!("{:?}", meta.encoding))]));
    table.add_row(Row::from(vec![Cell::new("Layout"), Cell::new(format!("{:?}", meta.layout))]));
    let shape_str = if meta.shape.len() == 0 {
        "(scalar)".to_string()
    } else {
        format!("[{}]", meta.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    table.add_row(Row::from(vec![Cell::new("Offset"), Cell::new(meta.offset.to_string())]));
    table.add_row(Row::from(vec![Cell::new("Size (on-disk)"), Cell::new(format_size(meta.size, DECIMAL))]));
    if let Some(endian) = &meta.data_endianness {
        table.add_row(Row::from(vec![Cell::new("Data Endianness"), Cell::new(format!("{:?}", endian))]));
    }
    if let Some(cs) = &meta.checksum {
        table.add_row(Row::from(vec![Cell::new("Checksum"), Cell::new(cs)]));
    }
    if !meta.custom_fields.is_empty() {
        table.add_row(Row::from(vec![Cell::new("Custom fields"), Cell::new(format!("{:?}", meta.custom_fields))]));
    }
    println!("{}", table);
}

pub fn print_tensors_table(tensors: &[TensorMetadata]) {
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_HORIZONTAL_ONLY)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "#", "Name", "Shape", "DType", "Encoding", "Layout", "Offset", "Size", "On-disk Size"
    ]);
    for (i, meta) in tensors.iter().enumerate() {
        let shape_str = if meta.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!("[{}]", meta.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        };
        let dtype_str = format!("{:?}", meta.dtype);
        let encoding_str = format!("{:?}", meta.encoding);
        let layout_str = format!("{:?}", meta.layout);
        let offset_str = meta.offset.to_string();
        let numel = meta.shape.iter().product::<u64>();
        let size = match meta.dtype {
            ztensor::DType::Float64 | ztensor::DType::Int64 | ztensor::DType::Uint64 => 8 * numel,
            ztensor::DType::Float32 | ztensor::DType::Int32 | ztensor::DType::Uint32 => 4 * numel,
            ztensor::DType::BFloat16 | ztensor::DType::Float16  | ztensor::DType::Int16 | ztensor::DType::Uint16 => 2 * numel,
            ztensor::DType::Int8 | ztensor::DType::Uint8 | ztensor::DType::Bool => 1 * numel,
            _ => 0,
        };
        let size_str = if size > 0 { format_size(size, DECIMAL) } else { "?".to_string() };
        let on_disk_str = format_size(meta.size, DECIMAL);
        table.add_row(Row::from(vec![
            Cell::new(i),
            Cell::new(&meta.name),
            Cell::new(shape_str),
            Cell::new(dtype_str),
            Cell::new(encoding_str),
            Cell::new(layout_str),
            Cell::new(offset_str),
            Cell::new(size_str),
            Cell::new(on_disk_str),
        ]));
    }
    println!("{}", table);
}
