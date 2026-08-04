//! CBOR codec: decoding must never panic, and anything the (deterministic)
//! decoder accepts must re-encode and re-decode cleanly.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = ztensor::format::cbor::decode(data) {
        let reencoded = ztensor::format::cbor::encode(&value).expect("accepted value must encode");
        ztensor::format::cbor::decode(&reencoded).expect("re-encoded value must decode");
    }
});
