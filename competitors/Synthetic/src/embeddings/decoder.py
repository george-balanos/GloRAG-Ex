import base64
import zlib
import struct
import numpy as np

def decode_vector(encoded: str, dim: int) -> np.ndarray:
    compressed = base64.b64decode(encoded)
    decompressed = zlib.decompress(compressed)
    vec = np.frombuffer(decompressed, dtype=np.float16)
    assert len(vec) == dim, f"Vector dim mismatch: got {len(vec)}, expected {dim}"
    return vec