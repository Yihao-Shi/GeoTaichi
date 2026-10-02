"""Read GeoTaichi VTU results without optional VTK or meshio dependencies."""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path


_VTK_TYPES = {
    "Int8": ("b", 1),
    "UInt8": ("B", 1),
    "Int16": ("h", 2),
    "UInt16": ("H", 2),
    "Int32": ("i", 4),
    "UInt32": ("I", 4),
    "Int64": ("q", 8),
    "UInt64": ("Q", 8),
    "Float32": ("f", 4),
    "Float64": ("d", 8),
}
_FRAME_FILE = re.compile(r"^(.*\D)(\d+)\.vtu$", re.IGNORECASE)


@dataclass(frozen=True)
class VTUFrame:
    """Blender-compatible geometry decoded from one VTU file."""

    points: tuple[tuple[float, float, float], ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]

    @property
    def is_point_cloud(self) -> bool:
        return not self.edges and not self.faces


@dataclass(frozen=True)
class VTUSequence:
    """A numbered VTU sequence sharing one directory and filename prefix."""

    directory: Path
    prefix: str
    frame_numbers: tuple[int, ...]
    frame_width: int

    def path_at(self, index: int) -> Path:
        frame_number = self.frame_numbers[index]
        return self.directory / (self.prefix + str(frame_number).zfill(self.frame_width) + ".vtu")


def discover_vtu_sequences(directory: str | Path, recursive: bool = True) -> tuple[VTUSequence, ...]:
    """Group numbered ``*.vtu`` files into deterministic animation sequences."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError("VTU result directory does not exist: %s" % root)
    paths = root.rglob("*.vtu") if recursive else root.glob("*.vtu")
    groups: dict[tuple[Path, str, int], set[int]] = {}
    for path in paths:
        if path.name.startswith("._"):
            continue
        match = _FRAME_FILE.match(path.name)
        if match is None:
            continue
        prefix, digits = match.groups()
        groups.setdefault((path.parent, prefix, len(digits)), set()).add(int(digits))
    sequences = [
        VTUSequence(directory=parent, prefix=prefix, frame_numbers=tuple(sorted(frames)), frame_width=width)
        for (parent, prefix, width), frames in groups.items()
    ]
    return tuple(sorted(sequences, key=lambda item: (str(item.directory), item.prefix, item.frame_width)))


def read_vtu_geometry(path: str | Path) -> VTUFrame:
    """Decode points and cells from ASCII, raw-appended, or VTK binary VTU."""
    source = Path(path)
    raw = source.read_bytes()
    root, appended = _parse_document(raw, source)
    byte_order = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_type = root.get("header_type", "UInt32")
    compressor = root.get("compressor", "")
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError("VTU contains no UnstructuredGrid/Piece: %s" % source)

    points_array = piece.find("./Points/DataArray")
    if points_array is None:
        raise ValueError("VTU contains no point coordinates: %s" % source)
    point_values = _read_array(points_array, appended, byte_order, header_type, compressor)
    components = int(points_array.get("NumberOfComponents", "3"))
    points = _points(point_values, components)

    cell_arrays = {element.get("Name", "").lower(): element for element in piece.findall("./Cells/DataArray")}
    if not {"connectivity", "offsets", "types"}.issubset(cell_arrays):
        return VTUFrame(points=points, edges=(), faces=())
    connectivity = tuple(
        int(value) for value in _read_array(cell_arrays["connectivity"], appended, byte_order, header_type, compressor)
    )
    offsets = tuple(
        int(value) for value in _read_array(cell_arrays["offsets"], appended, byte_order, header_type, compressor)
    )
    cell_types = tuple(
        int(value) for value in _read_array(cell_arrays["types"], appended, byte_order, header_type, compressor)
    )
    edges, faces = _surface_geometry(connectivity, offsets, cell_types)
    return VTUFrame(points=points, edges=edges, faces=faces)


def _parse_document(raw: bytes, source: Path) -> tuple[ET.Element, bytes | None]:
    appended_start = raw.find(b"<AppendedData")
    if appended_start < 0:
        return ET.fromstring(raw), None
    opening_end = raw.find(b">", appended_start)
    closing_start = raw.rfind(b"</AppendedData>")
    marker = raw.find(b"_", opening_end + 1, closing_start)
    if opening_end < 0 or closing_start < 0 or marker < 0:
        raise ValueError("invalid VTU AppendedData section: %s" % source)
    root = ET.fromstring(raw[: marker + 1] + raw[closing_start:])
    appended_element = root.find("./AppendedData")
    encoding = appended_element.get("encoding", "raw").lower() if appended_element is not None else "raw"
    payload = raw[marker + 1 : closing_start]
    if encoding == "base64":
        payload = _decode_base64(payload.decode("ascii"))
    elif encoding != "raw":
        raise ValueError("unsupported VTU AppendedData encoding: %s" % encoding)
    return root, payload


def _read_array(element, appended, byte_order: str, header_type: str, compressor: str):
    vtk_type = element.get("type", "Float32")
    if vtk_type not in _VTK_TYPES:
        raise ValueError("unsupported VTU scalar type: %s" % vtk_type)
    data_format = element.get("format", "ascii").lower()
    if data_format == "ascii":
        text = element.text or ""
        converter = float if vtk_type.startswith("Float") else int
        return tuple(converter(value) for value in text.split())
    if data_format == "appended":
        if appended is None:
            raise ValueError("VTU DataArray references missing AppendedData")
        encoded = appended[int(element.get("offset", "0")) :]
    elif data_format == "binary":
        encoded = _decode_base64(element.text or "")
    else:
        raise ValueError("unsupported VTU DataArray format: %s" % data_format)

    if compressor:
        payload = _decompress_vtk(encoded, byte_order, header_type)
    else:
        header_format, header_size = _VTK_TYPES[header_type]
        if len(encoded) < header_size:
            raise ValueError("truncated VTU binary array header")
        payload_size = struct.unpack_from(byte_order + header_format, encoded)[0]
        payload = encoded[header_size : header_size + payload_size]
    scalar_format, scalar_size = _VTK_TYPES[vtk_type]
    if len(payload) % scalar_size:
        raise ValueError("VTU binary array has an incomplete scalar value")
    return tuple(value[0] for value in struct.iter_unpack(byte_order + scalar_format, payload))


def _decode_base64(text: str) -> bytes:
    """Decode VTK's optionally concatenated, independently padded blocks."""
    encoded = "".join(text.split())
    decoded = bytearray()
    while encoded:
        padding = encoded.find("=")
        if padding < 0:
            decoded.extend(base64.b64decode(encoded))
            break
        block_end = (padding // 4 + 1) * 4
        decoded.extend(base64.b64decode(encoded[:block_end]))
        encoded = encoded[block_end:]
    return bytes(decoded)


def _decompress_vtk(encoded: bytes, byte_order: str, header_type: str) -> bytes:
    header_format, word_size = _VTK_TYPES[header_type]
    if len(encoded) < 3 * word_size:
        raise ValueError("truncated VTK compression header")
    number_of_blocks, block_size, last_block_size = struct.unpack_from(byte_order + header_format * 3, encoded)
    header_size = (3 + number_of_blocks) * word_size
    compressed_sizes = struct.unpack_from(byte_order + header_format * number_of_blocks, encoded, 3 * word_size)
    position = header_size
    chunks = []
    for compressed_size in compressed_sizes:
        block = encoded[position : position + compressed_size]
        chunks.append(zlib.decompress(block))
        position += compressed_size
    payload = b"".join(chunks)
    expected_size = (number_of_blocks - 1) * block_size + last_block_size if number_of_blocks else 0
    if len(payload) != expected_size:
        raise ValueError("VTK compressed payload size does not match its header")
    return payload


def _points(values, components: int) -> tuple[tuple[float, float, float], ...]:
    if components < 1:
        raise ValueError("VTU point component count must be positive")
    points = []
    for start in range(0, len(values), components):
        coordinates = values[start : start + components]
        if len(coordinates) != components:
            raise ValueError("VTU point array has an incomplete coordinate")
        points.append(tuple(float(coordinates[index]) if index < components else 0.0 for index in range(3)))
    return tuple(points)


def _surface_geometry(connectivity, offsets, cell_types):
    edges = []
    surface_faces = []
    boundary_faces = {}
    start = 0
    for offset, cell_type in zip(offsets, cell_types):
        cell = tuple(connectivity[start:offset])
        start = offset
        cell_edges, cell_surface, cell_volume = _cell_geometry(cell_type, cell)
        edges.extend(cell_edges)
        surface_faces.extend(cell_surface)
        for face in cell_volume:
            key = tuple(sorted(face))
            if key in boundary_faces:
                boundary_faces[key] = None
            else:
                boundary_faces[key] = face

    surface_faces.extend(face for face in boundary_faces.values() if face is not None)
    return tuple(edges), tuple(surface_faces)


def _cell_geometry(cell_type: int, cell: tuple[int, ...]):
    if cell_type in {1, 2}:
        return (), (), ()
    if cell_type in {3, 21}:
        return ((cell[0], cell[1]),), (), ()
    if cell_type == 4:
        return tuple(zip(cell[:-1], cell[1:])), (), ()
    if cell_type == 5:
        return (), (cell[:3],), ()
    if cell_type == 6:
        triangles = []
        for index in range(len(cell) - 2):
            triangle = cell[index : index + 3]
            triangles.append(triangle if index % 2 == 0 else (triangle[1], triangle[0], triangle[2]))
        return (), tuple(triangles), ()
    if cell_type == 7:
        return (), (cell,), ()
    if cell_type == 8:
        return (), ((cell[0], cell[1], cell[3], cell[2]),), ()
    if cell_type in {9, 23}:
        return (), (cell[:4],), ()
    if cell_type in {10, 24}:
        a, b, c, d = cell[:4]
        return (), (), ((a, c, b), (a, b, d), (b, c, d), (c, a, d))
    if cell_type == 11:
        a, b, c, d, e, f, g, h = cell[:8]
        return (), (), ((a, c, d, b), (e, f, h, g), (a, e, g, c), (b, d, h, f), (a, b, f, e), (c, g, h, d))
    if cell_type in {12, 25}:
        a, b, c, d, e, f, g, h = cell[:8]
        return (), (), ((a, d, c, b), (e, f, g, h), (a, b, f, e), (b, c, g, f), (c, d, h, g), (d, a, e, h))
    if cell_type in {13, 26}:
        a, b, c, d, e, f = cell[:6]
        return (), (), ((a, c, b), (d, e, f), (a, b, e, d), (b, c, f, e), (c, a, d, f))
    if cell_type in {14, 27}:
        a, b, c, d, e = cell[:5]
        return (), (), ((a, d, c, b), (a, b, e), (b, c, e), (c, d, e), (d, a, e))
    return (), (), ()
