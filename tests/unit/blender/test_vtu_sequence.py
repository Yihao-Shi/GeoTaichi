from __future__ import annotations

import base64
import struct
import zlib

from blender.core.vtu import discover_vtu_sequences, read_vtu_geometry


def test_discovers_numbered_sequences_recursively(tmp_path):
    result = tmp_path / "output" / "vtks"
    result.mkdir(parents=True)
    for name in (
        "GraphicMPMParticle000002.vtu",
        "GraphicMPMParticle000000.vtu",
        "Phase2Particle000003.vtu",
        "not-a-sequence.vtu",
    ):
        (result / name).touch()

    sequences = discover_vtu_sequences(tmp_path)

    assert [(item.prefix, item.frame_numbers, item.frame_width) for item in sequences] == [
        ("GraphicMPMParticle", (0, 2), 6),
        ("Phase2Particle", (3,), 6),
    ]
    assert sequences[0].path_at(1).name == "GraphicMPMParticle000002.vtu"


def test_reads_ascii_tetrahedra_and_removes_the_shared_face(tmp_path):
    path = tmp_path / "FEM000000.vtu"
    path.write_text(
        """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" byte_order="LittleEndian">
  <UnstructuredGrid><Piece NumberOfPoints="5" NumberOfCells="2">
    <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
      0 0 0  1 0 0  0 1 0  0 0 1  0 0 -1
    </DataArray></Points>
    <Cells>
      <DataArray Name="connectivity" type="Int32" format="ascii">0 1 2 3  0 2 1 4</DataArray>
      <DataArray Name="offsets" type="Int32" format="ascii">4 8</DataArray>
      <DataArray Name="types" type="UInt8" format="ascii">10 10</DataArray>
    </Cells>
  </Piece></UnstructuredGrid>
</VTKFile>""",
        encoding="utf-8",
    )

    frame = read_vtu_geometry(path)

    assert len(frame.points) == 5
    assert len(frame.faces) == 6
    assert (0, 1, 2) not in {tuple(sorted(face)) for face in frame.faces}
    assert not frame.is_point_cloud


def test_reads_pyevtk_raw_appended_point_output(tmp_path):
    path = tmp_path / "GraphicMPMParticle000000.vtu"
    arrays = [
        struct.pack("<6d", 0.0, 0.0, 0.0, 1.0, 2.0, 3.0),
        struct.pack("<2i", 0, 1),
        struct.pack("<2i", 1, 2),
        struct.pack("<2B", 1, 1),
    ]
    blocks = [struct.pack("<Q", len(payload)) + payload for payload in arrays]
    offsets = []
    position = 0
    for block in blocks:
        offsets.append(position)
        position += len(block)
    xml = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" header_type="UInt64">
  <UnstructuredGrid><Piece NumberOfPoints="2" NumberOfCells="2">
    <Points><DataArray type="Float64" NumberOfComponents="3" format="appended" offset="%d"/></Points>
    <Cells>
      <DataArray Name="connectivity" type="Int32" format="appended" offset="%d"/>
      <DataArray Name="offsets" type="Int32" format="appended" offset="%d"/>
      <DataArray Name="types" type="UInt8" format="appended" offset="%d"/>
    </Cells>
  </Piece></UnstructuredGrid>
  <AppendedData encoding="raw">_""" % tuple(
        offsets
    )
    path.write_bytes(xml.encode("ascii") + b"".join(blocks) + b"</AppendedData></VTKFile>")

    frame = read_vtu_geometry(path)

    assert frame.points == ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert frame.is_point_cloud


def test_reads_meshio_style_inline_zlib_binary(tmp_path):
    path = tmp_path / "FEM000000.vtu"

    def compressed(payload):
        block = zlib.compress(payload)
        header = struct.pack("<4I", 1, 32768, len(payload), len(block))
        return base64.b64encode(header).decode("ascii") + base64.b64encode(block).decode("ascii")

    points = compressed(struct.pack("<9d", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    connectivity = compressed(struct.pack("<3q", 0, 1, 2))
    offsets = compressed(struct.pack("<q", 3))
    cell_types = compressed(struct.pack("<B", 5))
    path.write_text(
        """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid><Piece NumberOfPoints="3" NumberOfCells="1">
    <Points><DataArray type="Float64" NumberOfComponents="3" format="binary">%s</DataArray></Points>
    <Cells>
      <DataArray Name="connectivity" type="Int64" format="binary">%s</DataArray>
      <DataArray Name="offsets" type="Int64" format="binary">%s</DataArray>
      <DataArray Name="types" type="UInt8" format="binary">%s</DataArray>
    </Cells>
  </Piece></UnstructuredGrid>
</VTKFile>"""
        % (points, connectivity, offsets, cell_types),
        encoding="utf-8",
    )

    frame = read_vtu_geometry(path)

    assert frame.points[1] == (1.0, 0.0, 0.0)
    assert frame.faces == ((0, 1, 2),)
