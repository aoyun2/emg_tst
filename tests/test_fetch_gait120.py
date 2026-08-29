from __future__ import annotations

import hashlib
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from emg_tst import fetch_gait120


PAYLOAD = b"gait120-payload" * 5000


def _record(payload: bytes = PAYLOAD, name: str = "part.zip") -> dict:
    return {
        "name": name,
        "size": len(payload),
        "supplied_md5": hashlib.md5(payload).hexdigest(),
        "download_url": "https://example.invalid/part.zip",
    }


class _Response:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._offset = 0
        self.status = status

    def read(self, n: int) -> bytes:
        block = self._body[self._offset : self._offset + n]
        self._offset += len(block)
        return block

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FetchResumeTests(unittest.TestCase):
    """The unattended run depends on these paths; each one has bitten in practice."""

    def test_fresh_download_verifies_and_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen",
                return_value=_Response(PAYLOAD),
            ):
                out = fetch_gait120._download(_record(), downloads)
            self.assertEqual(out.read_bytes(), PAYLOAD)
            self.assertFalse(out.with_suffix(out.suffix + ".part").exists())

    def test_partial_file_resumes_from_its_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            part = downloads / "part.zip.part"
            part.write_bytes(PAYLOAD[:1000])

            captured: dict = {}

            def fake_urlopen(request, timeout=None):
                captured["range"] = request.get_header("Range")
                return _Response(PAYLOAD[1000:], status=206)

            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                out = fetch_gait120._download(_record(), downloads)
            self.assertEqual(captured["range"], "bytes=1000-")
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_oversized_partial_file_is_discarded_not_resumed(self) -> None:
        """A stale-offset resume double-writes; the result is not a valid prefix."""
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            part = downloads / "part.zip.part"
            part.write_bytes(PAYLOAD + b"\x00" * 7340032)

            def fake_urlopen(request, timeout=None):
                # A restart must ask for the whole file, not a range.
                self.assertIsNone(request.get_header("Range"))
                return _Response(PAYLOAD)

            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                out = fetch_gait120._download(_record(), downloads)
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_complete_partial_file_is_verified_without_a_range_request(self) -> None:
        """Asking for bytes past the end returns HTTP 416, which killed a run."""
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            (downloads / "part.zip.part").write_bytes(PAYLOAD)

            def fail(request, timeout=None):
                raise AssertionError("must not request a completed transfer")

            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen", side_effect=fail
            ):
                out = fetch_gait120._download(_record(), downloads)
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_http_416_is_treated_as_a_finished_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            # One byte short, so the size check runs after the 416 is absorbed.
            (downloads / "part.zip.part").write_bytes(PAYLOAD[:-1])

            def fake_urlopen(request, timeout=None):
                raise urllib.error.HTTPError(
                    "u", 416, "Range Not Satisfiable", {}, None
                )

            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                with self.assertRaises(RuntimeError):
                    fetch_gait120._download(_record(), downloads)

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            corrupt = b"x" * len(PAYLOAD)
            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen",
                return_value=_Response(corrupt),
            ):
                with self.assertRaises(RuntimeError):
                    fetch_gait120._download(_record(), downloads)

    def test_already_downloaded_file_is_not_refetched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp)
            (downloads / "part.zip").write_bytes(PAYLOAD)

            def fail(request, timeout=None):
                raise AssertionError("must not re-download a verified file")

            with mock.patch.object(
                fetch_gait120.urllib.request, "urlopen", side_effect=fail
            ):
                out = fetch_gait120._download(_record(), downloads)
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_archive_member_escaping_the_destination_is_refused(self) -> None:
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dest"
            dest.mkdir()
            archive = Path(tmp) / "evil.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escaped.txt", "no")
            with self.assertRaises(RuntimeError):
                fetch_gait120._extract(archive, dest)


if __name__ == "__main__":
    unittest.main()
