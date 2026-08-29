from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mocap_phys_eval import experts


PAYLOAD = b"expert-tarball-bytes" * 4000


class _Response:
    """urlopen stand-in whose body may end early, as a dropped connection does."""

    def __init__(self, body: bytes, *, content_length: int | None, status: int = 200):
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, n: int) -> bytes:
        block = self._body[self._offset : self._offset + n]
        self._offset += len(block)
        return block

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ExpertDownloadTruncationTests(unittest.TestCase):
    """A truncated transfer must never be promoted to the final name.

    urllib returns an empty chunk both when a file finishes and when the
    connection drops, so a read loop alone cannot tell them apart. Promoting a
    short file made the corruption permanent: the completed-file check accepted
    it on every later attempt and extraction failed forever.
    """

    def test_short_transfer_keeps_the_partial_file_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_1.tar.gz"
            short = PAYLOAD[: len(PAYLOAD) // 2]
            with mock.patch.object(
                experts.urllib.request, "urlopen",
                return_value=_Response(short, content_length=len(PAYLOAD)),
            ):
                with self.assertRaises(IOError):
                    experts._stream_download(
                        "https://example.invalid/experts_1.tar.gz", dst
                    )
            self.assertFalse(dst.exists(), "a short transfer must not be promoted")
            part = dst.with_suffix(dst.suffix + ".part")
            self.assertTrue(part.exists())
            self.assertEqual(part.stat().st_size, len(short))

    def test_errorless_oserror_is_not_mistaken_for_a_disk_full_error(self) -> None:
        """An OSError with no errno must be classified, not crash the classifier.

        The truncation error is raised without an errno, and errno is then None
        rather than absent, so a getattr default never fires. int(None) raised a
        TypeError inside the resume loop and escalated a recoverable dropped
        connection into a whole-stage failure.
        """
        self.assertFalse(experts._is_no_space_error(IOError("truncated")))
        self.assertFalse(experts._is_no_space_error(OSError()))
        self.assertFalse(experts._is_no_space_error(ValueError("not an OSError")))
        self.assertTrue(experts._is_no_space_error(OSError(28, "No space left")))
        self.assertTrue(experts._is_no_space_error(OSError(112, "Disk full")))

    def test_resume_loop_retries_a_truncated_transfer_without_escalating(self) -> None:
        """A drop must be absorbed here rather than surfacing to the caller."""
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_5.tar.gz"
            calls = {"n": 0}

            def fake_urlopen(request, timeout=None):
                if request.get_method() == "HEAD":
                    return _Response(b"", content_length=len(PAYLOAD))
                calls["n"] += 1
                if calls["n"] == 1:
                    # First attempt is cut off half way.
                    return _Response(
                        PAYLOAD[: len(PAYLOAD) // 2], content_length=len(PAYLOAD)
                    )
                start = int(request.headers.get("Range", "bytes=0-").split("=")[1].strip("-"))
                return _Response(
                    PAYLOAD[start:], content_length=len(PAYLOAD) - start, status=206
                )

            with mock.patch.object(
                experts.urllib.request, "urlopen", side_effect=fake_urlopen
            ), mock.patch.object(experts.time, "sleep", lambda _s: None):
                out = experts._download_expert_tarball(
                    "https://example.invalid/experts_5.tar.gz",
                    dst,
                    force=False,
                    timeout_s=30.0,
                )
            self.assertEqual(out.read_bytes(), PAYLOAD)
            self.assertGreaterEqual(calls["n"], 2)

    def test_complete_transfer_is_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_1.tar.gz"
            with mock.patch.object(
                experts.urllib.request, "urlopen",
                return_value=_Response(PAYLOAD, content_length=len(PAYLOAD)),
            ):
                out = experts._stream_download(
                    "https://example.invalid/experts_1.tar.gz", dst
                )
            self.assertEqual(out.read_bytes(), PAYLOAD)
            self.assertFalse(dst.with_suffix(dst.suffix + ".part").exists())

    def test_partial_file_resumes_from_its_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_1.tar.gz"
            dst.with_suffix(dst.suffix + ".part").write_bytes(PAYLOAD[:5000])
            seen: dict = {}

            def fake_urlopen(request, timeout=None):
                seen["range"] = request.headers.get("Range")
                return _Response(
                    PAYLOAD[5000:],
                    content_length=len(PAYLOAD) - 5000,
                    status=206,
                )

            with mock.patch.object(
                experts.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                out = experts._stream_download(
                    "https://example.invalid/experts_1.tar.gz", dst
                )
            self.assertEqual(seen["range"], "bytes=5000-")
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_previously_promoted_truncated_file_is_demoted_and_resumed(self) -> None:
        """The failure actually seen: a short file already sitting at the final name."""
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_2.tar.gz"
            dst.write_bytes(PAYLOAD[:5000])

            def fake_urlopen(request, timeout=None):
                if request.get_method() == "HEAD":
                    return _Response(b"", content_length=len(PAYLOAD))
                self.assertEqual(request.headers.get("Range"), "bytes=5000-")
                return _Response(
                    PAYLOAD[5000:], content_length=len(PAYLOAD) - 5000, status=206
                )

            with mock.patch.object(
                experts.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                out = experts._stream_download(
                    "https://example.invalid/experts_2.tar.gz", dst
                )
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_complete_file_at_published_size_is_not_refetched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_3.tar.gz"
            dst.write_bytes(PAYLOAD)

            def fake_urlopen(request, timeout=None):
                if request.get_method() == "HEAD":
                    return _Response(b"", content_length=len(PAYLOAD))
                raise AssertionError("must not re-download a complete file")

            with mock.patch.object(
                experts.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                out = experts._stream_download(
                    "https://example.invalid/experts_3.tar.gz", dst
                )
            self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_unknown_published_size_does_not_block_a_complete_file(self) -> None:
        """A server that will not report a length must not stall the pipeline."""
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "experts_4.tar.gz"
            dst.write_bytes(PAYLOAD)
            with mock.patch.object(
                experts, "_published_size", return_value=None
            ):
                out = experts._stream_download(
                    "https://example.invalid/experts_4.tar.gz", dst
                )
            self.assertEqual(out.read_bytes(), PAYLOAD)


if __name__ == "__main__":
    unittest.main()
