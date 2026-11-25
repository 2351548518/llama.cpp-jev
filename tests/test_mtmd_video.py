"""Exercise the real video probe through the public C API; no model download is needed."""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class VideoParams(ctypes.Structure):
    _fields_ = [
        ("fps_target", ctypes.c_float), ("ffmpeg_bin_dir", ctypes.c_char_p),
        ("timestamp_interval_ms", ctypes.c_int64), ("no_soundtrack", ctypes.c_bool),
        ("decoder", ctypes.c_void_p), ("center_frames", ctypes.c_int32),
        ("prompt_template", ctypes.c_int), ("max_full_soundtrack_seconds", ctypes.c_float),
    ]


class Options(ctypes.Structure):
    _fields_ = [("video_params", VideoParams)]


class MediaMeta(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
        ("sample_rate", ctypes.c_int32), ("duration", ctypes.c_float), ("n_frames", ctypes.c_int32),
    ]


class VideoProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="mtmd-video-", dir=Path.cwd())
        cls.folder = Path(cls.temp.name)
        cls.library = ctypes.CDLL(str(LIBRARY))
        cls.library.mtmd_helper_init_opt_default.restype = Options
        cls.library.mtmd_helper_get_media_meta.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, Options, ctypes.POINTER(MediaMeta),
        ]
        cls.library.mtmd_helper_get_media_meta.restype = ctypes.c_int32
        cls.inputs = {}
        for name, args in [("zero.webm", []), ("offset.webm", ["-output_ts_offset", "2"])]:
            path = cls.folder / name
            subprocess.run([
                FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=1",
                "-c:v", "libvpx-vp9", "-threads", "1", "-write_crc32", "0", *args, "-y", str(path),
            ], check=True, timeout=30)
            cls.inputs[name] = path.read_bytes()
        path = cls.folder / "long-audio.mp4"
        subprocess.run([
            FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=duration=2", "-c:v", "mpeg4", "-threads", "1", "-c:a", "aac",
            "-y", str(path),
        ], check=True, timeout=30)
        cls.inputs[path.name] = path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, data, folder=None):
        options = self.library.mtmd_helper_init_opt_default()
        if folder is not None:
            options.video_params.ffmpeg_bin_dir = os.fsencode(folder)
        result = MediaMeta()
        buffer = ctypes.create_string_buffer(data)
        code = self.library.mtmd_helper_get_media_meta(buffer, len(data), options, ctypes.byref(result))
        return code, result

    def test_webm_container_duration(self):
        # Rename the EBML duration tag without changing the layout of the file.
        data = self.inputs["zero.webm"].replace(b"\x45\xa3\x88DURATION", b"\x45\xa3\x88DURATIOX", 1)
        path = self.folder / "container-duration.webm"
        path.write_bytes(data)
        metadata = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=duration,nb_frames:stream_tags=DURATION:format=duration", "-of", "json", str(path),
        ], text=True, timeout=10))
        stream = metadata["streams"][0]
        self.assertNotIn("duration", stream)
        self.assertNotIn("nb_frames", stream)
        self.assertNotIn("DURATION", stream.get("tags", {}))
        self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.0, places=4)
        code, result = self.probe(data)
        self.assertEqual(code, 0)
        self.assertEqual((result.type, result.width, result.height), (3, 64, 48))
        self.assertAlmostEqual(result.duration, 1.0, places=4)

    def test_webm_nonzero_start(self):
        code, result = self.probe(self.inputs["offset.webm"])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(result.duration, 1.0, places=4)

    def test_video_stream_duration_precedes_longer_audio(self):
        code, result = self.probe(self.inputs["long-audio.mp4"])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(result.duration, 1.0, places=3)

    def test_invalid_media(self):
        code, result = self.probe(b"not a valid video container")
        self.assertNotEqual(code, 0)
        self.assertEqual(result.type, 0)

    def test_probe_cache_is_content_and_tool_specific(self):
        if os.name == "nt":
            self.skipTest("The probe wrapper uses a POSIX executable script")
        folder = self.folder / "counted"
        folder.mkdir()
        count = folder / "calls"
        wrapper = folder / "ffprobe"
        wrapper.write_text(
            "#!" + sys.executable + "\nimport subprocess, sys\n"
            "with open(" + repr(str(count)) + ", 'ab') as f: f.write(b'probe\\n')\n"
            "raise SystemExit(subprocess.call([" + repr(FFPROBE) + ", *sys.argv[1:]]))\n"
        )
        wrapper.chmod(0o755)
        for _ in range(3):
            self.assertEqual(self.probe(self.inputs["zero.webm"], folder)[0], 0)
        self.assertEqual(len(count.read_bytes().splitlines()), 1)
        self.assertEqual(self.probe(self.inputs["offset.webm"], folder)[0], 0)
        self.assertEqual(len(count.read_bytes().splitlines()), 2)
        other = self.folder / "other-tool"
        other.mkdir()
        shutil.copy2(wrapper, other / "ffprobe")
        self.assertEqual(self.probe(self.inputs["zero.webm"], other)[0], 0)
        self.assertEqual(len(count.read_bytes().splitlines()), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    LIBRARY = args.library.resolve(strict=True)
    FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not FFMPEG or not FFPROBE:
        print("SKIP: ffmpeg and ffprobe are needed for the video probe regression")
        sys.exit(77)
    encoders = subprocess.check_output([FFMPEG, "-v", "error", "-encoders"], text=True, timeout=10)
    rows = (line.split() for line in encoders.splitlines())
    names = {row[1] for row in rows if len(row) >= 2}
    missing = {"libvpx-vp9", "mpeg4", "aac"} - names
    if missing:
        print("SKIP: missing FFmpeg fixture encoders", ", ".join(sorted(missing)))
        sys.exit(77)
    if os.name == "nt":
        dll_directory = os.add_dll_directory(str(LIBRARY.parent))
    unittest.main(argv=[sys.argv[0], *remaining])
