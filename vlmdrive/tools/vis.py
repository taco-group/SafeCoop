#!/usr/bin/env python3
"""
img2video.py — Convert images in a directory tree into video(s).

Examples
--------
# One video from all images under a path
python img2video.py /path/to/results_bk/results_driving_safecoop_4agents --fps 10 -o out.mp4

# One video per agent_* folder (recursively)
python img2video.py /path/to/results_bk/results_driving_safecoop_4agents --per-agent --fps 8 --stride 2 -O videos/

# Only images under a specific timestamp folder
python img2video.py /path/to/.../image_buffer/20250826_100645 --per-agent --fps 12

# Resize frames to 1280x720
python img2video.py /path/to/root --fps 15 --size 1280x720 -O videos/
"""
import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2  # pip install opencv-python


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    """Sort like a human: frame_2.png < frame_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def list_images(
    root: Path,
    pattern: str = "*",
    recursive: bool = True,
    sort_by: str = "name",
) -> List[Path]:
    if recursive:
        it = root.rglob(pattern)
    else:
        it = root.glob(pattern)
    files = [p for p in it if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

    if sort_by == "name":
        files.sort(key=lambda p: natural_key(p.name))
    elif sort_by == "path":
        files.sort(key=lambda p: natural_key(str(p)))
    elif sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        raise ValueError(f"Unsupported sort_by: {sort_by}")
    return files


def parse_size(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    m = re.match(r"^(\d+)[xX](\d+)$", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("size must be like 1280x720")
    return int(m.group(1)), int(m.group(2))


def ensure_out_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_video(
    images: List[Path],
    out_path: Path,
    fps: float,
    size: Optional[Tuple[int, int]] = None,
    codec: str = "mp4v",
    stride: int = 1,
    start: int = 0,
    end: Optional[int] = None,
    verbose: bool = True,
):
    if not images:
        if verbose:
            print(f"[skip] No images for {out_path}")
        return

    # Apply slicing / stride
    imgs = images[start:end:stride]
    if not imgs:
        if verbose:
            print(f"[skip] After slicing, no frames for {out_path}")
        return

    # Read first frame to get size
    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {imgs[0]}")
    h0, w0 = first.shape[:2]
    if size is None:
        frame_size = (w0, h0)
    else:
        frame_size = size

    fourcc = cv2.VideoWriter_fourcc(*codec)
    ensure_out_dir(out_path)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] Cannot read: {p}, skipping")
            continue
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)
        vw.write(img)
        if verbose and (i + 1) % 100 == 0:
            print(f"[{out_path.name}] wrote {i+1}/{len(imgs)} frames")
    vw.release()
    if verbose:
        print(f"[done] {out_path}  ({len(imgs)} frames @ {fps} fps)")


def collect_agent_dirs(root: Path) -> List[Path]:
    """
    Find all 'agent_*' directories under the given root (any depth).
    e.g., .../image_buffer/20250826_100645/agent_0, agent_1, ...
    """
    return sorted([p for p in root.rglob("agent_*") if p.is_dir()],
                  key=lambda p: natural_key(str(p)))


def main():
    ap = argparse.ArgumentParser(
        description="Convert images to video(s) from a results directory."
    )
    ap.add_argument("input", type=Path,
                    help="Input root directory (e.g., .../image_buffer/20250826_100645 or the higher-level results folder).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output file (single video). If not set and --per-agent is used, videos are written into -O/--outdir.")
    ap.add_argument("-O", "--outdir", type=Path, default=Path("videos"),
                    help="Output directory when generating multiple videos (e.g., per agent). Default: ./videos")
    ap.add_argument("--per-agent", action="store_true",
                    help="Create one video per agent_* directory (recursively).")
    ap.add_argument("--pattern", default="*",
                    help="Glob pattern for images (default: '*').")
    ap.add_argument("--recursive", action="store_true",
                    help="Search images recursively (default: False unless --per-agent).")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="Frames per second (default: 12).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth frame (frequency selection). Default: 1 (take all).")
    ap.add_argument("--start", type=int, default=0,
                    help="Start index (inclusive).")
    ap.add_argument("--end", type=int, default=None,
                    help="End index (exclusive).")
    ap.add_argument("--size", type=parse_size, default=None,
                    help="Resize frames to WxH, e.g., 1280x720. If omitted, uses first frame size.")
    ap.add_argument("--sort-by", choices=["name", "path", "mtime"], default="name",
                    help="Frame ordering (default: name).")
    ap.add_argument("--codec", default="mp4v",
                    help="FourCC codec for VideoWriter (default: mp4v). Common: mp4v, avc1, XVID.")
    ap.add_argument("--ext", default="mp4",
                    help="Output extension (default: mp4).")
    args = ap.parse_args()

    root = args.input.resolve()

    if args.per_agent:
        agent_dirs = collect_agent_dirs(root)
        if not agent_dirs:
            print(f"[warn] No agent_* directories found under {root}")
        for ad in agent_dirs:
            imgs = list_images(ad, pattern=args.pattern, recursive=False, sort_by=args.sort_by)
            rel = ad.relative_to(root)
            out_path = (args.outdir / rel).with_suffix(f".{args.ext}")
            write_video(
                images=imgs,
                out_path=out_path,
                fps=args.fps,
                size=args.size,
                codec=args.codec,
                stride=max(1, args.stride),
                start=max(0, args.start),
                end=args.end,
            )
    else:
        # One video from everything under root (respect --recursive)
        imgs = list_images(root, pattern=args.pattern,
                           recursive=bool(args.recursive), sort_by=args.sort_by)
        if args.output is None:
            # Name by folder
            out_name = f"{root.name}.{args.ext}"
            out_path = (args.outdir / out_name) if args.outdir else Path(out_name)
        else:
            out_path = args.output if args.output.suffix else args.output.with_suffix(f".{args.ext}")

        write_video(
            images=imgs,
            out_path=out_path,
            fps=args.fps,
            size=args.size,
            codec=args.codec,
            stride=max(1, args.stride),
            start=max(0, args.start),
            end=args.end,
        )


if __name__ == "__main__":
    main()