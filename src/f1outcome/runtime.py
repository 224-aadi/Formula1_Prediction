from __future__ import annotations

import ctypes
import glob
import os
import site
import sys


def prepare_lightgbm_runtime() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_libs = [
        os.path.join(repo_root, "vendor", "libgomp.so.1"),
        "/var/task/vendor/libgomp.so.1",
    ]

    for base_path in [*site.getsitepackages(), *sys.path, "/tmp/_vc_deps"]:
        if not base_path or not os.path.exists(base_path):
            continue

        candidate_libs.extend(glob.glob(os.path.join(base_path, "**", "libgomp*.so*"), recursive=True))

    for libgomp_path in candidate_libs:
        if not os.path.exists(libgomp_path):
            continue

        for base_path in [*site.getsitepackages(), *sys.path, "/tmp/_vc_deps"]:
            if not base_path or not os.path.exists(base_path):
                continue

            for link_dir in [
                os.path.dirname(libgomp_path),
                *glob.glob(os.path.join(base_path, "**", "lightgbm", "lib"), recursive=True),
            ]:
                try:
                    os.makedirs(link_dir, exist_ok=True)
                    link_path = os.path.join(link_dir, "libgomp.so.1")
                    if not os.path.exists(link_path):
                        os.symlink(libgomp_path, link_path)
                except OSError:
                    pass

        try:
            ctypes.CDLL(libgomp_path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
