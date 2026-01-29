import json, time
from pathlib import Path
from typing import Any, Dict, Optional
import requests

# Jolpica rate limits: 4 req/sec, 500 req/hour :contentReference[oaicite:3]{index=3}
class JolpicaClient:
    def __init__(self, base_url: str, cache_dir: Path, min_interval_s: float = 0.30):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.session = requests.Session()

    def _sleep_if_needed(self):
        dt = time.time() - self._last_call
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        # Stable cache key
        safe = path.strip("/").replace("/", "_").replace(".json", "")
        pkey = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        fname = f"{safe}__{pkey}.json" if pkey else f"{safe}.json"
        fpath = self.cache_dir / fname

        if fpath.exists():
            return json.loads(fpath.read_text(encoding="utf-8"))

        self._sleep_if_needed()
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = self.session.get(url, params=params, timeout=30)
        self._last_call = time.time()

        # If you hit 429s you’re over the rate limit :contentReference[oaicite:4]{index=4}
        r.raise_for_status()
        data = r.json()
        fpath.write_text(json.dumps(data), encoding="utf-8")
        return data
