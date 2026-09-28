"""Docker HEALTHCHECK: a describe -> info round trip (proves the event loop and handler run, not just the socket).

The port only opens after the engines are loaded and warmed up, so "healthy" also means "ready".
The URI comes from the same config file as the server; a --uri given to the server must be given here too.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

from wyoming.client import AsyncClient
from wyoming.info import Describe, Info

from .config import DEFAULT_CONFIG, load_config


def _connect_uri(uri: str) -> str:
    u = urlparse(uri)
    if u.scheme == "tcp" and u.hostname in ("0.0.0.0", "::", None):
        return f"tcp://127.0.0.1:{u.port}"
    return uri


async def check(uri: str) -> Info:
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Describe().event())
        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("connection closed without info")
            if Info.is_type(event.type):
                info = Info.from_event(event)
                if not (info.asr or info.tts):
                    raise RuntimeError("info lists no ASR and no TTS program")
                return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--uri")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    uri = "?"
    try:
        uris = [args.uri] if args.uri else [ep.uri for ep in load_config(args.config).endpoints]
        for uri in uris:  # every endpoint must answer
            asyncio.run(asyncio.wait_for(check(_connect_uri(uri)), args.timeout))
    except Exception as err:  # Docker shows this text in `docker inspect`
        print(f"unhealthy: {uri}: {str(err) or type(err).__name__}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
