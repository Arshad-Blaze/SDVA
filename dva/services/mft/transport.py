"""Transport abstraction for MFT access.

The File Mover (Boundary 1) owns all MFT connectivity, but it must stay
independently testable (Engineering rule 14) against a mock/local source.
It therefore talks to a ``Transport`` instead of paramiko directly:

- ``SftpTransport``  -> the real MFT server over SFTP (reference
  ``file_transfer.py``).
- ``LocalTransport`` -> a local folder stood in for the MFT share, used
  by unit/integration tests and local development.
- any test can provide a minimal fake implementing the same methods.

paramiko is imported lazily so that non-MFT tests never need it installed.
"""

from __future__ import annotations

import shutil
import stat as stat_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RemoteFileInfo:
    """A remote directory entry as seen by the transport."""

    name: str
    size: int | None = None
    mtime: float | None = None
    is_dir: bool = False


class Transport(Protocol):
    """Minimal surface the File Mover needs from an MFT backend."""

    def list_files(self, remote_path: str) -> list[RemoteFileInfo]:
        """List non-recursive entries under ``remote_path``."""
        ...

    def get_size(self, remote_path: str) -> int | None:
        """Byte size of one remote file, or None when unknown."""
        ...

    def get(self, remote_path: str, local_path: str) -> None:
        """Copy one remote file to a local path."""
        ...

    def close(self) -> None:
        """Release any held connections/resources."""
        ...


class SftpTransport:
    """SFTP transport against the MFT server, built on paramiko."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int = 22,
    ) -> None:
        self._host = host
        self._username = username
        self._password = password
        self._port = port
        self._sftp = None
        self._transport = None

    # ------------------------------------------------------------------
    def _ensure_sftp(self):
        """Connect lazily and cache the SFTP client for reuse."""
        if self._sftp is not None:
            return self._sftp

        import paramiko

        transport = paramiko.Transport((self._host, self._port))
        transport.connect(username=self._username, password=self._password)
        self._transport = transport
        self._sftp = paramiko.SFTPClient.from_transport(transport)
        return self._sftp

    # ------------------------------------------------------------------
    def list_files(self, remote_path: str) -> list[RemoteFileInfo]:
        sftp = self._ensure_sftp()
        return [
            RemoteFileInfo(
                name=entry.filename,
                size=entry.st_size,
                mtime=entry.st_mtime,
                is_dir=stat_mod.S_ISDIR(entry.st_mode),
            )
            for entry in sftp.listdir_attr(remote_path)
        ]

    def get_size(self, remote_path: str) -> int | None:
        sftp = self._ensure_sftp()
        return sftp.stat(remote_path).st_size

    def get(self, remote_path: str, local_path: str) -> None:
        sftp = self._ensure_sftp()
        sftp.get(remote_path, local_path)

    def close(self) -> None:
        for client in (self._sftp, self._transport):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        self._sftp = None
        self._transport = None


class LocalTransport:
    """Local-filesystem stand-in for the MFT, for tests and development.

    ``root`` is treated as the remote share root. Remote paths passed to
    this transport are resolved either relative to ``root`` or absolute.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    # ------------------------------------------------------------------
    def _resolve(self, remote_path: str) -> Path:
        # MFT paths are virtual share paths: strip any leading slash so
        # "/" and "/sales.txt" map onto the local root, never the real
        # filesystem root.
        return self._root / Path(remote_path.lstrip("/"))

    # ------------------------------------------------------------------
    def list_files(self, remote_path: str) -> list[RemoteFileInfo]:
        base = self._resolve(remote_path)
        if not base.exists():
            return []
        return [
            RemoteFileInfo(
                name=entry.name,
                size=entry.stat().st_size if entry.is_file() else None,
                mtime=entry.stat().st_mtime,
                is_dir=entry.is_dir(),
            )
            for entry in sorted(base.iterdir(), key=lambda item: item.name)
        ]

    def get_size(self, remote_path: str) -> int | None:
        return self._resolve(remote_path).stat().st_size

    def get(self, remote_path: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._resolve(remote_path), local_path)

    def close(self) -> None:
        """Nothing to release for a local transport."""
        return None
