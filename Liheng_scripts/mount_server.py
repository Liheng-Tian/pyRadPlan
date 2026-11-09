#!/usr/bin/env python3
import argparse, os, pathlib, subprocess, sys
def ensure_drive_mounted(win_letter: str = "P") -> pathlib.Path:
    """
    Try to use /mnt/<letter>. If we cannot create that directory because of
    a permission error, fall back to a directory inside the user's home
    (e.g. ~/p_mount). Returns the final mount point.
    """
    # Preferred system location
    sys_mp = pathlib.Path("/mnt") / win_letter.lower()
    try:
        # Try to create the system mount point (may raise PermissionError)
        sys_mp.mkdir(parents=True, exist_ok=True)
        mount_point = sys_mp
    except PermissionError:
        # Fallback to a user‑owned location
        home_mp = pathlib.Path.home() / f"{win_letter.lower()}_mount"
        home_mp.mkdir(parents=True, exist_ok=True)
        mount_point = home_mp
        print(f"⚠️  No permission for /mnt/{win_letter.lower()}; using {mount_point}")

    # If already has content, assume it is already mounted
    if any(mount_point.iterdir()):
        return mount_point

    # Perform the actual mount (needs the password‑less sudo rule)
    subprocess.run(
        ["sudo", "mount", "-t", "drvfs", f"{win_letter}:", str(mount_point)],
        check=True,
    )
    return mount_point
# ----------------------------------------------------------------------
# Argument parsing (keep your original options)


wsl_path = pathlib.Path("/home/mlihtian/scripts/pix2pix")
os.chdir(wsl_path)

# Mount the Windows drive (will exit if it cannot be mounted)
p_mount = ensure_drive_mounted("P")

# Example usage of the mounted path:
server_root = p_mount / "data" / "SynthRAD2023_KD"
print(f"🚀 Server data root = {server_root}")

# ----------------------------------------------------------------------
# … the rest of your original script follows, using pathlib or
# str(server_root) wherever you need the path string.
# ----------------------------------------------------------------------