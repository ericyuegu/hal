#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
import sys

# Archive file extensions to process.
ARCHIVE_EXTS = (".zip", ".7z", ".rar", ".gz")


def check_dependency() -> None:
    """Check that 7z is installed."""
    if shutil.which("7z") is None:
        print("Error: 7z is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def extract_archive(filepath, dry_run=False, remove=False) -> bool:
    """
    Extracts the archive using 7z into its own directory.
    In dry-run mode, prints the command without executing it.
    Streams output from 7z in real time.
    If remove is True, deletes the archive after successful extraction.
    """
    directory = os.path.dirname(filepath)
    command = ["7z", "x", filepath, "-o" + directory]

    if dry_run:
        print(f"[DRY RUN] Would execute: {shlex.join(command)}")
        if remove:
            print(f"[DRY RUN] Would remove '{filepath}' after extraction.")
        return True

    print(f"Extracting '{filepath}' to '{directory}'...")

    try:
        # Start the process with stdout and stderr merged, and enable line-buffered output.
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            # Stream output line by line.
            for line in proc.stdout:
                print(line, end="")  # Already contains a newline.
            proc.wait()
            retcode = proc.returncode

        if retcode != 0:
            print(f"Error: 7z exited with code {retcode} for file '{filepath}'", file=sys.stderr)
            return False

        if remove:
            os.remove(filepath)
            print(f"Removed '{filepath}' after extraction.")

        return True

    except Exception as e:
        print(f"Error extracting '{filepath}': {e}", file=sys.stderr)
        return False


def find_archives(root="."):
    """
    Walks the directory tree starting at 'root'
    and yields paths to files that have one of the archive extensions.
    """
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(ARCHIVE_EXTS):
                yield os.path.join(dirpath, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively extract archives using 7z with real-time output streaming."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without performing extraction.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove archives after successful extraction.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to start searching for archives (default: current directory).",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    remove = args.remove
    root = args.root

    check_dependency()

    # Set to track archives that have been processed (by absolute path)
    processed_archives = set()
    iteration = 0

    # In dry-run mode, only one iteration is performed.
    if dry_run:
        print("Dry run mode enabled: Only one iteration will be performed.\n")

    while True:
        iteration += 1
        print(f"\nIteration {iteration}: Searching for archives in '{root}'...")
        archives = list(find_archives(root))
        new_archive_extracted = False

        for archive in archives:
            abspath = os.path.abspath(archive)
            # Skip archives that were already processed if we're not removing them.
            if abspath in processed_archives:
                continue
            success = extract_archive(archive, dry_run=dry_run, remove=remove)
            if success:
                new_archive_extracted = True
                # Only mark as processed if we are not removing the archive.
                if not remove and not dry_run:
                    processed_archives.add(abspath)

        # If in dry-run mode, exit after the first iteration.
        if dry_run:
            break

        # If no new archives were extracted, stop iterating.
        if not new_archive_extracted:
            print("No new archives extracted. Exiting.")
            break


if __name__ == "__main__":
    main()
