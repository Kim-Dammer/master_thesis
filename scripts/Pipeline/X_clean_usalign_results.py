from pathlib import Path
import shutil 
import argparse

def delete_usalign_outputs(root: str | Path, dry_run: bool = False) -> list[Path]:
    """
    Recursively find and delete all 'usalign_outputs' directories under `root`.

    Args:
        root: Path to the directory to search.
        dry_run: If True, only report what would be deleted without deleting.

    Returns:
        List of paths that were (or would be) deleted.
    """
    root = Path(root)
    deleted = []

    # Collect matches first, so we don't mutate the tree while os.walk/rglob is iterating it
    targets = sorted(root.rglob("usalign_outputs"), key=lambda p: len(p.parts), reverse=True)

    for path in targets:
        if not path.is_dir():
            continue
        # Skip if it's inside a directory we've already deleted
        if any(str(path).startswith(str(d) + "/") for d in deleted):
            continue

        deleted.append(path)
        if dry_run:
            print(f"[dry-run] Would delete: {path}")
        else:
            print(f"Deleting: {path}")
            shutil.rmtree(path)

    return deleted

_ = delete_usalign_outputs("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/t11_RM_TM_updated_CF_pipeline/CombFold", dry_run=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete 'usalign_outputs' directories.")
    parser.add_argument("root", type=str, help="Root directory to search for 'usalign_outputs'.")
    parser.add_argument("--dry", action="store_true", help="Only report what would be deleted without deleting.")
    args = parser.parse_args()

    delete_usalign_outputs(args.root, dry_run=args.dry)