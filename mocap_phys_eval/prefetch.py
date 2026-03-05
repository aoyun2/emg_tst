from __future__ import annotations

from pathlib import Path

from .config import EvalConfig
from .experts import discover_expert_snippets, ensure_full_expert_zoo
from .reference_bank import ExpertSnippetBank, build_expert_snippet_bank
from .utils import ensure_dir


def main() -> None:
    """Prefetch heavy assets (experts + reference bank) without running simulation."""
    cfg = EvalConfig()
    print("[mocap_phys_eval] status: prefetch starting")
    print(f"[mocap_phys_eval] config: experts_root={str(Path(cfg.experts_root).resolve())}")
    print(f"[mocap_phys_eval] config: experts_downloads_dir={str(Path(cfg.experts_downloads_dir).resolve())}")
    print(f"[mocap_phys_eval] config: artifacts_dir={str(Path(cfg.artifacts_dir).resolve())}")

    ensure_full_expert_zoo(experts_root=cfg.experts_root, downloads_dir=cfg.experts_downloads_dir)
    n = len(discover_expert_snippets(cfg.experts_root))
    print(f"[mocap_phys_eval] status: experts discovered (n_snippets={n})")
    if n < 2500:
        raise RuntimeError(
            f"Expected ~2589 expert snippets, but found only {n} under {str(cfg.experts_root)!r}. "
            "Re-run to resume downloads/extraction."
        )

    out_root = ensure_dir(cfg.artifacts_dir)
    bank_path = out_root / "reference_bank" / "mocapact_expert_snippets_right.npz"
    bank_ok = False
    if bank_path.exists():
        try:
            bank = ExpertSnippetBank.load_npz(bank_path)
            bank_ok = bool(len(bank) >= int(n))
        except Exception:
            bank_ok = False
    if not bank_ok:
        print("[mocap_phys_eval] status: building expert snippet reference bank")
        bank = build_expert_snippet_bank(experts_root=cfg.experts_root, side="right")
        bank.save_npz(bank_path)
    print(f"[mocap_phys_eval] status: prefetch complete (bank={str(bank_path)})")


if __name__ == "__main__":
    main()

