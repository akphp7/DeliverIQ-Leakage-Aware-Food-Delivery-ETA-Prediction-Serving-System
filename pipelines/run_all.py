"""Re-run every phase in order (about 20 minutes on a laptop).

    python -m pipelines.run_all            # all phases
    python -m pipelines.run_all --fast     # skips the slow Phase 1 CV section
"""
import sys
import time

from pipelines import (p0_data_fixes, p1_evaluation, p2_models, p3_statistics,
                       p4_sql_features, p5_production, p6_paper_benchmark)


def main():
    fast = "--fast" in sys.argv
    steps = [("Phase 0", p0_data_fixes.main),
             ("Phase 1", (lambda: None) if fast else p1_evaluation.main),
             ("Phase 2", p2_models.main),
             ("Phase 3", p3_statistics.main),
             ("Phase 4", p4_sql_features.main),
             ("Phase 5", p5_production.main),
             ("Phase 6", p6_paper_benchmark.main)]
    for name, fn in steps:
        t = time.time()
        print(f"\n##### {name} #####")
        fn()
        print(f"##### {name} done in {time.time() - t:.0f}s #####")


if __name__ == "__main__":
    main()
