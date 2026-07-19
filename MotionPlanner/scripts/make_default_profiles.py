"""Regenerate the checked-in default HardwareProfile JSONs (deterministic).

Run from anywhere: python MotionPlanner/scripts/make_default_profiles.py
Two runs produce byte-identical files (save_profile is canonical) — the
profiles double as a determinism probe, like the MelodyExtractor fixtures.

Device-id convention (informational until multi-axis firmware exists, D-029):
0x01 is the firmware MVP single motor; we reserve 2=belt, 3/4=bow differential
pair, then finger axes upward from 5 in unit order (x, press, z per unit).
"""
from __future__ import annotations

from pathlib import Path

from motion_planner.hardware import FingerAxis, FingerUnit, HardwareProfile, Topology
from motion_planner.profile_io import save_profile

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


def _finger(base_device: int, roaming: bool) -> FingerUnit:
    unit = FingerUnit()
    if not roaming:
        # Concept B: the finger lives on its string — no X (string-select) axis.
        unit = FingerUnit(x=FingerAxis(range_m=0.0, v_max_mps=0.01, a_max_mps2=1.0),
                         press=unit.press, z=unit.z, motor=dict(unit.motor),
                         device_ids={})
    return FingerUnit(x=unit.x, press=unit.press, z=unit.z, motor=dict(unit.motor),
                      device_ids={"x": base_device, "press": base_device + 1, "z": base_device + 2})


def main() -> None:
    a1 = HardwareProfile(
        name="concept_a_1finger",
        comment="Concept A proof-of-concept: one roaming 3-DoF finger (four-bar XY + lead-screw Z).",
        topology=Topology(concept="A", n_fingers=1, finger_home_string=(None,)),
        fingers=(_finger(5, roaming=True),),
    ).validate()

    a2 = HardwareProfile(
        name="concept_a_2finger",
        comment="Concept A target: two roaming fingers — full fingerboard with 6 motors; "
                "evaluates neighbor-string/neighbor-finger transitions.",
        topology=Topology(concept="A", n_fingers=2, finger_home_string=(None, None)),
        fingers=(_finger(5, roaming=True), _finger(8, roaming=True)),
    ).validate()

    b4 = HardwareProfile(
        name="concept_b_4finger",
        comment="Concept B (GhostPlay-style): one finger per string — 4 traverse axes + "
                "4 pressing units, no string-select DoF.",
        topology=Topology(concept="B", n_fingers=4, finger_home_string=(0, 1, 2, 3)),
        fingers=tuple(_finger(5 + 3 * i, roaming=False) for i in range(4)),
    ).validate()

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    for profile in (a1, a2, b4):
        path = save_profile(profile, PROFILES_DIR / f"{profile.name}.json")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
