from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .associate import center_inside, head_region, torso_region, xyxy_iou

FINAL_CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]


@dataclass
class WorkerCompliance:
    worker_box: list[float]
    status: str
    has_helmet: bool
    has_vest: bool
    violation_reasons: list[str]
    confidence: float


def _split_detections(detections: list[dict[str, Any]]):
    by_cls = {c: [] for c in FINAL_CLASSES}
    for det in detections:
        name = det.get("class_name")
        if name in by_cls:
            by_cls[name].append(det)
    return by_cls


def evaluate_compliance(detections: list[dict[str, Any]], require_vest: bool = True) -> dict[str, Any]:
    """Evalúa cumplimiento por frame.

    Entrada esperada por detección:
    {
      "class_name": "person" | "helmet" | "no_helmet" | "vest" | "no_vest",
      "confidence": float,
      "xyxy": [x1,y1,x2,y2]
    }

    Si no hay cajas de persona, se usan detecciones de helmet/no_helmet como
    pseudo-trabajadores para que el demo siga funcionando en datasets que solo
    anotan cabeza/casco.
    """
    by_cls = _split_detections(detections)
    workers: list[WorkerCompliance] = []

    person_boxes = by_cls["person"]

    if person_boxes:
        for p in person_boxes:
            pbox = p["xyxy"]
            hregion = head_region(pbox)
            tregion = torso_region(pbox)

            helmet_hits = [d for d in by_cls["helmet"] if center_inside(d["xyxy"], hregion) or xyxy_iou(d["xyxy"], hregion) > 0.05]
            no_helmet_hits = [d for d in by_cls["no_helmet"] if center_inside(d["xyxy"], hregion) or xyxy_iou(d["xyxy"], hregion) > 0.05]
            vest_hits = [d for d in by_cls["vest"] if center_inside(d["xyxy"], tregion) or xyxy_iou(d["xyxy"], tregion) > 0.05]
            no_vest_hits = [d for d in by_cls["no_vest"] if center_inside(d["xyxy"], tregion) or xyxy_iou(d["xyxy"], tregion) > 0.05]

            has_helmet = bool(helmet_hits) and not bool(no_helmet_hits)
            has_vest = bool(vest_hits) and not bool(no_vest_hits)

            reasons = []
            if not has_helmet:
                reasons.append("sin_casco")
            if require_vest and not has_vest:
                reasons.append("sin_chaleco")

            status = "compliant" if not reasons else "non_compliant"
            confs = [p.get("confidence", 0.0)] + [d.get("confidence", 0.0) for d in helmet_hits + no_helmet_hits + vest_hits + no_vest_hits]
            workers.append(
                WorkerCompliance(
                    worker_box=list(map(float, pbox)),
                    status=status,
                    has_helmet=has_helmet,
                    has_vest=has_vest,
                    violation_reasons=reasons,
                    confidence=max(confs) if confs else 0.0,
                )
            )
    else:
        # Fallback: cada detección de cabeza/casco funciona como pseudo-worker.
        for d in by_cls["helmet"]:
            workers.append(WorkerCompliance(list(map(float, d["xyxy"])), "compliant", True, not require_vest, [], d.get("confidence", 0.0)))
        for d in by_cls["no_helmet"]:
            workers.append(WorkerCompliance(list(map(float, d["xyxy"])), "non_compliant", False, False, ["sin_casco"], d.get("confidence", 0.0)))

    total = len(workers)
    compliant = sum(1 for w in workers if w.status == "compliant")
    non_compliant = total - compliant

    return {
        "num_workers": total,
        "num_compliant": compliant,
        "num_non_compliant": non_compliant,
        "compliance_rate": compliant / total if total else 0.0,
        "workers": [asdict(w) for w in workers],
    }
