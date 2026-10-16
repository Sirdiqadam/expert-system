"""Load and validate the knowledge base (rubric, rules, policies, content points).

Validation errors name the offending file and line wherever the underlying YAML
parser provides that information.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schemas import (
    ContentPoint,
    Policies,
    Rule,
    Rubric,
    TaskPrompt,
)

KB_FILES = ("rubric.yaml", "rules.yaml", "content_points.yaml", "policies.yaml")


class KnowledgeError(Exception):
    """Raised when the knowledge base is missing, malformed or inconsistent."""

    def __init__(self, message: str, path: Path | None = None, line: int | None = None):
        self.path = path
        self.line = line
        location = ""
        if path is not None:
            location = f"{path.name}" + (f":{line}" if line else "")
            location = f"[{location}] "
        super().__init__(f"{location}{message}")


@dataclass
class KnowledgeBase:
    """The validated knowledge base, ready for the inference engine."""

    rubric: Rubric
    rules: list[Rule]
    policies: Policies
    prompts: list[TaskPrompt] = field(default_factory=list)
    content_points: list[ContentPoint] = field(default_factory=list)
    version: str = ""

    def prompt(self, prompt_id: str) -> TaskPrompt | None:
        for p in self.prompts:
            if p.id == prompt_id:
                return p
        return None

    def rule(self, rule_id: str) -> Rule | None:
        for r in self.rules:
            if r.id == rule_id:
                return r
        return None

    def referenced_facts(self) -> set[str]:
        """All fact names referenced by any rule condition."""

        names: set[str] = set()
        for r in self.rules:
            names.update(r.if_.referenced_facts())
        return names


def _load_yaml(path: Path) -> object:
    if not path.exists():
        raise KnowledgeError("knowledge file not found", path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - message formatting
        line = None
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            line = mark.line + 1
        raise KnowledgeError(f"invalid YAML: {exc.problem}", path, line) from exc


def _validate_rubric(rubric: Rubric, path: Path) -> None:
    if not rubric.criteria:
        raise KnowledgeError("rubric has no criteria", path)
    seen: set[str] = set()
    for crit in rubric.criteria:
        if crit.id in seen:
            raise KnowledgeError(f"duplicate criterion id '{crit.id}'", path)
        seen.add(crit.id)
        if crit.mode == "levels":
            if not crit.levels:
                raise KnowledgeError(
                    f"criterion '{crit.id}' (mode=levels) has no levels", path
                )
            levels = {lvl.level for lvl in crit.levels}
            if len(levels) != len(crit.levels):
                raise KnowledgeError(f"criterion '{crit.id}' has duplicate levels", path)
            top = max(crit.levels, key=lambda l: l.level)
            if top.range[1] != crit.max_score:
                raise KnowledgeError(
                    f"criterion '{crit.id}' top level range max {top.range[1]} "
                    f"does not equal max_score {crit.max_score}",
                    path,
                )
        elif crit.mode == "points":
            if crit.levels:
                raise KnowledgeError(
                    f"criterion '{crit.id}' (mode=points) must not define levels", path
                )
            if crit.max_score <= 0:
                raise KnowledgeError(
                    f"criterion '{crit.id}' (mode=points) needs max_score > 0", path
                )
    if rubric.aggregate == "sum":
        total = sum(c.max_score for c in rubric.criteria)
        if total != rubric.max_total:
            raise KnowledgeError(
                f"max_total {rubric.max_total} does not equal sum of criterion "
                f"max_scores {total}",
                path,
            )
    else:  # best_n
        if not rubric.best_n or rubric.best_n <= 0:
            raise KnowledgeError("aggregate 'best_n' requires best_n > 0", path)
        if rubric.best_n > len(rubric.criteria):
            raise KnowledgeError(
                f"best_n {rubric.best_n} exceeds the number of criteria "
                f"{len(rubric.criteria)}",
                path,
            )


def _validate_rules(rules: list[Rule], rubric: Rubric, path: Path) -> None:
    ids: set[str] = set()
    reachable: dict[str, set[int]] = {c.id: set() for c in rubric.criteria}
    point_reachable: dict[str, bool] = {
        c.id: False for c in rubric.criteria if c.mode == "points"
    }
    for rule in rules:
        if rule.id in ids:
            raise KnowledgeError(f"duplicate rule id '{rule.id}'", path)
        ids.add(rule.id)
        if not rule.then:
            raise KnowledgeError(f"rule '{rule.id}' has no actions", path)
        crit = rubric.criterion(rule.criterion) if rule.criterion else None
        if rule.criterion and crit is None:
            raise KnowledgeError(
                f"rule '{rule.id}' references unknown criterion "
                f"'{rule.criterion}'",
                path,
            )
        for action in rule.then:
            if action.action == "assert":
                if not action.fact:
                    raise KnowledgeError(
                        f"rule '{rule.id}' assert action requires 'fact'", path
                    )
            elif action.action == "score":
                target = action.criterion or rule.criterion
                target_crit = rubric.criterion(target) if target else None
                if target_crit is None:
                    raise KnowledgeError(
                        f"rule '{rule.id}' score action references unknown "
                        f"criterion '{target}'",
                        path,
                    )
                if target_crit.mode == "points":
                    raise KnowledgeError(
                        f"rule '{rule.id}' scores point criterion '{target}'; "
                        "use an 'add' action instead",
                        path,
                    )
                if action.level is not None:
                    if target_crit.level(action.level) is None:
                        raise KnowledgeError(
                            f"rule '{rule.id}' scores unknown level "
                            f"{action.level} for '{target}'",
                            path,
                        )
                    reachable[target_crit.id].add(action.level)
                elif action.value is None:
                    raise KnowledgeError(
                        f"rule '{rule.id}' score action requires 'value' or "
                        f"'level'",
                        path,
                    )
            elif action.action == "add":
                target = action.criterion or rule.criterion
                target_crit = rubric.criterion(target) if target else None
                if target_crit is None:
                    raise KnowledgeError(
                        f"rule '{rule.id}' add action references unknown "
                        f"criterion '{target}'",
                        path,
                    )
                if action.amount is None:
                    raise KnowledgeError(
                        f"rule '{rule.id}' add action requires 'amount'", path
                    )
                point_reachable[target_crit.id] = True
            elif action.action == "penalty":
                if not (action.criterion or rule.criterion):
                    raise KnowledgeError(
                        f"rule '{rule.id}' penalty action requires 'criterion'",
                        path,
                    )
                if action.amount is None:
                    raise KnowledgeError(
                        f"rule '{rule.id}' penalty action requires 'amount'", path
                    )
            elif action.action == "flag":
                if not action.reason:
                    raise KnowledgeError(
                        f"rule '{rule.id}' flag action requires 'reason'", path
                    )

    for crit in rubric.criteria:
        if crit.mode != "levels":
            continue
        needed = {lvl.level for lvl in crit.levels}
        if not needed.issubset(reachable[crit.id]):
            missing = sorted(needed - reachable[crit.id])
            raise KnowledgeError(
                f"criterion '{crit.id}' has no score rules reaching levels "
                f"{missing}",
                path,
            )
    for cid, ok in point_reachable.items():
        if not ok:
            raise KnowledgeError(
                f"point criterion '{cid}' has no 'add' rules", path
            )


def load_knowledge(kb_dir: str | Path) -> KnowledgeBase:
    """Load and validate every KB file in *kb_dir*."""

    base = Path(kb_dir)
    rubric = Rubric.model_validate(_load_yaml(base / "rubric.yaml"))
    rules_raw = _load_yaml(base / "rules.yaml") or []
    if not isinstance(rules_raw, list):
        raise KnowledgeError("rules.yaml must contain a list of rules", base / "rules.yaml")
    rules = [Rule.model_validate(r) for r in rules_raw]
    points_raw = _load_yaml(base / "content_points.yaml") or {}
    policies = Policies.model_validate(_load_yaml(base / "policies.yaml") or {})

    prompts = [TaskPrompt.model_validate(p) for p in points_raw.get("prompts", [])]
    content_points = [
        ContentPoint.model_validate(c) for c in points_raw.get("content_points", [])
    ]

    _validate_rubric(rubric, base / "rubric.yaml")
    _validate_rules(rules, rubric, base / "rules.yaml")
    _validate_content_points(prompts, content_points, base / "content_points.yaml")

    version = _kb_version(base)
    return KnowledgeBase(
        rubric=rubric,
        rules=rules,
        policies=policies,
        prompts=prompts,
        content_points=content_points,
        version=version,
    )


def _validate_content_points(
    prompts: list[TaskPrompt], points: list[ContentPoint], path: Path
) -> None:
    known = {c.id for c in points}
    for prompt in prompts:
        for kp in prompt.key_points:
            if kp not in known:
                raise KnowledgeError(
                    f"prompt '{prompt.id}' references unknown content point "
                    f"'{kp}'",
                    path,
                )


def _kb_version(base: Path) -> str:
    """A short content hash over the KB files, used for versioning edits."""

    digest = hashlib.sha256()
    for name in KB_FILES:
        f = base / name
        if f.exists():
            digest.update(f.read_bytes())
    return digest.hexdigest()[:12]
