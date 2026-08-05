"""One-shot out-of-sample evaluation under an append-only custody ledger.

Every observation the canonical run has ever read ends on 2014-12-31, and the
trial count behind the incumbent specification is unrecoverable, so no
statistic computed on that history can support promotion.  The only way out is
data the pipeline has never seen, scored exactly once against criteria written
down beforehand.

This module supplies the machinery for that and nothing more.  It does not make
the incumbent promotable: the evaluation available here is narrow enough that
overstating it would be worse than not running it at all.  What it does is make
the protocol executable, so a future properly-frozen challenger has a path that
is enforced by code rather than described in prose.

The ledger is deliberately separate from ``registry``.  A trial registry
records hypotheses and their results; this records *custody* — which bytes were
sealed, when, under what acceptance criteria, and the single moment they were
spent.  Keeping them apart means adding one cannot destabilise the other, and
the two chains can be audited independently.

The scarce resource is the *look*, not the data.  A dataset may be registered
once and consumed once; the ledger refuses a second consumption, because the
second look is the one that turns a holdout into reused history.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .registry import ZERO_SHA256, _canonical_json, _sha256, _text, _utc, content_sha256


SCHEMA_VERSION = 1
HOLDOUT_REGISTER = "HOLDOUT_REGISTER"
HOLDOUT_CONSUME = "HOLDOUT_CONSUME"

# The evaluation this repository can actually run today, stated once so every
# artifact carries it.  Each entry is a fact about the available data, not a
# caveat added for comfort.
SUBSET_LIMITATIONS = (
    "twelve of fifty-nine traded roots: eleven government bonds and gold",
    "no equity index, FX, energy or agricultural exposure in the scored book",
    "trend sleeve only; the vendor supplies no unadjusted post-2014 series, so "
    "the basis-momentum sleeve cannot be reconstructed",
    "valuation and impact use back-adjusted levels, which differ from contract "
    "levels by the roll gaps accumulated to the series anchor",
    "no post-2014 financing series, so the funded basis is not available",
    "roughly five hundred sessions, which is far too short to resolve a Sharpe "
    "difference of the size the promotion gates require",
)


class HoldoutLedgerError(ValueError):
    """Raised when a custody rule would be violated."""


@dataclass(frozen=True)
class HoldoutDataset:
    """The sealed bytes, identified so a later run cannot quietly differ."""

    dataset_id: str
    description: str
    file_sha256: Mapping[str, str]
    window_start: str
    window_end: str
    roots: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "description", _text(self.description, "description"))
        if not self.file_sha256:
            raise ValueError("file_sha256 cannot be empty")
        digests = {}
        for name, digest in sorted(dict(self.file_sha256).items()):
            digests[_text(name, "file name")] = _sha256(digest, "file digest")
        object.__setattr__(self, "file_sha256", digests)
        if not self.roots:
            raise ValueError("roots cannot be empty")
        object.__setattr__(self, "roots", tuple(sorted(self.roots)))
        if self.window_start >= self.window_end:
            raise ValueError("window_start must precede window_end")

    @property
    def dataset_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class HoldoutRegistration:
    """A sealed dataset plus the criteria that will judge the one look at it."""

    dataset: HoldoutDataset
    model_fingerprint_sha256: str
    config_fingerprint_sha256: str
    protocol_sha256: str
    acceptance_criteria: Mapping[str, float]
    limitations: tuple[str, ...]
    custodian: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, HoldoutDataset):
            raise TypeError("dataset must be a HoldoutDataset")
        for name in (
            "model_fingerprint_sha256",
            "config_fingerprint_sha256",
            "protocol_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not self.acceptance_criteria:
            raise ValueError(
                "acceptance_criteria cannot be empty: a look with no "
                "pre-declared threshold is not an evaluation"
            )
        criteria = {}
        for name, value in sorted(dict(self.acceptance_criteria).items()):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"acceptance criterion {name} must be finite")
            criteria[_text(name, "criterion")] = numeric
        object.__setattr__(self, "acceptance_criteria", criteria)
        if not self.limitations:
            raise ValueError("limitations cannot be empty")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "custodian", _text(self.custodian, "custodian"))

    @property
    def registration_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class HoldoutConsumption:
    """The single scored look, bound to what was sealed."""

    dataset_id: str
    registration_sha256: str
    result_artifact_sha256: str
    observed: Mapping[str, float]
    completed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self,
            "registration_sha256",
            _sha256(self.registration_sha256, "registration_sha256"),
        )
        object.__setattr__(
            self,
            "result_artifact_sha256",
            _sha256(self.result_artifact_sha256, "result_artifact_sha256"),
        )
        observed = {}
        for name, value in sorted(dict(self.observed).items()):
            observed[_text(name, "metric")] = float(value)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "completed_at", _utc(self.completed_at, "completed_at"))


@dataclass(frozen=True)
class HoldoutEvent:
    """One link in the custody chain."""

    sequence: int
    event_type: str
    occurred_at: datetime
    previous_event_sha256: str
    registration: HoldoutRegistration | None = None
    consumption: HoldoutConsumption | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must start at one")
        event_type = _text(self.event_type, "event_type").upper()
        if event_type not in {HOLDOUT_REGISTER, HOLDOUT_CONSUME}:
            raise ValueError(f"unknown holdout event type: {event_type!r}")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, "occurred_at"))
        object.__setattr__(
            self,
            "previous_event_sha256",
            _sha256(self.previous_event_sha256, "previous_event_sha256"),
        )
        if event_type == HOLDOUT_REGISTER:
            if not isinstance(self.registration, HoldoutRegistration):
                raise ValueError("a registration event requires a registration")
            if self.consumption is not None:
                raise ValueError("a registration event cannot carry a consumption")
        else:
            if not isinstance(self.consumption, HoldoutConsumption):
                raise ValueError("a consumption event requires a consumption")
            if self.registration is not None:
                raise ValueError("a consumption event cannot carry a registration")

    @property
    def event_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class HoldoutLedger:
    """Append-only custody chain over sealed evaluation datasets."""

    events: tuple[HoldoutEvent, ...] = ()

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, HoldoutEvent) for event in events):
            raise ValueError("events must contain only HoldoutEvent values")
        object.__setattr__(self, "events", events)
        self._validate_chain()

    def _validate_chain(self) -> None:
        previous = ZERO_SHA256
        last_time: datetime | None = None
        registered: dict[str, HoldoutRegistration] = {}
        consumed: set[str] = set()
        for position, event in enumerate(self.events, start=1):
            if event.sequence != position:
                raise HoldoutLedgerError("holdout event sequence is not contiguous")
            if event.previous_event_sha256 != previous:
                raise HoldoutLedgerError("holdout chain linkage is broken")
            if last_time is not None and event.occurred_at < last_time:
                raise HoldoutLedgerError("holdout events must be non-decreasing in time")
            if event.event_type == HOLDOUT_REGISTER:
                dataset_id = event.registration.dataset.dataset_id
                if dataset_id in registered:
                    raise HoldoutLedgerError(
                        f"dataset {dataset_id!r} is already registered"
                    )
                registered[dataset_id] = event.registration
            else:
                dataset_id = event.consumption.dataset_id
                registration = registered.get(dataset_id)
                if registration is None:
                    raise HoldoutLedgerError(
                        "a holdout cannot be consumed before it is registered"
                    )
                if (
                    event.consumption.registration_sha256
                    != registration.registration_sha256
                ):
                    raise HoldoutLedgerError(
                        "consumption does not reconcile with the sealed registration"
                    )
                if dataset_id in consumed:
                    # The scarce resource is the look, not the bytes.
                    raise HoldoutLedgerError(
                        f"dataset {dataset_id!r} has already been consumed; a "
                        "second look would make it reused history"
                    )
                consumed.add(dataset_id)
            previous = event.event_sha256
            last_time = event.occurred_at

    @property
    def head_sha256(self) -> str:
        return self.events[-1].event_sha256 if self.events else ZERO_SHA256

    def register(
        self,
        registration: HoldoutRegistration,
        *,
        occurred_at: Any,
    ) -> HoldoutLedger:
        event = HoldoutEvent(
            sequence=len(self.events) + 1,
            event_type=HOLDOUT_REGISTER,
            occurred_at=occurred_at,
            previous_event_sha256=self.head_sha256,
            registration=registration,
        )
        return HoldoutLedger(events=self.events + (event,))

    def consume(
        self,
        consumption: HoldoutConsumption,
        *,
        occurred_at: Any,
    ) -> HoldoutLedger:
        event = HoldoutEvent(
            sequence=len(self.events) + 1,
            event_type=HOLDOUT_CONSUME,
            occurred_at=occurred_at,
            previous_event_sha256=self.head_sha256,
            consumption=consumption,
        )
        return HoldoutLedger(events=self.events + (event,))

    def is_extension_of(self, earlier: HoldoutLedger) -> bool:
        """Whether this ledger only appends to an earlier one."""

        if len(earlier.events) > len(self.events):
            return False
        return all(
            mine.event_sha256 == theirs.event_sha256
            for mine, theirs in zip(self.events, earlier.events)
        )

    def to_jsonl(self) -> str:
        lines = []
        for event in self.events:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "sequence": event.sequence,
                "event_type": event.event_type,
                "occurred_at": event.occurred_at.isoformat(),
                "previous_event_sha256": event.previous_event_sha256,
                "event_sha256": event.event_sha256,
                "registration": _serialize(event.registration),
                "consumption": _serialize(event.consumption),
            }
            lines.append(_canonical_json(payload))
        return "\n".join(lines) + ("\n" if lines else "")

    @classmethod
    def from_jsonl(cls, text: str) -> HoldoutLedger:
        events: list[HoldoutEvent] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            registration = payload.get("registration")
            consumption = payload.get("consumption")
            events.append(
                HoldoutEvent(
                    sequence=int(payload["sequence"]),
                    event_type=str(payload["event_type"]),
                    occurred_at=payload["occurred_at"],
                    previous_event_sha256=str(payload["previous_event_sha256"]),
                    registration=_deserialize_registration(registration),
                    consumption=_deserialize_consumption(consumption),
                )
            )
        ledger = cls(events=tuple(events))
        # Recomputing every digest on load is the point: a hand-edited file
        # cannot survive the chain check.
        for stored, event in zip(text.splitlines(), ledger.events):
            expected = json.loads(stored)["event_sha256"]
            if expected != event.event_sha256:
                raise HoldoutLedgerError(
                    f"holdout event {event.sequence} does not match its digest"
                )
        return ledger

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            previous = HoldoutLedger.from_jsonl(path.read_text(encoding="utf-8"))
            if not self.is_extension_of(previous):
                raise HoldoutLedgerError(
                    "refusing to rewrite an existing holdout ledger; the chain "
                    "is append-only"
                )
        path.write_text(self.to_jsonl(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HoldoutLedger:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_jsonl(path.read_text(encoding="utf-8"))


def _serialize(record: Any) -> Any:
    if record is None:
        return None
    payload = asdict(record)
    for key, value in list(payload.items()):
        if isinstance(value, datetime):
            payload[key] = value.isoformat()
    return payload


def _deserialize_registration(payload: Any) -> HoldoutRegistration | None:
    if payload is None:
        return None
    dataset = HoldoutDataset(**payload["dataset"])
    rest = {key: value for key, value in payload.items() if key != "dataset"}
    rest["limitations"] = tuple(rest["limitations"])
    return HoldoutRegistration(dataset=dataset, **rest)


def _deserialize_consumption(payload: Any) -> HoldoutConsumption | None:
    if payload is None:
        return None
    return HoldoutConsumption(**payload)


def seal_dataset(
    directory: Path,
    roots: Sequence[str],
    *,
    dataset_id: str,
    description: str,
    window_start: str,
    window_end: str,
    suffix: str = "_CCB.csv",
) -> HoldoutDataset:
    """Hash the files that will be scored, before anything reads their values."""

    directory = Path(directory)
    digests: dict[str, str] = {}
    for root in sorted(roots):
        path = directory / f"&{root}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"holdout input is missing: {path}")
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return HoldoutDataset(
        dataset_id=dataset_id,
        description=description,
        file_sha256=digests,
        window_start=window_start,
        window_end=window_end,
        roots=tuple(roots),
    )


def load_extension_frames(
    directory: Path,
    roots: Sequence[str],
    *,
    suffix: str = "_CCB.csv",
) -> dict[str, pd.DataFrame]:
    """Load the post-2014 continuation series for the overlapping roots.

    The vendor does not re-anchor its back-adjustment, so these files agree
    bit-for-bit with the canonical history wherever the two overlap and the
    continuation can simply be appended.  That property is verified rather than
    assumed by :func:`verify_continuity`.
    """

    directory = Path(directory)
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    deliveries: dict[str, pd.Series] = {}
    for root in roots:
        frame = pd.read_csv(
            directory / f"&{root}{suffix}", parse_dates=["Date"], index_col="Date"
        ).sort_index()
        closes[root] = frame["Close"].astype(float)
        volumes[root] = frame["Volume"].astype(float)
        deliveries[root] = frame["Delivery Month"].astype(float)
    return {
        "prices": pd.DataFrame(closes),
        "volumes": pd.DataFrame(volumes),
        "delivery_months": pd.DataFrame(deliveries),
    }


def load_extension_fx_rates(
    forex_dir: Path,
    canonical: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    currencies: Sequence[str] = ("EUR", "GBP", "JPY", "CHF", "CAD", "AUD"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extend USD conversion rates past the canonical history using spot FX.

    The canonical rates come from currency futures, which carry an interest
    rate basis over spot — on the join date the two differ by up to half a
    percent.  Splicing the spot series by its ratio at the last shared session
    keeps the join continuous, so no spurious step enters point values, notional
    or margin on the first out-of-sample session.  The alternative, using spot
    unscaled, would inject a level jump that the strategy would read as a
    one-day move in every foreign-denominated market at once.

    Returns the extended frame and a reconciliation report.
    """

    forex_dir = Path(forex_dir)
    rates: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for currency in currencies:
        spot = pd.read_csv(
            forex_dir / f"{currency}USD.csv", parse_dates=["Date"], index_col="Date"
        )["Close"].astype(float).sort_index()
        aligned = spot.reindex(canonical.index.union(spot.index)).ffill()
        # Per currency: the canonical frame is reindexed onto the extended
        # calendar, so its own last observation is where that series really
        # ends, not where the frame does.
        observed = canonical[currency].dropna()
        if observed.empty:
            raise ValueError(f"{currency} has no canonical observations to anchor on")
        join_date = observed.index.max()
        anchor_spot = float(aligned.loc[:join_date].iloc[-1])
        anchor_canonical = float(observed.iloc[-1])
        if not np.isfinite(anchor_spot) or anchor_spot <= 0:
            raise ValueError(f"{currency} spot anchor is unusable")
        scale = anchor_canonical / anchor_spot
        spliced = pd.concat(
            [
                canonical[currency].loc[:join_date],
                (aligned.loc[aligned.index > join_date] * scale),
            ]
        )
        rates[currency] = spliced.reindex(calendar).ffill(limit=10)
        rows.append(
            {
                "currency": currency,
                "join_date": join_date.date().isoformat(),
                "futures_rate": anchor_canonical,
                "spot_rate": anchor_spot,
                "splice_scale": scale,
                "basis_fraction": scale - 1.0,
            }
        )
    frame = pd.DataFrame(rates, index=calendar)
    frame["USD"] = 1.0
    return frame, pd.DataFrame(rows)


def verify_continuity(
    canonical: pd.DataFrame,
    extension: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Check the extension agrees with the canonical history where they overlap.

    An offset here would mean the vendor re-anchored its back-adjustment, in
    which case appending the continuation would silently restate the whole
    history and every signal computed across the join would be wrong.
    """

    rows: list[dict[str, Any]] = []
    for root in extension.columns:
        if root not in canonical:
            rows.append(
                {
                    "root": root,
                    "shared_sessions": 0,
                    "maximum_absolute_difference": np.nan,
                    "status": "BLOCKED",
                    "detail": "root is absent from the canonical history",
                }
            )
            continue
        joined = pd.concat(
            [canonical[root].rename("canonical"), extension[root].rename("extension")],
            axis=1,
            join="inner",
        ).dropna()
        if joined.empty:
            rows.append(
                {
                    "root": root,
                    "shared_sessions": 0,
                    "maximum_absolute_difference": np.nan,
                    "status": "BLOCKED",
                    "detail": "no shared sessions to reconcile",
                }
            )
            continue
        error = float((joined["canonical"] - joined["extension"]).abs().max())
        rows.append(
            {
                "root": root,
                "shared_sessions": int(len(joined)),
                "maximum_absolute_difference": error,
                "status": "PASS" if error <= tolerance else "BLOCKED",
                "detail": (
                    "extension continues the canonical series without re-anchoring"
                    if error <= tolerance
                    else "vendor re-anchored the back-adjustment; appending would "
                    "restate history"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_acceptance(
    observed: Mapping[str, float],
    criteria: Mapping[str, float],
) -> pd.DataFrame:
    """Compare the one look against what was written down before it.

    Criteria are lower bounds by convention except for drawdown, which is
    reported as a negative fraction and so is bounded below by its floor.
    Nothing here is a promotion decision; it records whether the pre-declared
    thresholds were met.
    """

    rows: list[dict[str, Any]] = []
    for name, threshold in sorted(dict(criteria).items()):
        if name not in observed:
            rows.append(
                {
                    "criterion": name,
                    "threshold": threshold,
                    "observed": np.nan,
                    "met": False,
                    "detail": "metric was not produced by the evaluation",
                }
            )
            continue
        value = float(observed[name])
        met = value >= threshold
        rows.append(
            {
                "criterion": name,
                "threshold": threshold,
                "observed": value,
                "met": bool(met),
                "detail": "pre-registered before the dataset was read",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "HOLDOUT_CONSUME",
    "HOLDOUT_REGISTER",
    "SCHEMA_VERSION",
    "SUBSET_LIMITATIONS",
    "HoldoutConsumption",
    "HoldoutDataset",
    "HoldoutEvent",
    "HoldoutLedger",
    "HoldoutLedgerError",
    "HoldoutRegistration",
    "evaluate_acceptance",
    "load_extension_frames",
    "seal_dataset",
    "verify_continuity",
]
