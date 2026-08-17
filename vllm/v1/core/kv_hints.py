# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import enum
import heapq
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)

logger = init_logger(__name__)


class HintState(enum.IntEnum):
    NORMAL = 0
    WILLNEED_SOON = 1
    WILLNEED_LATER = 2
    DEAD = 3
    WILLNEED_SHADOW = 4


@dataclass
class HintConfig:
    # TODO: add toggle in startup args
    enabled: bool = True

    shadow_mode: bool = False
    transfer_rtt_ms: float = 50.0

    bucket_split_ms: float | None = None
    bucket_split_frac: float = 0.5

    max_ttl_ms: float = 30_000.0

    pressure_threshold: float = 0.85

    recent_capacity: int = 512
    adaptive_rtt: bool = True
    adaptive_rtt_alpha: float = 0.2
    adaptive_rtt_min_samples: int = 20

    transfer_rtt_floor_ms: float = 0.0

    dontneed_requires_pressure: bool = False

    done_is_pageout: bool = False

    missing_estimate_action: str = "noop"

    protect_shared_run: bool = False

    record_chains: bool = True

    bytes_per_block: int = 0

    def __post_init__(self):
        if self.missing_estimate_action not in ("noop", "static"):
            raise ValueError(
                f"missing_estimate_action must be 'noop' or 'static', "
                f"got {self.missing_estimate_action!r}"
            )
        if self.bucket_split_ms is not None:
            if self.adaptive_rtt:
                logger.warning(
                    "bucket_split_ms is set (%.1f) while adaptive_rtt is on: "
                    "the split will not track the per-chain rtt that admits "
                    "blocks to willneed, so one bucket may stay empty. Leave "
                    "bucket_split_ms=None to split at %.2f * rtt.",
                    self.bucket_split_ms,
                    self.bucket_split_frac,
                )
            if self.bucket_split_ms >= self.transfer_rtt_ms:
                logger.warning(
                    "bucket_split_ms (%.1f) >= transfer_rtt_ms (%.1f): "
                    "_q_later will be empty on the static path, so bucket "
                    "ordering is inert.",
                    self.bucket_split_ms,
                    self.transfer_rtt_ms,
                )
        if not 0.0 < self.bucket_split_frac < 1.0:
            logger.warning(
                "bucket_split_frac=%.3f outside (0, 1): one willneed bucket "
                "will be empty by construction.",
                self.bucket_split_frac,
            )


@dataclass
class HintStats:
    hints_received: int = 0
    hints_unknown_req: int = 0
    hints_no_blocks: int = 0

    hints_missing_estimate: int = 0

    blocks_willneed: int = 0
    blocks_dontneed: int = 0
    blocks_pageout: int = 0
    blocks_expired: int = 0
    blocks_noop: int = 0

    hints_willneed: int = 0
    hints_pageout: int = 0
    hints_dontneed: int = 0
    hints_noop: int = 0

    blocks_willneed_soon: int = 0
    blocks_willneed_later: int = 0

    ttl_sum_willneed: float = 0.0
    ttl_sum_pageout: float = 0.0
    rtt_sum_willneed: float = 0.0
    rtt_sum_pageout: float = 0.0

    rtt_static_decisions: int = 0
    rtt_adaptive_decisions: int = 0

    blocks_dontneed_shadow: int = 0
    blocks_pageout_shadow: int = 0

    blocks_willneed_shadow: int = 0

    shadow_willneed_survived: int = 0
    shadow_willneed_evicted: int = 0

    blocks_resumed: int = 0

    blocks_dead_resurrected: int = 0

    blocks_stolen_dead: int = 0
    blocks_stolen_later: int = 0
    blocks_stolen_soon: int = 0

    apply_seconds: float = 0.0
    reap_seconds: float = 0.0

    record_seconds: float = 0.0
    record_calls: int = 0

    hint_queue_reached: int = 0

    head_resident_probes: int = 0
    head_absent: int = 0
    head_refcnt_zero: int = 0
    dead_included_root: int = 0
    head_protected: int = 0

    resurrect_head: int = 0
    resurrect_tail: int = 0

    shared_run_samples: int = 0
    shared_run_sum: int = 0
    shared_run_min: int = -1
    distinct_roots: int = 0
    hash_chain_entries: int = 0

    chain_blocks_sum: int = 0
    candidate_blocks_sum: int = 0
    skipped_shared_sum: int = 0
    skipped_refcnt_sum: int = 0
    skipped_absent_sum: int = 0
    skipped_floor_sum: int = 0

    blocks_from_normal: int = 0
    alloc_normal_after_dead: int = 0

    hinted_inside_shared_run: int = 0
    hint_lag_steps_sum: int = 0
    hint_lag_max: int = 0
    hint_lag_samples: int = 0

    prologue_probe_samples: int = 0
    prologue_run_sum: int = 0
    prologue_resident_sum: int = 0
    prologue_referenced_sum: int = 0
    prologue_first_hole_sum: int = 0

    remote_kv_waiters_sum: int = 0
    inflight_reserved_sum: int = 0
    census_samples: int = 0

    first_miss_probes: int = 0
    first_miss_was_hinted: int = 0
    first_miss_same_seq: int = 0
    first_miss_other_seq: int = 0
    first_miss_unknown_seq: int = 0
    first_miss_hint_age_sum: int = 0

    transfer_samples: int = 0
    transfer_ewma_ms_per_byte: float = 0.0
    transfer_rtt_static_ms: float = 0.0
    transfer_rtt_adaptive: int = 0

    def as_dict(self) -> dict[str, float]:
        d = dict(self.__dict__)
        if self.shared_run_samples:
            d["shared_run_mean"] = self.shared_run_sum / self.shared_run_samples
            n = self.shared_run_samples
            d["chain_blocks_mean"] = self.chain_blocks_sum / n
            d["candidate_blocks_mean"] = self.candidate_blocks_sum / n
            if self.chain_blocks_sum:
                d["candidate_fraction"] = (
                    self.candidate_blocks_sum / self.chain_blocks_sum
                )
        if self.record_calls:
            d["record_ms_mean"] = 1000.0 * self.record_seconds / self.record_calls
        if self.hints_willneed:
            d["ttl_ms_mean_willneed"] = self.ttl_sum_willneed / self.hints_willneed
            d["rtt_ms_mean_willneed"] = self.rtt_sum_willneed / self.hints_willneed
        if self.hints_pageout:
            d["ttl_ms_mean_pageout"] = self.ttl_sum_pageout / self.hints_pageout
            d["rtt_ms_mean_pageout"] = self.rtt_sum_pageout / self.hints_pageout
        decisions = self.hints_willneed + self.hints_pageout
        if decisions:
            d["willneed_decision_rate"] = self.hints_willneed / decisions
        if self.prologue_probe_samples:
            n = self.prologue_probe_samples
            d["prologue_first_hole_mean"] = self.prologue_first_hole_sum / n
            d["prologue_resident_mean"] = self.prologue_resident_sum / n
            d["prologue_referenced_mean"] = self.prologue_referenced_sum / n
            d["prologue_run_mean"] = self.prologue_run_sum / n
        if self.hint_lag_samples:
            d["hint_lag_steps_mean"] = self.hint_lag_steps_sum / self.hint_lag_samples
        if self.first_miss_was_hinted:
            d["first_miss_hint_age_mean"] = (
                self.first_miss_hint_age_sum / self.first_miss_was_hinted
            )
        shadow_total = self.shadow_willneed_survived + self.shadow_willneed_evicted
        if shadow_total:
            d["shadow_willneed_survival_rate"] = (
                self.shadow_willneed_survived / shadow_total
            )
        return d


class TransferMonitor:
    """EWMA of observed CPU->GPU KV reload time per byte."""

    def __init__(self, cfg: HintConfig):
        self.cfg = cfg
        self.ewma_ms_per_byte: float | None = None
        self.n = 0
        self.recent = deque(maxlen=256)

    def observe(self, seconds: float, num_bytes: int) -> None:
        """One completed CPU->GPU load batch, from the offload connector."""
        if num_bytes <= 0 or seconds <= 0:
            return
        ms_per_byte = (seconds * 1000.0) / num_bytes
        self.recent.append(ms_per_byte)
        self.n += 1
        a = self.cfg.adaptive_rtt_alpha
        self.ewma_ms_per_byte = (
            ms_per_byte
            if self.ewma_ms_per_byte is None
            else a * ms_per_byte + (1 - a) * self.ewma_ms_per_byte
        )

    def is_adaptive(self) -> bool:
        """Whether effective_rtt_ms would use the EWMA right now."""
        return (
            self.cfg.adaptive_rtt
            and self.ewma_ms_per_byte is not None
            and self.n >= self.cfg.adaptive_rtt_min_samples
            and self.cfg.bytes_per_block > 0
        )

    def effective_rtt_ms(self, chain_blocks: int) -> float:
        if not self.is_adaptive():
            return self.cfg.transfer_rtt_ms
        est = self.ewma_ms_per_byte * max(chain_blocks, 1) * self.cfg.bytes_per_block
        est = max(est, self.cfg.transfer_rtt_floor_ms)
        return min(max(est, 1.0), self.cfg.max_ttl_ms)


class KVHintManager:
    """Owns the secondary free queues, the expiry heap, and the recent map.

    BlockPool delegates here so the diff against vLLM stays small.
    """

    def __init__(self, config: HintConfig):
        self.config = config

        self._q_soon = FreeKVCacheBlockQueue([])
        self._q_later = FreeKVCacheBlockQueue([])

        self._q_dead = FreeKVCacheBlockQueue([])

        self._expiry: list[tuple[float, int, int, KVCacheBlock]] = []
        self._seq = 0
        self._epoch = 0

        self._recent: OrderedDict[str, list[BlockHashWithGroupId]] = OrderedDict()

        self.transfers = TransferMonitor(config)
        self.stats = HintStats()

        self._hash_chains: Counter[BlockHashWithGroupId] = Counter()
        self._tips: dict[str, tuple[BlockHashWithGroupId, int, str | None]] = {}
        self._roots: Counter[BlockHashWithGroupId] = Counter()

        self.sched_step = 0
        self._recorded_at: dict[str, int] = {}
        self._hinted: OrderedDict[BlockHashWithGroupId, tuple[int, str | None]] = (
            OrderedDict()
        )
        self._hinted_cap = 200_000

    def _queue_for(self, state: int) -> FreeKVCacheBlockQueue:
        if state == HintState.WILLNEED_SOON:
            return self._q_soon
        if state == HintState.WILLNEED_LATER:
            return self._q_later
        if state == HintState.DEAD:
            return self._q_dead
        raise AssertionError(f"no hint queue for state {state}")

    @property
    def num_free_blocks(self) -> int:
        return (
            self._q_dead.num_free_blocks
            + self._q_later.num_free_blocks
            + self._q_soon.num_free_blocks
        )

    @property
    def num_dead_blocks(self) -> int:
        return self._q_dead.num_free_blocks

    def note_normal_alloc(self, n: int) -> None:
        """Called from BlockPool.get_new_blocks when the corpse queue ran dry
        and the allocation had to reach into live LRU material."""
        if n > 0:
            self.stats.blocks_from_normal += n
            self.stats.alloc_normal_after_dead += 1

    def remove(self, block: KVCacheBlock) -> None:
        """Unlink a block from whichever hint queue holds it. Called from
        touch() on a prefix-cache hit.
        """
        state = block.hint_state
        self._queue_for(state).remove(block)
        block.hint_state = HintState.NORMAL
        block.hint_epoch = -1
        if state == HintState.DEAD:
            self.stats.blocks_dead_resurrected += 1
            if block.hint_in_head:
                self.stats.resurrect_head += 1
            else:
                self.stats.resurrect_tail += 1
            block.hint_in_head = False
        else:
            self.stats.blocks_resumed += 1

    def note_shadow_survived(self, block: KVCacheBlock) -> None:
        self.stats.shadow_willneed_survived += 1
        block.hint_state = HintState.NORMAL
        block.hint_epoch = -1

    def note_shadow_evicted(self, blocks: list[KVCacheBlock]) -> None:
        for b in blocks:
            if b.hint_state == HintState.WILLNEED_SHADOW:
                self.stats.shadow_willneed_evicted += 1
                b.hint_state = HintState.NORMAL
                b.hint_epoch = -1

    def popleft_dead_n(self, n: int) -> list[KVCacheBlock]:
        """Allocate corpses. Drained BEFORE the normal queue."""
        take = min(n, self._q_dead.num_free_blocks)
        if take <= 0:
            return []
        blocks = self._q_dead.popleft_n(take)
        for b in blocks:
            b.hint_state = HintState.NORMAL
            b.hint_epoch = -1
            b.hint_in_head = False
        self.stats.blocks_stolen_dead += len(blocks)
        return blocks

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Allocate from the willneed queues. Drained AFTER the normal queue.
        LATER before SOON."""
        out: list[KVCacheBlock] = []
        for q, counter in (
            (self._q_later, "blocks_stolen_later"),
            (self._q_soon, "blocks_stolen_soon"),
        ):
            if len(out) >= n:
                break
            take = min(n - len(out), q.num_free_blocks)
            if take <= 0:
                continue
            blocks = q.popleft_n(take)
            for b in blocks:
                b.hint_state = HintState.NORMAL
                b.hint_epoch = -1
            setattr(self.stats, counter, getattr(self.stats, counter) + len(blocks))
            out.extend(blocks)
        if out:
            self.stats.hint_queue_reached += 1
        return out

    def _decr(self, hashes) -> None:
        """Subtract a chain and prune anything that reaches zero, in one
        O(len(chain)) pass."""
        c = self._hash_chains
        for h in hashes:
            v = c.get(h, 0) - 1
            if v > 0:
                c[h] = v
            else:
                if v < 0:
                    logger.warning(
                        "hash_chains went negative for %s: subtract accounting "
                        "bug (a chain was decremented without being added)",
                        h,
                    )
                c.pop(h, None)

    def shared_run(self, hashes: list[BlockHashWithGroupId]) -> int:
        """Number of leading blocks present in >= 2 recorded chains."""
        i = 0
        n = len(hashes)
        while i < n and self._hash_chains[hashes[i]] >= 2:
            i += 1
        return i

    def record_finished(self, req_id, hashes, seq_id=None) -> None:
        """Record a freed request's chain."""
        self._recorded_at[req_id] = self.sched_step
        if not hashes:
            return
        t0 = time.perf_counter()

        old = self._recent.pop(req_id, None)
        if old is not None:
            self._decr(old)
            self._tips.pop(req_id, None)

        for old_id, (tip, L, old_seq) in list(self._tips.items()):
            if seq_id is not None and old_seq is not None and old_seq != seq_id:
                continue
            if len(hashes) >= L and hashes[L - 1] == tip:
                prev = self._recent.pop(old_id, None)
                if prev is not None:
                    self._decr(prev)
                self._tips.pop(old_id, None)

        self._roots[hashes[0]] += 1
        self._recent[req_id] = hashes
        self._tips[req_id] = (hashes[-1], len(hashes), seq_id)
        self._hash_chains.update(hashes)

        while len(self._recent) > self.config.recent_capacity:
            oid, dropped = self._recent.popitem(last=False)
            self._decr(dropped)
            self._tips.pop(oid, None)
            self._recorded_at.pop(oid, None)

        self.stats.distinct_roots = len(self._roots)
        self.stats.hash_chain_entries = len(self._hash_chains)
        self.stats.record_seconds += time.perf_counter() - t0
        self.stats.record_calls += 1

    def sync_transfer_stats(self) -> None:
        """TransferMonitor state lives outside HintStats; mirror it in before
        serialising or it is invisible to every consumer."""
        self.stats.transfer_samples = self.transfers.n
        self.stats.transfer_ewma_ms_per_byte = self.transfers.ewma_ms_per_byte or 0.0
        self.stats.transfer_rtt_static_ms = self.config.transfer_rtt_ms
        self.stats.transfer_rtt_adaptive = int(self.transfers.is_adaptive())

    def _note_hinted(self, blocks, sequence_id) -> None:
        for b in blocks:
            self._hinted[b.block_hash] = (self.sched_step, sequence_id)
            self._hinted.move_to_end(b.block_hash)
        while len(self._hinted) > self._hinted_cap:
            self._hinted.popitem(last=False)

    def apply(
        self,
        req_id: str,
        *,
        expect_return_ms: float | None,
        done: bool,
        pool,
        sequence_id: str | None = None,
    ) -> dict:
        """Apply a hint. `pool` is the BlockPool, passed rather than held to
        avoid a reference cycle."""
        t0 = time.perf_counter()
        if not self.config.enabled:
            return {"applied": 0, "reason": "disabled"}
        self.stats.hints_received += 1

        hashes = self._recent.get(req_id)
        if hashes is None:
            logger.warning(
                "hint req=%s unknown; _recent has %d entries, recent keys=%s",
                req_id,
                len(self._recent),
                list(self._recent.keys())[-3:],
            )
            self.stats.hints_unknown_req += 1
            return {"applied": 0, "reason": "unknown_request_id"}

        run = self.shared_run(hashes)
        self.stats.shared_run_samples += 1
        self.stats.shared_run_sum += run
        if self.stats.shared_run_min < 0 or run < self.stats.shared_run_min:
            self.stats.shared_run_min = run

        floor = run if self.config.protect_shared_run else 0

        skipped_shared = skipped_refcnt = skipped_floor = skipped_absent = 0
        blocks: list[KVCacheBlock] = []
        idx_of: dict[int, int] = {}
        for i, h in enumerate(hashes):
            if i < floor:
                skipped_floor += 1
                continue
            if self._hash_chains[h] >= 2:
                skipped_shared += 1
                continue
            b = pool.cached_block_hash_to_block.get_one_block(h)
            if b is None or b.is_null or b.block_hash != h:
                skipped_absent += 1
                continue
            if b.ref_cnt != 0:
                skipped_refcnt += 1
                continue
            blocks.append(b)
            idx_of[b.block_id] = i
        if skipped_floor:
            self.stats.head_protected += 1

        self.stats.chain_blocks_sum += len(hashes)
        self.stats.candidate_blocks_sum += len(blocks)
        self.stats.skipped_shared_sum += skipped_shared
        self.stats.skipped_refcnt_sum += skipped_refcnt
        self.stats.skipped_absent_sum += skipped_absent
        self.stats.skipped_floor_sum += skipped_floor

        root = hashes[0]
        rb = pool.cached_block_hash_to_block.get_one_block(root)
        self.stats.head_resident_probes += 1
        if rb is None:
            self.stats.head_absent += 1
            root_rc = -1
        else:
            root_rc = rb.ref_cnt
            if root_rc == 0:
                self.stats.head_refcnt_zero += 1

        first_idx = min(idx_of.values()) if idx_of else -1
        last_idx = max(idx_of.values()) if idx_of else -1
        if first_idx == 0:
            self.stats.dead_included_root += 1

        rec = self._recorded_at.pop(req_id, None)
        lag = -1
        if rec is not None:
            lag = self.sched_step - rec
            self.stats.hint_lag_steps_sum += lag
            self.stats.hint_lag_samples += 1
            self.stats.hint_lag_max = max(self.stats.hint_lag_max, lag)

        if 0 <= first_idx < run:
            self.stats.hinted_inside_shared_run += 1

        if not blocks:
            self.stats.hints_no_blocks += 1
            logger.info(
                "hint req=%s done=%s chain=%d run=%d floor=%d shared=%d "
                "refcnt=%d absent=%d cand=0 action=no_blocks lag=%d "
                "usage=%.3f",
                req_id,
                done,
                len(hashes),
                run,
                floor,
                skipped_shared,
                skipped_refcnt,
                skipped_absent,
                lag,
                pool.get_usage(),
            )
            self.stats.apply_seconds += time.perf_counter() - t0
            return {"applied": 0, "reason": "no_resident_blocks"}

        action, ttl, rtt = self._decide(blocks, done, expect_return_ms, pool)
        shadow = self.config.shadow_mode
        adaptive = self.transfers.is_adaptive()
        bucket = None

        logger.info(
            "hint req=%s done=%s chain=%d run=%d floor=%d shared=%d refcnt=%d "
            "absent=%d cand=%d ttl=%.1f rtt=%.1f adaptive=%s action=%s "
            "shadow=%s lag=%d root_rc=%d root_count=%d roots=%d idx=[%s..%s] "
            "usage=%.3f",
            req_id,
            done,
            len(hashes),
            run,
            floor,
            skipped_shared,
            skipped_refcnt,
            skipped_absent,
            len(blocks),
            ttl,
            rtt,
            adaptive,
            action,
            shadow,
            lag,
            root_rc,
            self._hash_chains[root],
            len(self._roots),
            first_idx,
            last_idx,
            pool.get_usage(),
        )

        if action == "dontneed":
            self.stats.hints_dontneed += 1
            if shadow:
                self.stats.blocks_dontneed_shadow += len(blocks)
                self._note_hinted(blocks, sequence_id)
            else:
                for b in blocks:
                    b.hint_in_head = idx_of.get(b.block_id, len(hashes)) < run
                self._to_dead(list(reversed(blocks)), pool)
                self.stats.blocks_dontneed += len(blocks)
                self._note_hinted(blocks, sequence_id)
        elif action == "pageout":
            self.stats.hints_pageout += 1
            self.stats.ttl_sum_pageout += ttl
            self.stats.rtt_sum_pageout += rtt
            if shadow:
                self.stats.blocks_pageout_shadow += len(blocks)
                self._note_hinted(blocks, sequence_id)
            else:
                self._demote(list(reversed(blocks)), pool, front=True)
                self.stats.blocks_pageout += len(blocks)
                self._note_hinted(blocks, sequence_id)
        elif action == "willneed":
            self.stats.hints_willneed += 1
            self.stats.ttl_sum_willneed += ttl
            self.stats.rtt_sum_willneed += rtt
            if shadow:
                for b in blocks:
                    b.hint_state = HintState.WILLNEED_SHADOW
                self.stats.blocks_willneed_shadow += len(blocks)
                bucket = "shadow"
            else:
                bucket = self._promote(blocks, pool, ttl_ms=ttl, rtt_ms=rtt)
                self.stats.blocks_willneed += len(blocks)
        else:
            self.stats.hints_noop += 1
            self.stats.blocks_noop += len(blocks)

        self.stats.apply_seconds += time.perf_counter() - t0
        return {
            "applied": len(blocks),
            "action": action,
            "shared_run": run,
            "chain": len(hashes),
            "ttl_ms": round(ttl, 1),
            "rtt_ms": round(rtt, 1),
            "adaptive": adaptive,
            "bucket": bucket,
        }

    def _decide(self, blocks, done, expect_return_ms, pool) -> tuple[str, float, float]:
        """Returns (action, ttl_ms, rtt_ms)."""
        under_pressure = pool.get_usage() >= self.config.pressure_threshold
        rtt = self.transfers.effective_rtt_ms(len(blocks))

        if done:
            if self.config.dontneed_requires_pressure and not under_pressure:
                return "noop", 0.0, rtt
            if self.config.done_is_pageout:
                return "pageout", 0.0, rtt
            return "dontneed", 0.0, rtt

        if self.transfers.is_adaptive():
            self.stats.rtt_adaptive_decisions += 1
        else:
            self.stats.rtt_static_decisions += 1

        if expect_return_ms is None:
            self.stats.hints_missing_estimate += 1
            if self.config.missing_estimate_action == "noop":
                return "noop", 0.0, rtt
            expect_return_ms = self.config.transfer_rtt_ms

        ttl = min(float(expect_return_ms), self.config.max_ttl_ms)

        return ("willneed" if ttl <= rtt else "pageout"), ttl, rtt

    def _bucket_split_ms(self, rtt_ms: float) -> float:
        """Deadline that separates _q_soon from _q_later."""
        if self.config.bucket_split_ms is not None:
            return self.config.bucket_split_ms
        return self.config.bucket_split_frac * rtt_ms

    def _promote(
        self, blocks: list[KVCacheBlock], pool, ttl_ms: float, rtt_ms: float
    ) -> str:
        deadline = time.monotonic() + ttl_ms / 1000.0
        split = self._bucket_split_ms(rtt_ms)
        soon = ttl_ms <= split
        target = self._q_soon if soon else self._q_later
        state = HintState.WILLNEED_SOON if soon else HintState.WILLNEED_LATER

        self._epoch += 1
        epoch = self._epoch
        moved: list[KVCacheBlock] = []
        for b in blocks:
            if b.hint_state == HintState.NORMAL:
                pool.free_block_queue.remove(b)
            else:
                self._queue_for(b.hint_state).remove(b)
            b.hint_state = state
            b.hint_epoch = epoch
            self._seq += 1
            heapq.heappush(self._expiry, (deadline, self._seq, epoch, b))
            moved.append(b)
        target.append_n(list(reversed(moved)))

        if soon:
            self.stats.blocks_willneed_soon += len(moved)
        else:
            self.stats.blocks_willneed_later += len(moved)
        return "soon" if soon else "later"

    def _demote(self, blocks: list[KVCacheBlock], pool, front: bool) -> None:
        moved: list[KVCacheBlock] = []
        for b in blocks:
            if b.hint_state != HintState.NORMAL:
                self._queue_for(b.hint_state).remove(b)
                b.hint_state = HintState.NORMAL
            else:
                pool.free_block_queue.remove(b)
            b.hint_epoch = -1
            moved.append(b)
        if front:
            pool.free_block_queue.prepend_n(moved)
        else:
            pool.free_block_queue.append_n(moved)

    def _to_dead(self, blocks: list[KVCacheBlock], pool) -> None:
        """Move a finished trajectory's blocks into _q_dead."""
        moved: list[KVCacheBlock] = []
        for b in blocks:
            if b.hint_state == HintState.DEAD:
                continue
            if b.hint_state != HintState.NORMAL:
                self._queue_for(b.hint_state).remove(b)
            else:
                pool.free_block_queue.remove(b)
            b.hint_state = HintState.DEAD
            b.hint_epoch = -1
            moved.append(b)
        self._q_dead.append_n(moved)

    def reap(self, pool, now: float | None = None) -> int:
        """Expire overdue hints. Called once per scheduler step.
        Expired blocks go to the rear of the normal queue.
        """
        if not self._expiry:
            return 0
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        stale: list[KVCacheBlock] = []
        while self._expiry and self._expiry[0][0] <= now:
            _, _, epoch, block = heapq.heappop(self._expiry)
            if block.hint_epoch != epoch:
                continue
            if block.hint_state not in (
                HintState.WILLNEED_SOON,
                HintState.WILLNEED_LATER,
            ):
                continue
            self._queue_for(block.hint_state).remove(block)
            block.hint_state = HintState.NORMAL
            block.hint_epoch = -1
            stale.append(block)
        if stale:
            pool.free_block_queue.append_n(list(reversed(stale)))
            self.stats.blocks_expired += len(stale)
        self.stats.reap_seconds += time.perf_counter() - t0
        return len(stale)

    def probe_first_miss(
        self,
        block_hashes: list[BlockHashWithGroupId],
        num_hit_blocks: int,
        req_id: str,
    ) -> None:
        if num_hit_blocks >= len(block_hashes):
            return
        h = block_hashes[num_hit_blocks]
        self.stats.first_miss_probes += 1
        rec = self._hinted.get(h)
        if rec is None:
            return
        step, owner = rec
        self.stats.first_miss_was_hinted += 1
        self.stats.first_miss_hint_age_sum += self.sched_step - step
        logger.info(
            "first_miss_hinted req=%s hit_blocks=%d chain=%d "
            "hinted_for_seq=%s hint_age_steps=%d",
            req_id,
            num_hit_blocks,
            len(block_hashes),
            owner,
            self.sched_step - step,
        )

    def probe_prologue(self, pool) -> None:
        """Walk the shared prologue and record where the cache lookup dies."""
        if not self._recent:
            return
        chain = next(reversed(self._recent.values()))
        run = self.shared_run(chain)
        if run == 0:
            return
        resident = referenced = 0
        first_hole = -1
        for i, h in enumerate(chain[:run]):
            b = pool.cached_block_hash_to_block.get_one_block(h)
            if b is None or b.block_hash != h:
                if first_hole < 0:
                    first_hole = i
                continue
            resident += 1
            if b.ref_cnt > 0:
                referenced += 1
        s = self.stats
        s.prologue_probe_samples += 1
        s.prologue_run_sum += run
        s.prologue_resident_sum += resident
        s.prologue_referenced_sum += referenced
        s.prologue_first_hole_sum += first_hole if first_hole >= 0 else run

    def drain_all(self, pool) -> None:
        """Return every held block to the normal queue. Used by
        reset_prefix_cache, which asserts on total free-block count.
        """
        for q in (self._q_dead, self._q_later, self._q_soon):
            n = q.num_free_blocks
            if n:
                blocks = q.popleft_n(n)
                for b in blocks:
                    b.hint_state = HintState.NORMAL
                    b.hint_epoch = -1
                    b.hint_in_head = False
                pool.free_block_queue.prepend_n(blocks)
        self._expiry.clear()
        self._recent.clear()
        self._tips.clear()
        self._hash_chains.clear()
        self.stats.hash_chain_entries = 0
        self._recorded_at.clear()
        self._hinted.clear()

    def reset_stats(self) -> None:
        self.stats = HintStats()
        self.stats.distinct_roots = len(self._roots)
        self.stats.hash_chain_entries = len(self._hash_chains)
