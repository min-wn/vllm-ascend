"""CPU contracts for fixed-query replay experiments, not service routing."""

from dataclasses import dataclass


def _merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def inspect_replay_trace(payload):
    """Measure overlap of device kernels from distinct captured models, not wait tasks."""
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    device_pids = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name" and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    kernels = [
        event
        for event in events
        if event.get("pid") in device_pids
        and event.get("ph") == "X"
        and event.get("args", {}).get("Task Type", "").startswith("KERNEL")
        and float(event.get("dur", 0)) > 0
    ]
    rows = []
    for marker in events:
        name = marker.get("name", "")
        if marker.get("ph") != "X" or not name.startswith("replay."):
            continue
        start = float(marker["ts"])
        end = start + float(marker["dur"])
        models = {}
        for kernel in kernels:
            left = max(start, float(kernel["ts"]))
            right = min(end, float(kernel["ts"]) + float(kernel["dur"]))
            model_id = kernel["args"].get("Model Id")
            if left < right and model_id not in (None, 4294967295):
                models.setdefault(model_id, []).append((left - start, right - start))
        expected = 1 if name.endswith(".single") else 2
        if len(models) != expected:
            raise ValueError(f"{name}: expected {expected} captured models, found {len(models)}")
        intervals = {key: _merged_intervals(values) for key, values in models.items()}
        busy = {str(key): sum(right - left for left, right in values) for key, values in intervals.items()}
        union = _merged_intervals([interval for values in intervals.values() for interval in values])
        union_us = sum(right - left for left, right in union)
        rows.append(
            dict(
                marker=name,
                model_kernel_busy_us=busy,
                kernel_union_us=union_us,
                two_model_kernel_overlap_us=max(0.0, sum(busy.values()) - union_us),
            )
        )
    if not rows:
        raise ValueError("trace contains no replay markers")
    return rows


def inspect_shared_dense_trace(payload):
    """Measure FIA overlap inside one graph as well as between staged graphs."""
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    device_pids = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name" and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    kernels = [
        event
        for event in events
        if event.get("pid") in device_pids
        and event.get("ph") == "X"
        and event.get("args", {}).get("Task Type", "").startswith("KERNEL")
        and float(event.get("dur", 0)) > 0
    ]
    rows = []
    for marker in events:
        name = marker.get("name", "")
        if marker.get("ph") != "X" or not name.startswith("layer."):
            continue
        start = float(marker["ts"])
        end = start + float(marker["dur"])
        selected = []
        for kernel in kernels:
            left = max(start, float(kernel["ts"]))
            right = min(end, float(kernel["ts"]) + float(kernel["dur"]))
            if left < right:
                selected.append((kernel, (left - start, right - start)))
        attention = [
            (kernel, interval) for kernel, interval in selected if "FusedInferAttentionScore" in kernel["name"]
        ]
        expected = 1 if name.startswith("layer.single_") else 2
        if len(attention) != expected:
            raise ValueError(f"{name}: expected {expected} FIA kernels, found {len(attention)}")

        def union_duration(intervals):
            return sum(right - left for left, right in _merged_intervals(intervals))

        attention_sum = sum(right - left for _, (left, right) in attention)
        matmuls = [(kernel, interval) for kernel, interval in selected if "matmul" in kernel["name"].lower()]
        attention_start = min(left for _, (left, _) in attention)
        attention_end = max(right for _, (_, right) in attention)
        inventory = {}
        for kernel, (left, right) in selected:
            item = inventory.setdefault(kernel["name"], dict(count=0, duration_sum_us=0.0))
            item["count"] += 1
            item["duration_sum_us"] += right - left
        host_stages = {
            event["name"]: float(event["dur"])
            for event in events
            if event.get("ph") == "X"
            and event.get("name", "").startswith("host.")
            and start <= float(event["ts"])
            and float(event["ts"]) + float(event["dur"]) <= end
        }
        rows.append(
            dict(
                marker=name,
                kernel_union_us=union_duration([interval for _, interval in selected]),
                kernel_duration_sum_us=sum(right - left for _, (left, right) in selected),
                matmul_count=len(matmuls),
                matmul_duration_sum_us=sum(right - left for _, (left, right) in matmuls),
                attention_count=len(attention),
                attention_stream_count=len({kernel["tid"] for kernel, _ in attention}),
                attention_duration_sum_us=attention_sum,
                attention_overlap_us=max(0.0, attention_sum - union_duration([interval for _, interval in attention])),
                attention_span_us=attention_end - attention_start,
                attention_gap_us=attention_end
                - attention_start
                - union_duration([interval for _, interval in attention]),
                attention_intervals=[
                    dict(stream=kernel["tid"], start_us=left, end_us=right) for kernel, (left, right) in attention
                ],
                kernel_inventory=inventory,
                host_stages_us=host_stages,
            )
        )
    if not rows:
        raise ValueError("trace contains no layer markers")
    return rows


@dataclass(frozen=True)
class ReplayWorkload:
    batch: int
    capacity: int
    query: int
    kv_length: int
    lanes: tuple[int, int]
    block_size: int = 128

    def __post_init__(self):
        if self.batch < 2 or self.capacity < self.batch:
            raise ValueError("capacity must cover at least two actual requests")
        if self.query not in (1, 2, 4, 8) or self.kv_length < self.query:
            raise ValueError("query must be 1/2/4/8 and no longer than KV")
        if len(self.lanes) != 2 or min(self.lanes) <= 0 or sum(self.lanes) != self.batch:
            raise ValueError("two positive request lanes must cover the actual batch")
        if self.block_size != 128:
            raise ValueError("this experiment requires 128-token KV pages")

    @property
    def pages_per_request(self):
        return (self.kv_length + self.block_size - 1) // self.block_size

    @property
    def kv_blocks(self):
        return self.capacity * self.pages_per_request

    def metadata_values(self, start, stop):
        if not 0 <= start < stop <= self.capacity:
            raise ValueError("invalid request interval")
        width = self.pages_per_request
        positions = list(range(self.kv_length - self.query, self.kv_length))
        return {
            "boundaries": [i * self.query for i in range(stop - start + 1)],
            "lengths": [self.kv_length] * (stop - start),
            "pages": [list(range(i * width, (i + 1) * width)) for i in range(start, stop)],
            "slots": [i * width * self.block_size + p for i in range(start, stop) for p in positions],
            "positions": positions * (stop - start),
        }

    def graph_values(self, start, stop):
        if not 0 <= start < min(stop, self.batch) <= stop <= self.capacity:
            raise ValueError("graph must contain actual requests within capacity")
        active_stop = min(stop, self.batch)
        values = self.metadata_values(start, active_stop)
        padding = stop - active_stop
        values["boundaries"] = [i * self.query for i in range(stop - start + 1)]
        values["lengths"] += [0] * padding
        values["pages"] += [[0] * self.pages_per_request for _ in range(padding)]
        values["slots"] += [-1] * (padding * self.query)
        values["positions"] += [0] * (padding * self.query)
        return values
