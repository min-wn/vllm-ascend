# Matching The Installed FLA Tiling

The installed wheel is `flash_linear_attention_npu_a2-26.9.1+deva4a7958`.
Its metadata identifies the upstream repository as
<https://github.com/flashserve/flash-linear-attention-npu>.
The complete revision for the wheel's commit prefix is
`a4a795857868729fbdbb0118c1e06c8655a7b31b`.

The matching host calculation is in
`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling_processor.h`,
specifically `CalcFixedUbBytes`, `CalcWorkingUbBytes`, `CalcVStepCoeff`,
`EvaluateBufferProfile`, `IsBetterProfile`, and `FinalizeVStepFromUb`.

Remote CANN platform configuration
`/usr/local/Ascend/cann-9.1.0/aarch64-linux/data/platform_config/Ascend910B3.ini`
reports `ub_size=196608`. For the qualified profile, the inputs to the host
calculation are BF16 state, Dk=Dv=128, aligned Nv=32, MAX_MTP=8,
scalar gamma present, per-key gamma absent:

| Quantity | Bytes or value |
| --- | ---: |
| Fixed buffers | `8*(4*128+2*128)+128+8*4*32+8*2*32 = 7808` |
| Fixed plus working buffers | `7808+8*(8*128+4*128+4*32) = 21120` |
| Per-V-row coefficient, queues 1/1 | `1548` |
| Per-V-row coefficient, queues 1/2 | `1552` |
| Per-V-row coefficient, queues 2/2 | `1808` |

Each candidate first computes
`floor((196608-21120)/coefficient/8)*8`. This yields 112, 112 and 96.
For each candidate, `repeatTime=ceil(128/initialVStep)=2`; balancing the two
iterations gives `vStep=ceil_align(ceil(128/2),8)=64` in every case.
The upstream tie-break then prefers the larger total queue depth, selecting
state-output depth 2 and attention-output depth 2. V1 used vStep 16, hence
eight V iterations rather than the official two.

V2 follows the selected tile and depths. It deliberately reserves only the
working tensor bytes in `ubRestBytes`, rather than upstream's entire unused
UB remainder, because the prototype adds two queued log buffers and one old
log buffer. At Nv24, its VECCALC reservation is 81408 bytes. The inherited
queues, this reservation, 6400 bytes of new log output buffers and 4224 bytes
of previous log records fit within the documented 196608-byte UB. No opaque
tiling blobs, installed OPP registration, or live package sources are modified.

These calculations explain a candidate source of V1 overhead. They do not
establish how much latency V2 recovers; that requires the dense/compact
qualification and the complete step measurements.
