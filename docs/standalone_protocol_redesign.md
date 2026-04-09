# Standalone Protocol Redesign

## Goal

Unify standalone `gateway -> interface -> dispatcher -> backend` around an explicit
session protocol so that:

- transport timing no longer determines synthesis semantics
- streaming and offline share the same second-layer split logic
- the first-layer group pre-split is enabled only when the declared input mode needs it
- future `voice_design`, `custom_voice`, `voice_clone_xvec`, `voice_clone_icl`,
  `base`, and `instruct` support can be added without changing the session protocol

## Layer Roles

### Gateway

The gateway is a protocol adapter only.

- Accept `start -> text* -> end/cancel`
- Validate and normalize `SessionConfig`
- Convert outgoing audio to the requested output format
- Do not infer offline/streaming behavior from chunk timing

### Interface

The interface is the external session facade for gateway/triton adapters.

- Owns session lifecycle and callback wiring
- Owns text/token ingestion and `input_mode`-based routing
- Owns ordered audio emission back to the caller
- Does not construct backend requests directly beyond delegating to dispatcher

### Dispatcher

The dispatcher is the backend-facing request translator.

- Converts `SegmentAction` to backend `EngineRequest`
- Emits `NEW_SESSION`, `SESSION_TEXT_DONE`, and cancel/control events
- Preserves split semantics such as `FLUSH_EOS` vs `FLUSH_NOP`

### Spliter

The Spliter owns the two-layer text segmentation policy.

- Layer 1: group pre-split
  - used for `LONG_SEGMENT` and `FULL_TEXT`
  - not used for `TOKEN` / `CLAUSE`
- Layer 2: state-machine-driven split
  - always authoritative for per-segment flushing and decode budget control

### Backend

The backend owns synthesis state.

- Prefill/decode execution
- cache and pause/resume state
- true streaming semantics when text is temporarily unavailable

## Session Protocol

### Capabilities

Before opening a synthesis session, a client may call `GetCapabilities`.

This returns the standalone engine's loaded contract:

- `variant`
- `loaded_model_type`
- `declared_supported_task_types`
- supported input modes / group policies / audio formats
- ref-audio availability for standalone preprocessing

The loaded model type is chosen when the engine is started. Runtime requests do
not switch models; they can only confirm that the client and server are using
the same model contract.

In standalone mode, the external loaded model type and the backend synthesis
branch are related but not identical:

- `base` -> internal `voice_clone` with x-vector path
- `icl` -> internal `voice_clone` with ICL path
- `custom_voice` -> internal `custom_voice`
- `voice_design` -> internal `voice_design`

### Start

`StartRequest` declares a `SessionConfig`.

Important fields:

- `task_type`
- `language`
- `speaker`
- `instruct`
- `ref_audio`
- `ref_text`
- `x_vector_only`
- `input_mode`
- `group_policy`
- `audio`

`task_type` is no longer the runtime model selector. The standalone engine binds
requests to the already loaded model type from the manifest. Clients may omit
`task_type`, or send the same value as an explicit handshake check.

The server also validates model-specific fields before synthesis:

- `base`: requires `ref_audio`
- `icl`: requires `ref_audio` and `ref_text`
- `custom_voice`: rejects ref-audio clone fields
- `voice_design`: requires `instruct`

### Text

`TextChunk` carries text only. Its transport arrival pattern must not change
session semantics.

### End

`EndRequest` means no more text will arrive for this session.

It must not be used to infer whether a session is "offline" or "streaming".

## Input Modes

### TOKEN

- client sends token-scale text updates
- no first-layer group pre-split
- dispatcher forwards tokenized text to layer-2 splitting immediately

### CLAUSE

- client sends clause-scale text updates
- no first-layer group pre-split
- dispatcher forwards clause text directly to layer-2 splitting

### LONG_SEGMENT

- client sends long text units
- each long text unit is pre-split into groups first
- each resulting group is then fed to layer-2 splitting

### FULL_TEXT

- explicit offline mode
- complete text is buffered until `end`
- then layer-1 pre-split runs across the full text

## Group Policy

### AUTO

- use first-layer pre-split when `input_mode` is `LONG_SEGMENT` or `FULL_TEXT`

### NONE

- disable first-layer pre-split even for long input units
- still use layer-2 splitting

## Audio Output Contract

Gateway accepts an `AudioFormat` request and converts engine output from native
`PCM_F32@24kHz` into the requested wire format.

Current implementation supports:

- `PCM_F32`, mono, `24000` or `16000`
- `PCM_S16LE`, mono, `24000` or `16000`

Unsupported combinations should fail explicitly instead of silently degrading.

## Current Implementation Notes

Implemented in standalone engine:

- explicit `SessionConfig` plumbed through gateway, interface, dispatcher, and backend
- interface routing by `input_mode`
- long-segment `push_group_tokens()` path in `Spliter`
- backend prefill no longer waits for `text_complete` if initial text already exists
- `FLUSH_EOS` / `FLUSH_NOP` distinction preserved into backend requests
- streaming pause/resume semantics in backend instead of unconditional pad injection

Still pending for full parity with Triton orchestrator:

- standalone preprocessing for `voice_clone_xvec` / `voice_clone_icl` from `ref_audio`
- complete sampling parameter plumbing
- Triton gateway replacement for the current hand-written gRPC layer
