# ANPR Gate System — Project Scope & Roadmap

## Executive Summary

An AI-powered license plate recognition system that automates vehicle access control at the residential entrance gate. The system reads Colombian license plates (cars and motorcycles) from existing security cameras, matches them against an authorized resident database, and opens the gate automatically — without modifying the existing DVR or camera infrastructure.

---

## Current Status

✅ **Prototype complete and security-audited** (March 2026)
- Colombian plate recognition working (car + motorcycle formats)
- Fine-tuned YOLO model trained on local traffic footage
- 0 HIGH/MEDIUM security vulnerabilities (independently scanned)
- Code published on GitHub with full documentation

---

## Phases

### Phase 1 — Digital Security Audit of DVR System
**Trigger:** Board approval + access authorization
**Duration:** 1 week

| Task | Description |
|---|---|
| DVR reconnaissance | Identify brand, firmware version, open ports via Nmap scan |
| RTSP discovery | Locate and test camera stream URLs |
| Default credential check | Verify DVR is not using factory passwords |
| Vulnerability report | Deliver written report to administrator |

> ⚠️ This phase is **read-only** — no changes to any hardware or configuration.

---

### Phase 2 — Camera Integration
**Trigger:** DVR audit complete, admin approval for read access
**Duration:** 1–2 weeks

| Task | Description |
|---|---|
| Multi-stream support | Connect to 2–3 gate-area cameras simultaneously |
| Camera role assignment | `entry_gate` (1 cam) vs `exit_gate` (2 cams) |
| Exit plate reading | Open gate immediately on first confirmed match from any exit camera |
| Entry plate reading | Read plate, short delay for guard to observe |

---

### Phase 3 — Gate Hardware Integration
**Trigger:** Written authorization to connect hardware to gate
**Duration:** 1 week

| Task | Description |
|---|---|
| USB relay installation | $8–15 relay module wired to gate emergency button terminals |
| Gate trigger logic | Exit: immediate open / Entry: configurable delay |
| Fail-safe | Gate defaults to manual control if software fails |
| Electrician review | Recommended before wiring |

---

### Phase 4 — Production & Handover
**Duration:** 2 weeks

| Task | Description |
|---|---|
| Resident plate database | Admin tool to add/remove authorized plates |
| Audit log | Record every gate open/close event with timestamp and plate |
| Confidence voting | Accumulate across frames to reduce false gates |
| Final security audit | Full scan before live deployment |
| Resident demo | Present to admin and building council |
| Documentation handover | Operations manual for building admin |

---

## What Will NOT Change

- ❌ No changes to existing DVR configuration
- ❌ No changes to existing camera settings or placement  
- ❌ No cloud services — all data stays on the local residence network
- ❌ No subscription costs — fully open-source stack

---

## Hardware Requirements (estimated cost)

| Item | Est. Cost (COP) | Notes |
|---|---|---|
| USB Relay Module (1-ch) | $30,000–$60,000 | Connects PC to gate terminals |
| Dedicated mini-PC (optional) | $400,000–$800,000 | If running 24/7 on dedicated hardware |
| Network cable / switch port | $0–$50,000 | If ANPR PC needs wired connection to DVR network |

---

## Data Privacy (Ley 1581 de 2012)

- License plate data stored **locally only** — no external servers
- Only authorized residents' plates stored in the database
- Residents can request removal of their data at any time
- No biometric or face recognition — plates only

---

## Immediate Next Steps (Before Board Meeting)

1. The board should review this scope document
2. Confirm which cameras are nearest to the entrance gate
3. Provide DVR brand, model, and local network IP
4. Designate a contact for hardware authorization (Phase 3)
