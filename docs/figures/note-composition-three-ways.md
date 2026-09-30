# Note: three ways to compose the organ with a model

*The note behind Figure 3 of the whitepaper (§3.6, composing the organ with other models), also shown in the README under "Use it with your own model". Drawn on 2026-09-30 at the founder's request (register Rev80), from the code and the record. The figure is `composition-three-ways.svg`, original, and carries no measured number; the organ is drawn identically in the three panels because it is the same object in the three ways.*

![Three ways to compose the organ with a model](composition-three-ways.svg)

**Figure 3.** Three ways to compose the organ with a model. In every panel the organ is the same: a journal sealed per scope with one key per journal, a gate on the write, an index on the read, a lifecycle; a cue goes in, a window of retrieved lines comes out and rises to the model; the model's answer, cite a label and give the attribute or abstain, goes to a harness that maps the label to the pointer and verifies it. What changes between the panels is the model, how the window enters it, where the cue comes from, whether anything is trained, and what the record has measured.

**(i) As a library, any model decoding.** The persistent tier (`Journal`, `WritePath`, `JournalPath`, `lifecycle`) around a model of any size behind any runtime; the window enters as text in the integrator's own prompt or harness; the cue is the integrator's encoder or the protocol's hash seeded address; no training. It inherits the organ's guarantees by design (whitepaper §4.9) and the organ level rows of the record; the pair of a model and the organ is the integrator's to measure, with the protocol.

**(ii) A model people use, through the frozen prompt.** `cortex_c2b.external_arm` around an open model with frozen weights at a pinned revision, in its published dtype; the window enters as text in a frame whose hash is in every record; the cue is the organ's hash seeded address; no training. First execution retained and not graved (the agreement of two executions on one hardware is pending): the episodic conditions hold and the skill arm fails at the size run, the prompt's cost on the model's ordinary text (§5d, door 2).

**(iii) Trained with the organ, in the decode loop.** `cortex_c2b.lm_bridge` and `cortex_c2b.organ_use`: the window enters as content through a zero initialised cross attention gated by the router; the cue encoder is learned by contrast; one fine tune teaches the contract and the organ's use. The model alone is INVALID at this size, attributed; the system, the model deciding and the organ answering under the organ side policy, holds the four thresholds on two live seeds (§4.7b).

The line under the panels is the rule of §7a: a result on any model, PASS, FAIL or INVALID, is a row of the record with its attribution. The organ is the same in every row.
