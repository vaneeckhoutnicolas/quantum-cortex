# Note: where the common pipeline and quantum-cortex part ways

*The note behind Figure 1 of the whitepaper (§1.5, relation to other approaches), with the short form used in the README. Drafted on 2026-09-30 in the advisory conversation, then corrected against the record on two points before entering the paper (register Rev76): the v series is a post training stage, so the cortex does not have "no post training" but a different one; and the recurrent base arms are measured at stage A with stage B declared and not run, not "under test". The figure is `pipeline-vs-cortex.svg`, original, and carries no measured number.*

![The common pipeline and quantum-cortex, side by side](pipeline-vs-cortex.svg)

**Figure 1.** The common pipeline and quantum-cortex, side by side. Steps 1 to 5 are shared; the two part at step 6 and again at the memory.

Popular accounts of how a model such as ChatGPT is made list six steps: data, tokens, numbers, a transformer, pretraining, post training. quantum-cortex shares the first five, at a scale roughly four orders of magnitude smaller, and departs at the sixth, twice.

In the common pipeline, post training is where behaviour is put into the weights: refusal, tone and preferences are learned from instructions and human feedback, and the product is the weights. Memory, when the product has one, is the context window or a store retrieved by similarity and read on trust.

quantum-cortex has a post training stage of a different kind. Each run of the v series is one fine tune of the pretrained control, teaching a contract, cite or abstain, and the use of the organ; there is no instruction stage and no preference stage. What the common pipeline bakes into the weights is carried outside them, by an organ: an episodic memory sealed per domain, persistent across sessions, read under a policy in which the model decides and the organ answers. Refusal is the organ's veto on a citation whose line it does not mark above a threshold frozen before the run; it is a policy, not a weight, and it is measured.

Two steps appear here that popular accounts omit. Before data, provenance: the training slice is hashed and recorded. After every run, measurement: the run is read cold against thresholds declared before it, with negative controls, and graved as a row of the record.

The consequence is the thesis of this paper. Weights are replaced by versions; what must persist cannot live in them. The organ is where it lives.

---

## Short form for the README

Same first five steps as any language model, at a much smaller scale. Post training here is one fine tune that teaches a contract, not a stage that bakes behaviour into the weights: that behaviour lives in an organ outside them, measured against frozen thresholds. Two steps others leave out: provenance before data, measurement after every run.
