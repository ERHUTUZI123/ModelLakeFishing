# X3 pre-registration: the second (node-level) protocol and the S2 sibling prior

Written 2026-08-26, before any protocol-B number was computed. The audit
(`GOLD10_FAST_LEVERS_AUDIT.md` §4.2) requires this document to exist first,
because the change under test is a change of measurement convention rather than
a change of method, and a convention chosen after seeing the numbers is not
evidence. Everything listed in section 4 will be reported in `X3.md` whatever
the values turn out to be.

## 1. Why a second protocol is needed

The full-lake evaluation reports one protocol, root-aware splitting. Its split
unit is the root dataset: every `trained_on` edge whose dataset belongs to a
root lands on exactly one side. This is the correct protocol for the hardest
serving scenario, a query dataset from a benchmark family the system has never
seen.

It also makes the S2 sibling prior inert. If D is held out, every other
configuration of D's root is held out with it, so D has no train-visible
sibling. X1 §9 measured this over all queries of all three seeds and found zero
queries with a visible sibling; X2 turned that into an assertion in the P axis.

The scenario the root-aware protocol excludes is common. 11,313 of the 18,729
dataset nodes (60.4 per cent) sit in a root that holds at least two nodes, and
1,029 / 778 / 1,239 of the queries of seeds 0 / 1 / 2 (66.0 / 67.5 / 77.7 per
cent) are in such a root. For those queries the realistic serving situation is
"a new configuration of a benchmark family the system already knows", and the
root-aware protocol assigns that situation zero prior information by
construction.

The plan is therefore to keep root-aware as the primary protocol and to report a
second protocol alongside it, stating which serving situation each one
represents.

## 2. What the second protocol changes, and what it deliberately does not

The audit specifies that this step changes neither the training nor the index.
That constraint decides the design.

A complete node-level re-run would assign dataset nodes, rather than roots, to
train, validation and test, and would retrain on the resulting message graph.
That is not what is done here, for a reason worth stating precisely. The frozen
embeddings were trained under root-aware splitting. A node-level query set would
contain datasets whose roots were on the training side of that split, so the
trained model would have seen those datasets' own labels. Evaluating the frozen
embeddings on a node-level query set would therefore leak.

The second protocol is consequently defined at serving time only:

* The query set stays the root-aware test query set. The model never saw these
  datasets' labels under either protocol.
* The embeddings stay the frozen `z_*_eval` matrices of F7.
* The index is not rebuilt and is not used; all ranks are computed over the full
  3,016,439-model pool.
* Only the edge set that the serving prior is allowed to read changes.

Two consequences follow and will be stated in the report. First, the retrieval
backbone never saw the sibling edges, so the embedding half of protocol B is
weaker than a node-level-trained system's would be, and the protocol-B numbers
are a lower bound in that respect. Second, protocol B's prior is given access to
labels that a genuine node-level split would have placed in its own validation
and test folds, so the prior half is an upper bound. The two bounds run in
opposite directions and neither is claimed to be the value a full node-level
re-run would produce.

## 3. Definitions

Let D be a query dataset, R(D) its root, and T(D) its normalised task string.

Protocol A, root-aware, already reported in X2. The prior may read the train and
validation edges of the root-aware split of the given seed, written V_A. The
sibling group of D is empty by construction.

Protocol B, node-level at serving time. The prior may read every edge except
those incident to D itself. The sibling group of D is the set of datasets in
R(D) other than D; the task group of D is the set of datasets with task T(D)
other than D.

Three prior sources will be used, so that the sibling channel can be separated
from the widened task channel:

| source | contents |
|---|---|
| V_A | train and validation edges of the root-aware split, per seed |
| sibling(B) | edges of the datasets in R(D) other than D |
| task(B) | edges of the datasets with task T(D) other than D |

The sibling prior and the task prior follow `stage3HNSW/serving_rerank.py`
without modification: the sibling boost is the plain mean of normalised accuracy
over the group, the task boost is the shrunk mean with k = 5, and the fused
score is

    fused(D, m) = minmax(z_d[D] · z_m)[m] + alpha · sibling_boost(m) + beta · task_boost(m)

with alpha = beta = 1, the weights adopted in v6, and the min-max taken over the
whole lake as the reference implementation does.

## 4. What will be computed and reported

Seven rankings, on all three split seeds, over the full candidate pool:

| # | name | sibling source | task source |
|---|---|---|---|
| 1 | MIPS | none | none |
| 2 | A: task | none | V_A |
| 3 | B: sibling | sibling(B) | none |
| 4 | B: sibling, name-distinct only | sibling(B) restricted, see below | none |
| 5 | B: sibling + task(A) | sibling(B) | V_A |
| 6 | B: task | none | task(B) |
| 7 | B: sibling + task(B) | sibling(B) | task(B) |

Row 4 restricts the sibling group to datasets whose name, taken before the tab
separator, has character 3-gram Jaccard similarity of 0.5 or less with D's name.
Siblings share a root and are therefore often similar by name, so this row
measures how much of the sibling channel survives when the most textually
similar configurations are removed. It is reported as a sensitivity row, not as
a leakage correction; sibling configurations of one benchmark are the intended
subject of protocol B, not an artefact.

Rows 1, 2, 3, 5 and 7 will be reported on three strata, following the v6 D0
table: all queries; queries whose root contains at least one other dataset node
carrying at least one `trained_on` edge in the full graph ("has sibling"); and
the complement ("no sibling"). The size of each stratum will be reported.

Metrics per ranking and stratum: `gold@1`, `gold@10`, `gold@50`, `gold@100`,
`top3@10`, `gold-gap@10` and the median gold rank, all computed by the same
counting used in the P axis, with the probe's own diagonal masked.

The second layer of F9 will be recomputed on the top-10 lists of rows 1, 5 and 7
and reported in the same table as the first layer, as the audit §6 requires for
any channel that only scores models with records.

The primary comparison is row 3 against row 1 on the "has sibling" stratum,
which is where the sibling prior can act at all.

## 5. Expectation recorded in advance

On D0 with eight seeds, v6 measured the has-sibling stratum at 0.158 for plain
MIPS, 0.259 with the sibling prior alone, 0.239 with the task prior alone and
0.261 with both. The expectation for the full lake is that the sibling prior
raises `gold@10` on the has-sibling stratum relative to plain MIPS by a factor
between one and two, and that it is comparable to or weaker than the task prior
rather than clearly stronger, since X2 measured the task channel at 5.0 times
plain MIPS on this rung against v6's 1.5 times on D0.

There is no threshold that would cause a row to be withheld. If the sibling
prior turns out to be inert, harmful, or stronger than expected, the measured
values are reported as they are.

## 6. Validation planned

The MIPS row will be recomputed inside the same run and is expected to reproduce
F8's `gold@10` of 0.0700 / 0.0399 / 0.0696 exactly; if it does not, nothing else
in the run is usable.

Row 2 is expected to reproduce X2's `task_b1` row exactly, since it is the same
computation with the same sidecar.

For every query and every prior source, the group used to build the boost will
be asserted not to contain D itself.

The rank counted by the streaming pass and the membership of the independently
maintained top-10 list will be compared per query, as in X2.

## 7. Artifacts

The pre-registration file is this document. Its SHA-256 is recorded in the
result file `X3_PROTOCOL_B.json` under `prereg_sha256`, so that the ordering of
pre-registration and measurement is checkable after the fact.
