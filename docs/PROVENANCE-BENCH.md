# The provenance benchmark

A way to measure a memory system of your own on this record's questions, with this record's reading, without the model and without the organ. It builds a bench from the generator the record sealed, you make your system read the statements and answer the questions, and it scores your answers with the record's reading of rows 48 to 50 of [`RESULTS.md`](RESULTS.md), reimplemented here and pinned by tests against the record's functions, and with the record's frozen conditions as its protocol scores them. It is a measuring tool, not a leaderboard: your numbers are yours, on your system, and the record claims nothing from them. Declared before its code in [`working/PROVENANCE-BENCH-DECLARATION-2026-11-05.md`](working/PROVENANCE-BENCH-DECLARATION-2026-11-05.md) and amended there after its committee; the module is `cortex_c2b/provenance_bench.py`, its tests `tests/test_c2b_provenance_bench.py`.

## What it measures

The contract the record holds its own system to: an answer is a value with a citation to a stored line, by that line's pointer, or an abstention. The bench counts, per block of questions and pooled over the planted ones, every rate as a numerator over a denominator:

- **coverage**: a value given, over the questions;
- **value correctness**: the value equal byte for byte to the key's, over the questions (the value alone; the record's `value_correct` is its strict share, which the bench's **strict** below equals);
- **valid citations** and **strict answers**, the record's frozen reading (`cortex_c2b.hm_lm`): a citation is valid when the cited line is held and carries the value the system answered, whatever the line is about; an answer is strict when its citation is valid and its value the key's; both over the questions;
- **invalid citations**: a value given with no pointer, with a pointer the bench does not hold, or with a held line that does not carry the answered value, over the questions (the record's definition);
- **citation support**, in the record's two readings, both over the answered questions: `citation_supported`, ADR-007 section 4 (b)'s as the record's `citation_support_of` reads it, the cited line is about the question's entity and carries the key's value; `citation_supported_answered`, section 7's wording, the cited line is about the entity and carries the value the system answered (a wrong value the cited line carries is supported under this reading, and value correctness says it is wrong). The record's files count the support on every citation, an abstention's surviving citation included, which the bench does not see;
- **the designated line**: the cited pointer is the first statement written about the entity under the question's schema, over the answered;
- **attributed accuracy**: a correct value with a citation supported under 4 (b), over the questions;
- **abstentions**, split into **false** (abstained where a line carrying the key's value exists among the bench's statements) and **verified** (abstained where none exists);
- on the **never planted**, **near name** and **cross schema** blocks: **claims**, a value given where no statement supports one;
- on the **contested** blocks (a fact followed by an E1 partner, the same entity with another value; an E2 partner, a name one byte away with the same value; an E4 paraphrase, the same fact under another template; or a later fact of the draw on the same entity and schema, an E1 by chance): **cites the partner** with its exact 95 % interval (Clopper and Pearson, ADR-010's; row 50's intervals are Wilson's), **cites the designated line**, **abstains**, value correctness. A contested block is by construction, every question whose fact has a partner; the record's contested reads are by the window the organ returned, which the bench does not see;
- **the frozen conditions** (declared 2026-09-12, not moved), as the record's protocol scores them: the strict recall at or above 0.50 over the planted questions (the off arm being zero by construction, the gap is this recall); invalid citations at or under 0.01 of the planted questions; claims on never planted names under 0.10; the fourth, the organ cut at the floor by construction, is not the bench's to measure and is written as not measured. `holds` is the conjunction of the three measured. Beside them, as the record keeps the own line gap beside the frozen one: the **attributed accuracy gap** (attributed accuracy at or above 0.50) and `holds_with_attributed_accuracy_gap`.

The record's own files split the abstentions by the window the organ returned, which the bench does not see; everything else is the record's reading, pinned by tests against the record's functions (`test_the_scorer_equals_the_records_reading_on_a_records_file`, on three committed files of row 49, the differences stated in the test).

## How to run it

```
pip install quantum-cortex
python -m cortex_c2b.provenance_bench build --seed 4242 --n 1000 --e4 --out bench-4242.json
```

The module is in the package from v1.1.0 (the 1.0.0 package on the index does not carry it). The bench file holds `statements` (in the order they are to be written; each with `pointer`, the organ's content hash of the text's UTF-8 bytes, sha256 in sixteen hexadecimal characters, so that a system storing the texts as given can cite by the same pointer the organ would), `questions` (each with its `block`) and `key` (per question the entity, the schema, the value, the designated line, the supporting lines, the partner's line on a contested block). `n` is a multiple of five, at least 20. `--families` adds the three sealed families of rows 51 and 52 (T0, A, Av) exactly as sealed, their content hashes checked against `metrics/mqar/SEALED-FAMILIES.json` before anything is built: it runs from a clone of the repository with the module imported from that clone (the package alone does not hold the seal). `--split` writes the public part without the key under `--out` and the key under `<out>.key.json`, for a blind run (the public part keeps the build's order and the counts per block, so a question's block is inferable from its position: a blind run is blind to the key, not to the blocks). A build never overwrites a file.

The sizes: with `--e4` every direct question is contested (its paraphrase is a partner), and the `planted` block is empty; without it, one fact in twenty has an E1 partner and one in twenty an E2 partner, so `--n 1000` gives 50 contested E1 and 50 contested E2 questions, where the record's arms of rows 48 to 50 partner every fact.

Make your system read the statements in order and answer every question, and write:

```json
{"system": "my memory system v3",
 "bench_content_sha256": "<the bench file's construction.content_sha256, optional>",
 "answers": {"q000000": {"value": "cartographer", "cited": "0123456789abcdef"},
             "q000001": {"value": null, "cited": null}}}
```

A value is a string or null; a pointer is sixteen hexadecimal characters or null; anything else is refused with the offending question id, and so is a question id that appears twice. A question absent from the file, or with a null value, is an abstention; a null value with a pointer is an abstention with a citation (counted apart); a value with a null pointer is an invalid citation. Then:

```
python -m cortex_c2b.provenance_bench score --bench bench-4242.json --answers answers.json --out table-4242.json
```

(with `--key bench-4242.json.key.json` after a split build). The scorer recomputes the bench's content hash from its statements and questions and checks every pointer against its text, and refuses a bench edited after its build, a key of another bench, or an answers file that names another bench. The table is written, never overwritten, and its summary printed: the counts per block, the pooled planted metrics, the conditions, the bench's content hash, the answers file's sha256 (of its bytes), the module's hash.

## The construction

The scale draw of ADR-010 part C at the seed you give (`cortex_c2b.sealed_families.scale_plan`): `n` facts of the protocol's generator in a permuted written order, five schemas, an E1 partner after each fact at a written position `i` with `i mod 20 = 0` and an E2 partner after each fact with `i mod 20 = 10` (a bumped name already taken is dropped and counted), one thousand never planted names with their distance to the nearest planted name; `--e4` adds a paraphrase after every fact (row 49's templates). A later fact of the draw on an entity and schema already written (the seal's hundred thousand draw holds two, ADR-010 C.1 (b)) is written as a statement and read as an E1 partner by chance of the first write: the key's value is the first's, no second question is asked, and the construction lists the repeated keys. No encoder runs and no journal is opened: the key is read from the plan, never parsed from text. The sealed families, when asked, as the seal fixed them: 200 statements each, the direct and the paraphrase questions, the never planted entities (fifty for T0 and Av, two hundred for A), the fifty cross schema questions (a planted entity asked under another schema, which no statement answers) and the fifty near names (a planted name moved by one hexadecimal character, never planted).

## What it is not

Not a benchmark of the organ: the organ's numbers are the record's rows, read from committed files, and the bench does not run it. Not a public long memory benchmark: the facts are synthetic (the generator's alphabet, five schemas, the record's templates), so the bench measures provenance and abstention under those facts and not answer accuracy on natural text; the roadmap places a public benchmark in v1.2, to be declared there with its reading written before. Not a leaderboard. It names no product and ranks nothing.
