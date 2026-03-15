# Decisions Log

This document records architectural and implementation decisions made while improving the OfficeQA agent benchmark score.

## Baseline

- Starting score: **59 / 246** (23.98% overall, 30.09% easy, 18.80% hard)
- Route mix: 9 deterministic, 237 model

## Decision 1: Fix list number extraction (formatting.py)

**Problem:** The `NUMBER_TOKEN_PATTERN` regex greedily matches commas inside numbers. When the model returns a bracket-enclosed list without spaces between items (e.g. `[10102000000,4.73]`), the regex fuses them into a single token `10102000000,4.73`, which normalizes to `101020000004.73`.

**Solution:** Added `_extract_list_numeric_tokens()` — a list-aware extraction function that splits on commas first (preferring `, ` to preserve thousand-separator commas within numbers), then extracts one number per part. Used only in the list branch of `canonicalize_final_answer`.

**Impact:** Fixed UID0017 — prediction changed from `[101020000004.73]` to `[10102000000, 4.73]`, matching ground truth exactly. **+1 correct.**

## Decision 2: Expand list detection (formatting.py)

**Problem:** `_expects_list()` only matched narrow bracket-related keywords ("square brackets", "enclosed brackets", etc.). Questions asking for a "comma separated list" or "comma-separated list" were missed, causing list answers to be collapsed to scalars.

**Solution:** Added patterns: `"comma separated list"`, `"comma-separated list"`, `"as a list ["`, `"as a list starting"`, `"return as ["`. Also added auto-detection: if the model returns bracket-enclosed content with commas (regex `\[.+,.+\]`), treat it as a list regardless of question keywords.

**Impact:** Fixed UID0232 — prediction changed from scalar `1444` to `[1444, 3174]`, matching ground truth. **+1 correct.**

## Decision 3: Do NOT add % sign in post-processing

**Problem:** 10 failures where the judge expects a `%` sign but the model returns a bare number (e.g. pred=`9.89`, gt=`9.89%`). Conversely, some correct answers have questions with "percentage" keywords but ground truth without `%`.

**Analysis:** Adding `%` based on question keywords would fix 3 answers (UID0036, UID0046, UID0059) but break 2-3 currently correct answers (UID0090: gt=`56.24` no %, UID0123: gt=`[34.4, 0.391]` no %). The dataset is inconsistent — "percentage points", "what percentage of", and "percentage value" sometimes map to `%` in ground truth and sometimes don't.

**Decision:** Leave `_normalize_numeric_token` unchanged (only preserve existing `%`, never add it). The `%` sign decision is better left to the model via prompt guidance in future iterations.

## Decision 4: Revert prompt changes to preserve model determinism

**Problem:** Modifying the system prompt or user prompt invalidates the LLM response cache, causing all 246 questions to get fresh API calls. Even with temperature=0, this introduced variability — some previously correct answers changed to wrong values.

**Tested approaches:**
1. "Use ONLY retrieved context" — caused 17 N/A refusals, score dropped to 55/246.
2. Added % formatting guidance — caused UID0123 to add unwanted `%`, score was 58/246.
3. Reverted to original prompts — restored determinism, score rose to 61/246 with formatting fixes.

**Decision:** Keep prompts exactly as they were in the baseline. Formatting improvements go in `formatting.py` post-processing only, which doesn't affect the LLM cache.

## Decision 5: Revert retrieval scoring changes

**Problem:** Modified retrieval scoring (boosted table chunks, added number-density and month-name bonuses, added domain keywords) changed which context chunks were provided to the model. This shifted some answers for worse.

**Decision:** Reverted to original retrieval scoring. Retrieval changes are high-risk because they affect every question's context window. Future retrieval improvements should be tested more carefully.

## Decision 6: Remove dead code

**Problem:** Two unused functions found in the codebase.

**Removed:**
- `_infer_unit_scale()` in `table_parser.py` — private function with zero callers
- `extract_numbers()` in `normalize.py` — public function with zero callers (similar functionality exists in `parse_number()` which is used)

## Result

- Final score: **61 / 246** (24.80% overall, 30.97% easy, 19.55% hard)
- Net gain: **+2 correct** with **0 regressions**
- Gains: UID0017 (list parsing fix), UID0232 (list detection fix)

## Future Improvement Areas

1. **Retrieval quality** — better ranking of table-heavy chunks, page-hint boosting, year/month matching (needs careful A/B testing)
2. **Prompt tuning** — % sign guidance, list formatting instructions, precision instructions (must avoid cache invalidation regressions)
3. **Output normalization** — more aggressive prose stripping, better scalar extraction
4. **Tolerance** — current judge uses 0% tolerance; 26 answers are within 2% of correct
