# Name stimuli

## wilson_caliskan_names.csv

The 120 first names of Wilson & Caliskan, "Gender, Race, and Intersectional Bias in
Resume Screening via Language Model Retrieval" (AIES 2024), Appendix Tables 4–6,
reproduced as printed with the paper's frequency columns. First names come from the
name database of Elder & Hayes (2023); the surname is Williams for every name.

- `set = primary`: Table 4 (20 BF, 20 BM) and Table 5 (20 WF, 20 WM). The White names
  of Table 5 are about 5.5 times as frequent in the DOLMA corpus as the Black names.
- `set = supplemental_freq_matched`: Table 6 (20 WF, 20 WM), matched in corpus frequency
  to the Black names.
- `Courtney` appears in both Table 5 and Table 6, so the file has 119 distinct first names.
- `Elmer` is printed under WF in Table 6.

The thesis uses the frequency-matched design (`experiments/names.py`, design
`frequency_matched`): the 40 Black names of Table 4 and the 40 White names of Table 6,
with `Elmer` dropped, giving 79 names (20 BF, 20 BM, 19 WF, 20 WM). `keep_elmer=True`
retains it.

## lexical_probe_names.csv

Three first names chosen for this thesis because each is also a programming language:
Pascal, Ada and Ruby, with the surname Williams. They are grouped by gender only and
carry no race label. They probe whether a name that is also an O*NET surface form
changes the method's score.

## arbitrary_ids.csv, stimulus_token_lengths.csv

Twenty random two-block identifiers matched to the names' token-length distribution
under the Qwen3 tokenizer (`experiments/make_controls.py`), and the token lengths of
all stimuli under the three encoders.
