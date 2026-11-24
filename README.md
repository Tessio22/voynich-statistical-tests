# Voynich manuscript (Beinecke MS 408) — statistical tests with controls

**This is not a decipherment.** It is a set of reproducible statistical experiments on the Voynich
manuscript text. They rule out a range of simple cipher hypotheses across 15 languages and variants,
and describe two structural properties of the text. Every decipherment test was first checked on a
*positive control*: real text that we encrypted ourselves the same way. It was also compared against a
*mechanical generator* that produces Voynich-like text with no content.

> **AI disclosure.** The experiments, code and write-up were produced by Claude (Anthropic) at the
> request of, and under the direction of, the repository owner (November 2026). Numbers come from running the
> code in this repository; nothing was typed in by hand. Please treat it as a lab notebook to check,
> not as an authority. Corrections are welcome via issues.

## Main findings

1. **A line behaves like a verse.** The end of a word strongly predicts the start of the next word
   *within* a line, but this dependency vanishes *across* line breaks. Latin and Italian verse show the
   same pattern; prose does not.

   | Text | Within line | Across line break |
   |---|---|---|
   | Voynich (EVA, paragraph text) | 0.17–0.21 | 0.00–0.01 |
   | Latin verse (Macer Floridus) | 0.26 | 0.01 |
   | Dante, Inferno (verse lines) | 0.16 | 0.02 |
   | Dante, same text re-wrapped every 9 words | 0.135 | 0.140 |
   | Latin / Italian prose | 0.05–0.06 | 0.05–0.06 |

   (Excess mutual information in bits between word-final and next word-initial glyph, minus a shuffle
   baseline.) The result holds when uncertain spaces are joined, and when line-initial gallows and
   line-final `-m` are excluded (0.154 vs 0.001). Lines also have a positional template: `sh-` initial
   words fall from 18 % at position 2 to 3.5 % at the line end, while `o-`, `d-`, `a-` and `l-` rise. A
   smooth page-level drift exists, but there is no cyclic "key per line" and no rhyme.

2. **A layer of interchangeable variants hides word-level structure.** Glyph pairs that occur in the
   same word contexts were found from the data: the endings `iin`/`in`/`ir`/`r`/`l`, plus `k`/`t`,
   `ch`/`sh`, `ch`/`cth`/`ckh` and `d`/`s`. After merging them, word-to-word dependency rises from 0.039
   to 0.083, which is the level of Latin prose (0.084). The mechanical generator does not show this rise
   (0.005–0.015 → 0.008–0.013).

3. **A mechanical generator reproduces 11 of 12 measured statistics.** The generator uses a glyph
   chain, a word-junction rule, a position-in-line effect, line-to-line drift and 30 % self-copying.
   It falls short only on word-level dependency, where it scores 0.004–0.014 against 0.034 for
   Voynich and 0.08–0.22 for real languages.

## What was ruled out (each with a passing positive control)

| Hypothesis | Languages | Control solved | Result |
|---|---|---|---|
| glyph = letter, real spaces, homophones allowed | Latin, Italian, German | 92–100 % | rejected |
| letter = group of 1–3 glyphs | Latin, Italian, German | 92–100 % | rejected |
| glyph = letter (clustering + substitution solver) | Latin, Italian, Czech, Catalan, Occitan, Hebrew, Arabic, Persian, Nahuatl, Latin without vowels, abbreviated Latin | 79–99 % | Voynich reaches only 28–61 % of the way from shuffled text to the real language, whatever the language |
| word = letter (Naibbe-type cipher) | Latin, Italian, German, Czech, Catalan, Occitan | 44–85 % | no better than shuffled Voynich |
| anagrammed words (Hauer & Kondrak) | 13 languages incl. Hebrew, Arabic, Persian | 87–100 % | indistinguishable from the generator |
| reduced syllabary (Linear B-like) | Italian, Latin | 93–97.5 % | rejected |
| trimmed prefixes / suffixes are padding | Latin, Italian | 82–99 % | trimming moves the text *away* from the language |
| frequent words are nulls | — | 5/5 fake nulls found | no null words found |
| word = word (EM / HMM, Knight-style) | Latin (600k-word LM), Arabic | 69 % / 54 % of words | sequence gain 0.10, below every real language (0.14–0.29) |
| plant names spelled letter by letter | 17 identified herbal pages | 0.99 vs 0.70 | p = 0.29 |
| zodiac labels as day names/numbers | 10 signs, 299 labels | — | p = 0.35 |
| Hildegard's Lingua Ignota-type invented vocabulary | — | — | Voynich words have 7.4 one-edit neighbours (LI 0.5, languages 1–2.4, generator 8.4) |

Inconclusive: Turkish (glyph solver control failed), Italian without vowels (control 50 %).

## Reproduce

```bash
python3 -m pip install numpy
python3 fetch_data.py                 # downloads every corpus into ./data (15–30 min)
cd src
python3 exp1_baseline.py              # entropy + line-boundary dependency vs languages
python3 exp10_generators.py           # mechanical generator vs Voynich
python3 exp19b.py                     # variant merging
LANGS=latin.txt python3 exp8_cluster_solve.py glyph 25 10   # solver with control (~15 min)
python3 exp30_em.py latin 400 400 30  # EM word-level decipherment with controls (~15 min)
```

Logs of the runs we made are in `results/`. The scripts expect the data in `../data`, or set
`VOYNICH_DATA=/path/to/data`.

### Experiments

| Script | What it tests |
|---|---|
| `exp1_baseline.py` | entropy, word length, within/across-line dependency vs languages |
| `exp2_bpe.py`, `exp23_syll.py` | re-segmentation of glyphs into larger units; syllable profile |
| `exp4_unit_letter.py`, `exp4b_nospace.py` | glyph/unit = letter, with real or ignored spaces |
| `exp8_cluster_solve.py` | clustering + substitution solver, `word` / `glyph` modes (`LANGS`, `TRIM` env) |
| `exp9_linekey.py`, `exp9b_drift.py`, `exp9c.py` | per-line key, drift, periodicity, position template |
| `exp10_generators.py`, `exp10_grid.py` | mechanical generator and its parameter grid |
| `exp11_wordpairs.py`, `exp29_wordnet.py` | word-level dependency beyond junction glyphs |
| `exp14_keywords.py`, `exp14c.py` | keyword information (Montemurro & Zanette 2013) vs generator |
| `exp15_neighbors.py`, `exp16_collapse.py`, `exp19_subst.py`, `exp19b.py` | interchangeable variants |
| `exp6_selfcite.py` | self-citation locality (Timm & Schinner) |
| `exp13_plants.py`, `exp18_macer.py`, `exp20_three.py`, `exp17_pharma.py` | plant-name cribs, Macer Floridus properties, pharma labels |
| `exp22_anagram.py`, `exp22b.py` | anagram hypothesis (Hauer & Kondrak 2016) |
| `exp24_syllsolve.py` | reduced syllabary |
| `exp21_wordsub.py`, `exp27_wordsub2.py`, `exp30_em.py` | word-level substitution (hill-climbing, EM) |
| `exp26_zodiac.py`, `exp25_order.py` | zodiac labels; page-order similarity |
| `exp31_nulls.py`, `trim.py` | null words; trimmed prefixes/suffixes |
| `exp32_lingua.py` | comparison with Lingua Ignota |

## Data sources and credits

- Voynich text: ZL transliteration v3b (May 2025) by René Zandbergen, [voynich.nu](https://www.voynich.nu/transcr.html). Not redistributed here.
- Scans: [Beinecke Rare Book & Manuscript Library, Yale, MS 408](https://collections.library.yale.edu/catalog/2002046).
- Comparison texts: Latin, Italian, German, Czech, Catalan, Occitan, Hebrew, Arabic, Persian and Turkish Wikisource, and the Nahuatl Wikipedia (CC BY-SA). All are fetched by `fetch_data.py`, not stored here.
- Lingua Ignota word list: [Philip Neal, Voynich Sources](http://philipneal.net/voynichsources/lingua_ignota_hildegard/).

## Limitations

- All results depend on the EVA-based ZL transliteration and its word spacing.
- The comparison corpora are modest: 10k–650k words per language.
- The word-level solvers are far from perfect even on controls (EM: 69 % of words on Latin).
- "Ruled out" means "not supported under the tested model with a working control". It does not prove
  that no other encoding could work.

## License

Code and texts in this repository: MIT (see `LICENSE`). Downloaded data keeps its original license.
