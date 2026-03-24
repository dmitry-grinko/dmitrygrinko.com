# DNA and the Double Helix

DNA (deoxyribonucleic acid) is the molecule that stores the genetic instructions for building and running every living organism. Understanding its structure is the foundation for everything in genomics and computational biology.

## 1. What DNA Is Made Of

DNA is a polymer — a long chain of repeating units called **nucleotides**. Each nucleotide has three parts:

- **Phosphate group** — forms the backbone
- **Deoxyribose sugar** — connects phosphate to base
- **Nitrogenous base** — carries the actual information

There are four bases, usually written as single letters:

| Base | Letter | Type |
|------|--------|------|
| Adenine | A | Purine |
| Guanine | G | Purine |
| Cytosine | C | Pyrimidine |
| Thymine | T | Pyrimidine |

The sequence of these bases — e.g. `ATCGGCTA...` — is what encodes genetic information. In computational biology, DNA is almost always represented as a string over the alphabet `{A, T, C, G}`.

## 2. The Double Helix

DNA doesn't exist as a single strand. Two strands wind around each other to form the **double helix**, discovered by Watson and Crick in 1953.

The two strands are held together by **hydrogen bonds** between complementary bases:

```
A — T   (2 hydrogen bonds)
G — C   (3 hydrogen bonds)
```

This is called **base pairing** or **Watson-Crick pairing**. It's strict: A only pairs with T, and G only pairs with C.

```
5'— A T C G G C T A —3'   (strand 1)
    | | | | | | | |
3'— T A G C C G A T —5'   (strand 2, complement)
```

The two strands run in **opposite directions** — they are **antiparallel**. One runs 5'→3', the other 3'→5'. The 5' and 3' refer to carbon positions on the sugar, and this directionality matters enormously for replication and transcription.

## 3. Why Complementarity Matters for Computing

Base pairing is the reason DNA can be copied faithfully. Given one strand, you can always reconstruct the other:

```python
def complement(seq: str) -> str:
    """Return the complement of a DNA sequence (same direction)."""
    table = str.maketrans('ATCG', 'TAGC')
    return seq.translate(table)

def reverse_complement(seq: str) -> str:
    """Return the reverse complement (antiparallel strand)."""
    return complement(seq)[::-1]

print(complement('ATCGGCTA'))        # TAGCCGAT
print(reverse_complement('ATCGGCTA')) # TAGCCGAT reversed → ATCGGCTA... wait:
# ATCGGCTA → complement → TAGCCGAT → reverse → TAGCCGAT
```

The **reverse complement** is used constantly in bioinformatics — when you sequence DNA you may get either strand, so tools always check both a sequence and its reverse complement.

## 4. DNA Is Compactly Packaged

The human genome contains ~3 billion base pairs. Stretched out, that's about 2 meters of DNA per cell. It fits inside a nucleus ~6 micrometers wide by wrapping around proteins called **histones**, forming structures called **nucleosomes**, which coil further into **chromatin**.

This packaging isn't just structural — it regulates which genes are accessible and can be read. This is the domain of **epigenetics**, which is increasingly important in ML models of gene regulation.

## 5. The Genome

The complete DNA of an organism is its **genome**. Key numbers for the human genome:

| Property | Value |
|----------|-------|
| Total base pairs | ~3.2 billion |
| Chromosomes | 23 pairs (46 total) |
| Protein-coding genes | ~20,000 |
| Coding fraction | ~1.5% |
| Repetitive elements | ~50% |

The vast majority of the genome is non-coding. Much of it was once called "junk DNA," but large portions have regulatory roles — controlling when and where genes are turned on. This non-coding regulatory DNA is a major target for ML models.

## 6. Conclusion

Key things to remember:

- DNA is a sequence over `{A, T, C, G}` — this is what makes it directly amenable to sequence models
- The two strands are complementary and antiparallel — always consider the reverse complement
- A pairs with T (2 bonds), G pairs with C (3 bonds) — G-C pairs are stronger
- Most of the genome is non-coding, but not non-functional

Next: **The Central Dogma** — how the information in DNA gets read and turned into proteins.
