# DNA Replication

Every time a cell divides, it must copy its entire genome so each daughter cell gets a complete set of instructions. This process — DNA replication — is remarkably accurate, with an error rate of roughly 1 mistake per billion base pairs copied.

## 1. The Core Principle: Semi-Conservative Replication

Because DNA is double-stranded and the two strands are complementary, each strand can serve as a template for building a new partner strand. After replication, each new DNA molecule contains one original strand and one newly synthesized strand.

```
Original:   5'—ATCGGCTA—3'
            3'—TAGCCGAT—5'

After replication:

Molecule 1:  5'—ATCGGCTA—3'   (original strand)
             3'—TAGCCGAT—5'   (new strand)

Molecule 2:  5'—ATCGGCTA—3'   (new strand)
             3'—TAGCCGAT—5'   (original strand)
```

This is called **semi-conservative** replication — each copy retains half of the original.

## 2. Key Players

| Molecule | Role |
|----------|------|
| **Helicase** | Unwinds and separates the two strands |
| **Primase** | Synthesizes a short RNA primer to start synthesis |
| **DNA Polymerase** | Reads the template and adds new nucleotides |
| **Ligase** | Joins DNA fragments together |
| **Topoisomerase** | Relieves tension ahead of the replication fork |

**DNA polymerase** is the central enzyme. It can only add nucleotides to the 3' end of an existing strand — it cannot start from scratch. That's why primase first lays down a short RNA primer.

## 3. The Replication Fork

Replication starts at specific sites called **origins of replication**. In bacteria there's one origin; in humans there are thousands (replication would take too long otherwise).

At each origin, the two strands are separated, creating a **replication fork** that moves in both directions:

```
         ← fork moves left    fork moves right →

    3'←—————————————[origin]—————————————→5'
    5'→—————————————         —————————————→3'
```

### Leading and Lagging Strands

DNA polymerase can only synthesize in the 5'→3' direction. This creates an asymmetry:

- **Leading strand**: synthesized continuously in the direction the fork moves
- **Lagging strand**: synthesized in short fragments (called **Okazaki fragments**) in the opposite direction, then joined by ligase

```
Direction of fork →

Leading:   ——————————————————→  (continuous)

Lagging:   ←——  ←——  ←——  ←——  (Okazaki fragments, later joined)
```

## 4. Fidelity and Error Correction

DNA polymerase has a built-in **proofreading** function — it checks each newly added nucleotide and removes incorrect ones. This brings the error rate down to ~1 in 10⁷ during synthesis. Additional **mismatch repair** systems after replication push it further to ~1 in 10⁹.

Despite this, errors do occur. When they aren't repaired, they become **mutations** — permanent changes to the DNA sequence. Mutations are the raw material of evolution and the cause of many diseases, including cancer.

## 5. Why This Matters for Bioinformatics

### Sequencing Reads Come from Replication-Like Chemistry

Modern DNA sequencing (e.g., Illumina) works by performing a controlled, observable version of replication. Understanding replication helps you understand why:
- Reads have a directionality (5'→3')
- Both strands are sequenced (you get reads from both the forward and reverse strand)
- Errors in sequencing cluster at certain positions (analogous to polymerase error patterns)

### Copy Number and Coverage

When you sequence a genome, regions that were replicated more times (e.g., due to copy number variants) show up with higher read coverage. Detecting these **copy number variations (CNVs)** is a key bioinformatics task.

### Replication Timing

Different regions of the genome replicate at different times during the cell cycle. Early-replicating regions tend to be gene-rich and have different mutation patterns than late-replicating regions. This is a useful feature in ML models predicting mutation rates.

## 6. Telomeres and the End-Replication Problem

DNA polymerase can't replicate the very ends of linear chromosomes (it needs a primer upstream). This means chromosomes shorten slightly with each cell division. **Telomeres** — repetitive sequences (`TTAGGG` repeated thousands of times) at chromosome ends — act as a buffer, absorbing this shortening.

Telomere length is associated with aging and cancer. In cancer cells, an enzyme called **telomerase** re-extends telomeres, allowing unlimited division.

## 7. Conclusion

Key things to remember:

- Replication is **semi-conservative** — each new molecule has one old and one new strand
- **DNA polymerase** reads 3'→5' and synthesizes 5'→3', requiring a primer to start
- The **leading strand** is synthesized continuously; the **lagging strand** in fragments
- Error rate is ~1 per 10⁹ bases thanks to proofreading and mismatch repair
- Sequencing reads reflect the directionality and chemistry of replication

With DNA structure, the central dogma, and replication covered, you have the biological foundation needed to understand genomic data formats, variant calling, and sequence-based ML models.
