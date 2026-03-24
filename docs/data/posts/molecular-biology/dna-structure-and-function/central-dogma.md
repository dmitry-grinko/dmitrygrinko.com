# The Central Dogma

The central dogma of molecular biology describes the flow of genetic information inside a cell:

```
DNA  →  RNA  →  Protein
```

This is the core process by which the instructions stored in DNA are used to build the molecules that actually do work in the cell. Understanding it is essential for making sense of genomic data.

## 1. The Three Molecules

**DNA** stores information permanently. It's the master copy — rarely read directly, mostly kept safe.

**RNA** (ribonucleic acid) is a working copy. It's similar to DNA but single-stranded and uses **uracil (U)** instead of thymine (T). RNA is temporary and disposable.

**Protein** is the functional output. Proteins are chains of amino acids that fold into 3D structures and carry out virtually all cellular functions — enzymes, structural components, signaling molecules, etc.

## 2. Transcription: DNA → RNA

**Transcription** is the process of copying a segment of DNA into RNA. The resulting RNA molecule is called **messenger RNA (mRNA)**.

```
DNA template:  3'— T A C G G C T A —5'
                    | | | | | | | |
mRNA produced: 5'— A U G C C G A U —3'
```

Key points:
- Only one strand of DNA is used as the template (the **template strand**)
- The other strand (the **coding strand** or **sense strand**) has the same sequence as the mRNA, just with T instead of U
- Transcription is controlled by **promoters** — short DNA sequences upstream of a gene that signal where to start
- In humans, transcription happens in the nucleus

The region of DNA that gets transcribed into a functional RNA is called a **gene** (though the definition is more nuanced in practice).

## 3. RNA Processing (Eukaryotes)

In humans and other eukaryotes, the initial RNA transcript (pre-mRNA) is processed before it leaves the nucleus:

- **5' cap** added — protects the mRNA and helps with translation
- **Poly-A tail** added at the 3' end — stability and export signal
- **Splicing** — non-coding regions called **introns** are removed; coding regions called **exons** are joined together

```
pre-mRNA:  [exon1]—[intron]—[exon2]—[intron]—[exon3]
                    ↓ splicing
mRNA:      [exon1]—[exon2]—[exon3]
```

**Alternative splicing** allows different combinations of exons to be joined, producing multiple different proteins from a single gene. This is why humans have ~20,000 genes but hundreds of thousands of distinct proteins.

For ML: the distinction between genomic DNA coordinates and mRNA/transcript coordinates is a constant source of confusion in bioinformatics. Tools like GENCODE and Ensembl provide annotation files (GTF/GFF) that map genes, exons, and transcripts to genome coordinates.

## 4. Translation: RNA → Protein

**Translation** is the process of reading the mRNA sequence and building a protein. It happens on **ribosomes** in the cytoplasm.

The mRNA is read in triplets called **codons**. Each codon specifies one amino acid (or a stop signal):

```
mRNA:  5'— AUG — CCG — GAU — UAA —3'
            Met   Pro   Asp   Stop
```

- **AUG** (methionine) is the start codon — translation always begins here
- **UAA, UAG, UGA** are stop codons — translation ends here
- The mapping from codons to amino acids is the **genetic code**

```python
genetic_code = {
    'AUG': 'Met', 'UUU': 'Phe', 'UUC': 'Phe',
    'UUA': 'Leu', 'UUG': 'Leu', 'UCU': 'Ser',
    # ... 64 codons total
    'UAA': 'Stop', 'UAG': 'Stop', 'UGA': 'Stop'
}

def translate(mrna: str) -> list[str]:
    """Translate mRNA to amino acid sequence."""
    protein = []
    for i in range(0, len(mrna) - 2, 3):
        codon = mrna[i:i+3]
        aa = genetic_code.get(codon, '?')
        if aa == 'Stop':
            break
        protein.append(aa)
    return protein
```

There are 64 possible codons but only 20 amino acids — the code is **redundant** (multiple codons map to the same amino acid). This redundancy is not random; synonymous codons tend to differ only at the third position.

## 5. Types of RNA

Not all RNA becomes protein. Many RNA molecules are functional in their own right:

| Type | Function |
|------|----------|
| mRNA | Template for protein synthesis |
| tRNA | Carries amino acids to the ribosome |
| rRNA | Structural component of ribosomes |
| miRNA | Regulates gene expression post-transcriptionally |
| lncRNA | Long non-coding RNA; diverse regulatory roles |
| snRNA | Involved in splicing |

Non-coding RNAs are increasingly important in disease and are targets for ML models.

## 6. Gene Expression

The full process — from DNA to functional protein — is called **gene expression**. Cells regulate expression tightly: different cell types express different genes even though they all contain the same DNA.

Expression is measured experimentally by:
- **RNA-seq** — sequences all mRNA in a sample, giving a quantitative readout of which genes are active
- **Microarrays** — older technology, hybridization-based

The output is typically a matrix of **expression values** (read counts or normalized values) per gene per sample — a natural input for ML models.

## 7. Conclusion

The central dogma in one line: DNA is transcribed to RNA, which is translated to protein.

Key things to remember:
- **Transcription**: DNA → mRNA (in the nucleus)
- **Splicing**: introns removed, exons joined (eukaryotes only)
- **Translation**: mRNA → protein (in the cytoplasm), read in codons of 3
- **Gene expression** is tightly regulated and varies by cell type and condition
- Most ML models in genomics work at the DNA or RNA level, not the protein level

Next: **DNA Replication** — how the genome is copied every time a cell divides.
