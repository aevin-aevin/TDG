"""Supplementary Code 2. GC adjustment of the 22 by-gene parameters.

Usage:  python Supplementary_Code_2_GC_adjustment.py by_gene.txt gc.txt outdir
  by_gene.txt : tab-separated, columns Gene, NCBid, phyl17 and the 22 parameters (Supplementary Data 1, sheet by_gene)
  gc.txt      : tab-separated, same gene order, with the five GC-percent columns (Ensembl BioMart):
                GCperc_wholeGene, GCperc__CDS, GCperc__mRNA, GC_perc__mRNA (5'UTR), GCperc__1500UpTSS
Writes outdir/raw_cc.txt (unadjusted, complete-case genes), adjA.txt (adjusted for whole-gene GC, linear+quadratic),
adjB.txt (adjusted for all five GC measures) and gc_R2_by_param.csv. Each file is then analysed with
Supplementary Code 1:   MINN=1000 python Supplementary_Code_1_tdg_all.py outdir/adjB.txt outdir_B 100
"""
import sys, os, numpy as np, pandas as pd
src, gcf, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
a = pd.read_csv(src, sep='\t'); g = pd.read_csv(gcf, sep='\t')
assert len(a) == len(g) and (a.Gene.values == g.Gene.values).all(), 'gene order differs'
gc = ['GCperc_wholeGene','GCperc__CDS','GCperc__mRNA','GC_perc__mRNA','GCperc__1500UpTSS']
a = pd.concat([a, g[gc]], axis=1)
cc = a.dropna(subset=gc).reset_index(drop=True); print('complete-case genes', len(cc))
meta = ['Gene','NCBid','phyl17']; par = [c for c in a.columns if c not in meta+gc]
Z = cc[gc].apply(lambda x: (x-x.mean())/x.std())
dA = np.column_stack([np.ones(len(cc)), Z['GCperc_wholeGene'], Z['GCperc_wholeGene']**2])
dB = np.column_stack([np.ones(len(cc))]+[Z[c] for c in gc]+[Z['GCperc_wholeGene']**2, Z['GCperc__1500UpTSS']**2])
def resid(y, D):
    ok = y.notna().values; o = pd.Series(np.nan, index=y.index)
    beta = np.linalg.lstsq(D[ok], y[ok].values, rcond=None)[0]
    o[ok] = y[ok].values - D[ok]@beta + y[ok].mean()
    return o, 1-np.var(y[ok].values-D[ok]@beta)/np.var(y[ok].values)
A = cc[meta].copy(); B = cc[meta].copy(); r2 = []
for c in par:
    A[c], ra = resid(cc[c], dA); B[c], rb = resid(cc[c], dB)
    r2.append(dict(parameter=c, R2_wholeGeneGC=ra, R2_allGC=rb))
cc[meta+par].to_csv(f'{out}/raw_cc.txt', sep='\t', index=False)
A.to_csv(f'{out}/adjA.txt', sep='\t', index=False); B.to_csv(f'{out}/adjB.txt', sep='\t', index=False)
pd.DataFrame(r2).to_csv(f'{out}/gc_R2_by_param.csv', index=False)
