# Frozen pathway graph

The primary support contains the 18 returned edges among
`EGFR, ERBB2, ERBB3, SRC, AKT1, PTEN, TSC2, MTOR, RPS6, EIF4EBP1` from the
version-pinned STRING 12.0 API:

```text
https://version-12-0.string-db.org/api/tsv/network
species=9606
required_score=900
network_type=functional
add_nodes=0
```

The raw TSV response to the query made on 2026-08-08 has SHA-256
`17abcfd93fd67d30c8ed9783a926fe4fb81160f28690f181e2be41ffc3f5536d`.
The fields used by the analysis are frozen in `graph_manifest.csv`. STRING
describes these as functional associations; they are not necessarily direct
physical binding relationships. The complete 45-edge support is a required
sensitivity analysis because failure under the sparse support can reflect an
omitted edge rather than T-cell-associated heterogeneity.

API documentation: <https://version-12-0.string-db.org/help/api/>
