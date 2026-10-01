# ImageNet-100-C Legacy Artifacts

This bundle contains historical seed-0 training diagnostics and pilot figures.
The current scope and interpretation are maintained in
[docs/RESULTS.md](../docs/RESULTS.md) and
[docs/README_IMAGENET100C.md](../docs/README_IMAGENET100C.md).

These files are not report-grade held-out deployment results and must not be
presented as five-seed evidence.

## Focused seed-1/2 bundle

Available focused evaluations for seeds `1,2` cover balanced, long-tail, and
missing support for ERM, GroupDRO, and IRO. They use four corruption anchors,
five severities, and 1,000 class-stratified validation images.

Derived artifacts are under:

```text
tables/focused_seed12/
figures/focused_seed12/
```

This bundle contains 18 evaluation records. It is a partial two-seed artifact,
not the complete seven-method candidate matrix; IRM, INF-TASK, EQRM, and VREx
focused evaluations are not included here yet.
