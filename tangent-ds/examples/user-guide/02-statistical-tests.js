// ---
// title: 02 — Statistical tests
// id: statistical-tests
// ---

// %% [markdown]
/*
# 2 — Statistical tests: ANOVA, LM, LMM (illustrative)

Purpose:
- Fit linear models and run ANOVA to test hypotheses.
- Use mixed models (LMM) when data are clustered / grouped.

Notes:
- This notebook is illustrative: adapt to your concrete stats API (ds.stats.LinearModel / MixedModel).
*/

// %% [javascript]
import * as ds from "@tangent.to/ds";

try {
  if (ds.stats && ds.stats.LinearModel) {
    // Prepare a simple DataFrame-like array for the example
    globalThis.irisDF = globalThis.irisData.map(d => ({
      sepalLength: d.sepalLength,
      petalLength: d.petalLength,
      species: d.species
    }));

    // Fit linear model (example): sepalLength ~ petalLength
    globalThis.lm = new ds.stats.LinearModel({ formula: "sepalLength ~ petalLength" });
    await globalThis.lm.fit(globalThis.irisDF);
    console.log("LinearModel summary:");
    console.log(globalThis.lm.summary());

    // ANOVA example (if supported)
    if (ds.stats.anova) {
      // illustrate comparing full vs reduced model (pseudocode)
      const full = globalThis.lm;
      const reduced = new ds.stats.LinearModel({ formula: "sepalLength ~ 1" });
      await reduced.fit(globalThis.irisDF);
      const anovaRes = ds.stats.anova(full, reduced);
      console.log("ANOVA result:", anovaRes);
    } else {
      console.log("ANOVA helper not found in ds.stats — illustrative only.");
    }
  } else {
    console.log("ds.stats.LinearModel not available — statistical tests are illustrative.");
    // Pseudocode:
    // lm = LinearModel("y ~ x + (1|group)").fit(df)
    // print(lm.summary())
    // anova(lm1, lm2)
    // MixedModel("y ~ x + (1|group)").fit(df)
  }
} catch (e) {
  console.log("Statistical tests demo skipped due to error:", e.message);
}
