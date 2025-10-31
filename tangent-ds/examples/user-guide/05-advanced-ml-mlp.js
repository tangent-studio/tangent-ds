// ---
// title: 05 — Advanced ML (MLP)
// id: advanced-ml-mlp
// ---

// %% [markdown]
/*
# 5 — Advanced ML: MLP (tensorflow.js)

Purpose:
- Train small neural networks for classification/regression.
- Persist models using the tfjs IOHandler API (model.save / tf.loadLayersModel).
- DO NOT use model.toJSON for non-trivial LayersModel/MLP persistence.
*/

// %% [javascript]
import * as ds from "@tangent.to/ds";

if (ds.ml && ds.ml.MLPClassifier) {
  // Prepare data from previous preprocessing
  const X_train = globalThis.X_train_scaled;
  const X_test = globalThis.X_test_scaled;
  const y_train = globalThis.y_train_encoded;
  const y_test = globalThis.y_test_encoded;

  const mlp = new ds.ml.MLPClassifier({
    hiddenLayers: [10, 5],
    activation: "relu",
    epochs: 50,
    learningRate: 0.01,
    seed: 42
  });

  await mlp.fit(X_train, y_train);
  globalThis.mlp = mlp;

  const preds = mlp.predict(X_test);
  const acc = y_test.filter((y, i) => y === preds[i]).length / y_test.length;
  console.log(`MLP accuracy: ${(acc * 100).toFixed(1)}%`);
  console.log("Hidden layers:", mlp.hiddenLayers);

  // Persistence: prefer IOHandler API. Attempt to save using model.save if exposed.
  try {
    if (typeof mlp.save === "function") {
      // If the wrapper exposes a save method, prefer using it (implementation-specific)
      console.log("Saving MLP via mlp.save(...) (wrapper-provided).");
      // await mlp.save('file://./artifacts/mlp-model'); // example — uncomment in Node with tfjs-node
    } else if (mlp.model && typeof mlp.model.save === "function") {
      // Access underlying tfjs LayersModel and use standard IO
      console.log("Saving underlying tfjs model via model.save(...). Use 'file://' in Node or custom IOHandler.");
      // await mlp.model.save('file://./artifacts/mlp-model'); // example — uncomment in Node with tfjs-node
    } else {
      console.log("No save API found on MLP wrapper. Implement custom IOHandler and use tf.io.withSaveHandler / withLoadHandler.");
    }
  } catch (e) {
    console.log("Saving MLP failed or skipped (environment may not support file://).", e.message);
  }

  console.log("Reminder: avoid model.toJSON(...) for complex LayersModel persistence.");
} else {
  console.log("ds.ml.MLPClassifier not available — skip MLP demo.");
}
