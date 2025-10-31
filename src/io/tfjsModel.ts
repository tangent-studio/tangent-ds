TODO
- the workflow publishes to npm and pushes the site on tag
- implement Gaussian processes inspired by https://github.com/jmonlabs/jmon-algo/tree/main/src/algorithms/generative/gaussian-processes, then generate python tests and examples
- fix LDA
- can MLP (tensorflowjs) be saved/loaded safely as json?
  - Resolution: do NOT use model.toJSON for MLPs (it can throw errors for LayersModel / complex models). Use the tfjs IOHandler API instead:
    1. Save with model.save(ioHandlerOrURL) and load with tf.loadLayersModel(ioHandlerOrURL). IO handlers may be URL strings (e.g. 'file://path' in Node) or custom handlers created with tf.io.withSaveHandler / tf.io.withLoadHandler.
    2. In Node, prefer model.save('file://dir') — produces model.json + binary weight shard(s) (.bin). This is reliable and recommended.
    3. If you need a custom persistence target, implement a custom IOHandler rather than embedding weights via toJSON.
    4. Add helpers (src/io/tfjsModel.ts) that wrap model.save / tf.loadLayersModel and add a round-trip test (save -> load -> compare predictions within tolerance).
    5. Policy: avoid committing large .bin weight files to git; use model registries or release assets.
- manual tests in tangent-notebooks

DONE
- scikit-learn's-like API across the whole pakcage, ml, mvs and plot
- fix logistic regression
- fix kmeans
- test Python similarity
- implement KNN, decision trees, random forest and generalized additive models (both in classification and regression)
- implement python tests and examples for KNN, decision trees, random forest and generalized additive models
- fix PCA, implement SVD
- review train, validation, preprocessing, tunig and interpret
- review plot, one ordiplot module for ordinations (PCA, LDA, RDA), add plot for feature importance, add ROC curves, add other ploting facilitators you might think useful...
- model persistance in json format

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Ensure tfjs-node is loaded in Node so 'file://' IO works.
 * This is a no-op in environments where tfjs-node isn't available.
 */
export async function ensureTfNodeRegistered(): Promise<unknown | null> {
	try {
		// eslint-disable-next-line @typescript-eslint/no-var-requires
		const tfnode = require('@tensorflow/tfjs-node');
		return tfnode;
	} catch (e) {
		return null;
	}
}

/**
 * Save a LayersModel to a directory using the standard tfjs file:// IO.
 * Produces model.json + binary weight shard(s) (.bin).
 */
export async function saveModelToDir(model: tf.LayersModel, dirPath: string): Promise<tf.io.SaveResult> {
	await ensureTfNodeRegistered();
	await fs.promises.mkdir(dirPath, { recursive: true });
	const url = `file://${path.resolve(dirPath)}`;
	return model.save(url);
}

/**
 * Load a LayersModel from a directory saved by saveModelToDir.
 */
export async function loadModelFromDir(dirPath: string): Promise<tf.LayersModel> {
	await ensureTfNodeRegistered();
	const url = `file://${path.resolve(dirPath)}/model.json`;
	const model = await tf.loadLayersModel(url);
	return model as tf.LayersModel;
}

/**
 * Create a tfjs save IOHandler from a custom save function.
 * Useful when you want to persist model artifacts to a custom backend.
 */
export function createSaveHandler(saveFn: (artifacts: tf.io.ModelArtifacts) => Promise<tf.io.SaveResult>) {
	return tf.io.withSaveHandler(async (artifacts: tf.io.ModelArtifacts) => {
		return saveFn(artifacts);
	});
}

/**
 * Create a tfjs load IOHandler from a custom load function.
 * Useful when you want to load model artifacts from a custom backend.
 */
export function createLoadHandler(loadFn: (options?: tf.io.LoadOptions) => Promise<tf.io.ModelArtifacts>) {
	return tf.io.withLoadHandler(async (options?: tf.io.LoadOptions) => {
		return loadFn(options);
	});
}

// Note: do NOT use model.toJSON(...) to persist MLPs in general — it can throw for LayersModel/complex models.
// Prefer the IOHandler approach above (file://, indexeddb://, custom handlers).