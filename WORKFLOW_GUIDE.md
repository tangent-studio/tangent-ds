# Interactive Development Workflows for @tangent/ds

## TL;DR - Recommended Setup

**For rapid prototyping & debugging (no plots): Zed + Deno Kernel** ⭐ FASTEST
- Instant inline results
- Cell-by-cell execution
- No browser needed
- Perfect for testing algorithms, data processing, model training

**For testing with visualizations: Tangent Notebook** ⭐ REQUIRED FOR PLOTS
- Only option where plots actually render
- Interactive notebooks with markdown
- Same as production environment
- Use for ROC curves, confusion matrices, ordiplots, etc.

**For testing full examples: Node.js + watch mode**
**For production testing: npm link + consumer project**

### Quick Decision Tree

```
Need to see plots/visualizations?
├─ YES → Use Tangent Notebook (Option 3)
└─ NO → Testing algorithms/data processing?
    ├─ YES → Use Zed + Deno Kernel (Option 1) - FASTEST
    └─ NO → Running full examples?
        └─ YES → Use Node.js + watch (Option 2)
```

---

## Option 1: Zed + Deno Kernel (⭐ RECOMMENDED for iterative development)

### Why This is Best

✅ **Zero latency** - results appear inline instantly
✅ **No browser** - no switching windows, no refreshing
✅ **Cell-based** - test individual functions without re-running everything
✅ **Live updates** - change code, save, re-run cell
✅ **Full ESM** - works perfectly with your bundle

### Setup

**Terminal 1: Serve bundle with auto-rebuild**
```bash
cd tangent-ds
npm run watch  # Automatically rebuilds on src changes
```

**Terminal 2: Serve the bundle**
```bash
cd tangent-ds/dist
npx http-server -p 8080 --cors -c-1
```

**In Zed:**
1. Open `examples/ml_usage_deno.js`
2. Enable Deno kernel (should auto-detect `.js` with `await import`)
3. Run cells with `Cmd/Ctrl + Enter`
4. Edit code, save, re-run cell - instant feedback!

### Example Cell Workflow

```javascript
// Cell 1: Import (run once)
const ds = await import('http://localhost:8080/index.js');
const { ml, stats, core } = ds;

// Cell 2: Define data
const data = [[0,0], [1,1], [10,10]];

// Cell 3: Test function (re-run this cell to test changes)
ml.kmeans.fit(data, { k: 2 })
// Results appear inline immediately!

// Cell 4: Tweak and re-run
ml.kmeans.fit(data, { k: 3, maxIter: 50 })

// Cell 5: Debug a specific issue
const model = ml.kmeans.fit(data, { k: 2 });
console.log('Centroids:', model.centroids);
model.labels  // Shows inline
```

### When to Use
- **Debugging a specific function** - run just that cell
- **Exploring API** - test different parameters quickly
- **Prototyping** - try ideas without ceremony
- **Data exploration** - inspect intermediate results

### Limitations
- Need to serve bundle (but watch mode makes this seamless)
- Deno runtime differences (rare, but possible)

---

## Option 2: Node.js + Watch Mode (Good for full example testing)

### Setup

**Terminal 1: Watch mode**
```bash
npm run watch
```

**Terminal 2: Run example**
```bash
node examples/ml_usage_node.js

# Or with auto-reload on changes:
npx nodemon --watch src --watch examples examples/ml_usage_node.js
```

### When to Use
- Testing complete examples end-to-end
- Verifying Node.js compatibility
- Running full integration tests

### Limitations
- Have to re-run entire file
- No inline results
- Less interactive

---

## Option 3: Tangent Notebook (⭐ REQUIRED for visualizations)

**This is the ONLY option where plots will actually render!**

### Why Use This

✅ **Plots render** - Observable Plot configurations become actual visualizations
✅ **Interactive** - Cell-based execution with results
✅ **Complete environment** - Same as production usage
✅ **Markdown support** - Formatted explanations between cells

### Setup

**Terminal 1: Watch tangent-ds**
```bash
cd tangent-ds
npm run watch
```

**Terminal 2: Link package and run Tangent**
```bash
cd tangent-ds
npm link

cd ../tangent-notebook/frontend
npm link @tangent/ds
npm run dev
```

**Browser:**
- Open http://localhost:5173
- Load examples from `tangent-ds/examples/tangent/`
  - `01-machine-learning.js` - ML algorithms
  - `02-multivariate-analysis.js` - PCA, LDA, RDA with plots
  - `03-statistics.js` - Regression and statistical tests
  - `04-visualization.js` - All plot types (ROC, confusion matrix, etc.)
  - `05-preprocessing-validation.js` - Data preparation
- Run cells interactively

### When to Use
- **Testing visualizations** - plots only render here
- **Full examples** - with markdown explanations
- **Documentation** - creating user-facing examples
- **Presentations** - showcasing functionality

### Limitations
- Requires browser and Tangent Notebook
- Need to restart Vite after npm link
- Slower than Deno REPL for non-visual testing

### Important Notes

**Plot functions return configurations, not rendered plots:**
```javascript
// In Deno REPL - you only see the config object:
const config = ds.plot.plotROC(yTrue, yProb);
// Output: { type: 'roc', data: {...}, marks: [...] }

// In Tangent Notebook - this renders as an actual plot!
const config = ds.plot.plotROC(yTrue, yProb);
// Output: Beautiful ROC curve visualization
```

**For testing without plots, use Deno REPL (Option 1)**
**For testing with plots, use Tangent Notebook (Option 3)**

---

## Option 4: Node REPL (Quick one-liners)

```bash
node --experimental-repl-await
```

```javascript
// In REPL
const ds = await import('./src/index.js');
const { ml } = ds;

ml.kmeans.fit([[0,0], [1,1]], { k: 2 });
// { labels: [0, 1], centroids: [[0,0], [1,1]], ... }

// Quick iteration
const test = (data, k) => ml.kmeans.fit(data, { k });
test([[0,0], [1,1], [2,2]], 2);
```

### When to Use
- Quick tests
- Checking function signatures
- Debugging imports

---

## Option 5: Deno REPL (Alternative to Node REPL)

```bash
deno repl
```

```javascript
// Import from URL
const ds = await import('http://localhost:8080/index.js');
const { ml } = ds;

// Same API as Node REPL
ml.kmeans.fit([[0,0], [1,1]], { k: 2 });
```

### When to Use
- Prefer Deno over Node
- Want URL imports in REPL
- Testing bundle directly

---

## Workflow Comparison

| Workflow | Speed | Interactive | Node Compat | Browser | Setup |
|----------|-------|-------------|-------------|---------|-------|
| **Zed + Deno** | ⚡⚡⚡ | ✅✅✅ | ⚠️ | ✅ | Easy |
| **Node + Watch** | ⚡⚡ | ❌ | ✅✅✅ | ❌ | Easy |
| **Tangent NB** | ⚡ | ✅✅ | ❌ | ✅✅✅ | Medium |
| **Node REPL** | ⚡⚡⚡ | ✅ | ✅✅✅ | ❌ | Instant |
| **Deno REPL** | ⚡⚡⚡ | ✅ | ⚠️ | ✅ | Instant |

---

## Recommended Daily Workflow

### Morning Setup (Run Once)
```bash
# Terminal 1: Build watcher
cd tangent-ds && npm run watch

# Terminal 2: Serve bundle
cd tangent-ds/dist && npx http-server -p 8080 --cors -c-1

# Terminal 3: Ready for commands
cd tangent-ds
```

### Iteration Loop

1. **Code in Zed**: Edit `src/ml/kmeans.js`
2. **Auto-build**: Watch mode rebuilds dist/index.js
3. **Test in Zed**: Re-run cell in `ml_usage_deno.js`
4. **See results**: Inline, instantly
5. **Repeat**: No ceremony, no waiting

### Before Commit

```bash
# Run full test suite
npm run test:run

# Test Node.js examples
node examples/ml_usage_node.js

# Optional: Test in consumer
cd ../tangent-notebook/frontend && npm run dev
```

---

## Tips & Tricks

### Zed Keyboard Shortcuts
- `Cmd/Ctrl + Enter`: Run current cell
- `Shift + Enter`: Run cell and move to next
- `Cmd/Ctrl + Shift + Enter`: Run all cells

### Quick Debugging in Zed
```javascript
// Add this cell to inspect variables
console.table(model.centroids);
console.dir(model, { depth: 3 });
```

### Fast Package Testing
```javascript
// Cell 1: Import with cache busting
const ds = await import(`http://localhost:8080/index.js?${Date.now()}`);

// Cell 2: Test immediately
ds.ml.kmeans.fit([[0,0]], { k: 1 });
```

### Profile Performance
```javascript
// Cell: Time operations
console.time('kmeans');
const model = ml.kmeans.fit(largeData, { k: 5 });
console.timeEnd('kmeans');
```

---

## Troubleshooting

### "Module not found" in Zed
- ✅ Check http-server is running: `curl http://localhost:8080/index.js`
- ✅ Check watch mode rebuilt: Look for "⚡ Done" message
- ✅ Clear Deno cache: `deno cache --reload http://localhost:8080/index.js`

### "Changes not reflecting"
- ✅ Save file first
- ✅ Watch mode shows rebuild
- ✅ Re-run cell (not whole file)
- ✅ Check bundle timestamp: `stat dist/index.js`

### "Import map not working"
- Zed uses Deno, which needs full URLs
- Use `http://localhost:8080/index.js` not `@tangent/ds`
- For `@tangent/ds` imports, use Node.js workflow instead

---

## Best Practices

### ✅ DO
- Use Zed + Deno for prototyping and debugging
- Run `npm run test:run` before committing
- Test in Node.js before releasing
- Keep `npm run watch` running during development

### ❌ DON'T
- Don't test only in Deno (verify in Node too)
- Don't skip the test suite
- Don't forget to rebuild before npm pack
- Don't commit without verifying examples run

---

## Example: Adding a New Function

**1. Write function** (`src/ml/kmeans.js`):
```javascript
export function betterKmeans(data, options) {
  // implementation
}
```

**2. Export** (`src/ml/index.js`):
```javascript
export { betterKmeans } from './kmeans.js';
```

**3. Test in Zed** (new cell in `ml_usage_deno.js`):
```javascript
// Auto-rebuilds from watch mode
const ds = await import(`http://localhost:8080/index.js?${Date.now()}`);
ds.ml.betterKmeans([[0,0], [1,1]], { k: 2 })
```

**4. Write test** (`src/ml/kmeans.test.js`):
```javascript
import { test, expect } from 'vitest';
import { betterKmeans } from './kmeans.js';

test('betterKmeans works', () => {
  const result = betterKmeans([[0,0], [1,1]], { k: 2 });
  expect(result.labels).toHaveLength(2);
});
```

**5. Verify**:
```bash
npm run test:run
node examples/ml_usage_node.js
```

Total time: **seconds**, not minutes!

---

## Summary

**Use Zed + Deno kernel for 95% of your development work.** It's the fastest, most interactive way to build and debug your package. Reserve Node.js and Tangent Notebook for final verification before release.

Happy coding! 🚀
