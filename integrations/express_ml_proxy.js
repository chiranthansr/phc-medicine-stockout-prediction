// Optional integration for an existing Node.js 18+ / Express backend.
// Mount this router with: app.use('/api/ml', require('./integrations/express_ml_proxy'));
// Keep the Python FastAPI service private/local; the React UI talks to Express.
const express = require('express');
const router = express.Router();

const ML_API_BASE = process.env.PHC_ML_API_URL || 'http://127.0.0.1:8000';

async function forwardJson(path, req, res) {
  try {
    const upstream = await fetch(`${ML_API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body ?? {}),
    });
    const payload = await upstream.json().catch(() => ({ detail: 'ML service returned a non-JSON response.' }));
    return res.status(upstream.status).json(payload);
  } catch (error) {
    return res.status(502).json({
      error: 'ML_SERVICE_UNAVAILABLE',
      detail: 'Could not reach the Python PHC ML API. Check that api.py is running.',
    });
  }
}

router.get('/health', async (_req, res) => {
  try {
    const upstream = await fetch(`${ML_API_BASE}/health`);
    return res.status(upstream.status).json(await upstream.json());
  } catch (_error) {
    return res.status(502).json({ error: 'ML_SERVICE_UNAVAILABLE' });
  }
});

router.get('/catalog', async (_req, res) => {
  try {
    const upstream = await fetch(`${ML_API_BASE}/catalog`);
    return res.status(upstream.status).json(await upstream.json());
  } catch (_error) {
    return res.status(502).json({ error: 'ML_SERVICE_UNAVAILABLE' });
  }
});

router.post('/forecast', (req, res) => forwardJson('/forecast', req, res));
router.post('/stockout-risk', (req, res) => forwardJson('/stockout-risk', req, res));

module.exports = router;
