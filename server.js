
const cors = require('cors');
const express = require('express');
const pool = require('./db');

const app = express();
const PORT = Number(process.env.PORT) || 3000;

const ML_API_BASE =
    process.env.PHC_ML_API_URL || 'http://127.0.0.1:8000';

app.use(cors({
    origin: 'http://localhost:5173'
}));

app.use(express.json({ limit: '1mb' }));

// --------------------------------------------------
// 1. Health check
// --------------------------------------------------
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'PHC Backend API'
    });
});

// --------------------------------------------------
// 2. Get all medicines from PostgreSQL
// --------------------------------------------------
app.get('/api/medicines', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                medicine_id,
                medicine_name,
                unit_of_measure
            FROM medicines
            ORDER BY medicine_id;
        `);

        res.json({
            count: result.rows.length,
            medicines: result.rows
        });
    } catch (error) {
        console.error('Failed to fetch medicines:', error.message);

        res.status(500).json({
            error: 'DATABASE_QUERY_FAILED'
        });
    }
});

// --------------------------------------------------
// 3. Get daily inventory
// --------------------------------------------------
app.get('/api/inventory', async (req, res) => {
    try {
        const { phc_id, medicine_id } = req.query;
        const requestedLimit = Number(req.query.limit ?? 100);

        if (
            !Number.isInteger(requestedLimit) ||
            requestedLimit < 1 ||
            requestedLimit > 500
        ) {
            return res.status(400).json({
                error: 'limit must be an integer between 1 and 500'
            });
        }

        const conditions = [];
        const values = [];

        if (phc_id) {
            values.push(phc_id);
            conditions.push(`phc_id = $${values.length}`);
        }

        if (medicine_id) {
            values.push(medicine_id);
            conditions.push(`medicine_id = $${values.length}`);
        }

        const whereClause = conditions.length
            ? `WHERE ${conditions.join(' AND ')}`
            : '';

        values.push(requestedLimit);

        const result = await pool.query(`
            SELECT
                inventory_date,
                phc_id,
                medicine_id,
                opening_stock,
                expired_quantity,
                quantity_received,
                available_stock,
                total_demand,
                dispensed_quantity,
                unmet_demand,
                closing_stock,
                stockout_flag,
                patient_visits,
                emergency_flag,
                crisis_flag,
                minimum_stock_level,
                reorder_level,
                days_to_nearest_expiry
            FROM daily_inventory
            ${whereClause}
            ORDER BY inventory_date DESC, phc_id, medicine_id
            LIMIT $${values.length};
        `, values);

        res.json({
            count: result.rows.length,
            filters: {
                phc_id: phc_id || null,
                medicine_id: medicine_id || null,
                limit: requestedLimit
            },
            inventory: result.rows
        });
    } catch (error) {
        console.error('Failed to fetch inventory:', error.message);

        res.status(500).json({
            error: 'DATABASE_QUERY_FAILED'
        });
    }
});

// --------------------------------------------------
// 4. Get inventory history for one PHC and medicine
// --------------------------------------------------
app.get('/api/inventory/history', async (req, res) => {
    try {
        const { phc_id, medicine_id } = req.query;

        if (!phc_id || !medicine_id) {
            return res.status(400).json({
                error: 'Both phc_id and medicine_id are required.'
            });
        }

        const result = await pool.query(
            `SELECT
                inventory_date,
                phc_id,
                medicine_id,
                opening_stock,
                available_stock,
                total_demand,
                dispensed_quantity,
                unmet_demand,
                closing_stock,
                stockout_flag,
                minimum_stock_level,
                reorder_level
             FROM daily_inventory
             WHERE phc_id = $1
               AND medicine_id = $2
             ORDER BY inventory_date ASC`,
            [phc_id, medicine_id]
        );

        res.json({
            count: result.rows.length,
            phc_id,
            medicine_id,
            history: result.rows
        });
    } catch (error) {
        console.error('Failed to fetch inventory history:', error.message);

        res.status(500).json({
            error: 'DATABASE_QUERY_FAILED'
        });
    }
});

// --------------------------------------------------
// 5. Shared helper: call Python FastAPI and save result
// --------------------------------------------------
async function runAndSavePrediction(req, res, predictionType) {
    const { phc_id, medicine_id } = req.body ?? {};

    if (
        typeof phc_id !== 'string' ||
        !phc_id.trim() ||
        typeof medicine_id !== 'string' ||
        !medicine_id.trim()
    ) {
        return res.status(400).json({
            error: 'phc_id and medicine_id are required strings.'
        });
    }

    try {
        // Call the existing Python ML API.
        let mlResponse;

        try {
            mlResponse = await fetch(
                `${ML_API_BASE}/${predictionType === 'forecast'
                    ? 'forecast'
                    : 'stockout-risk'}`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(req.body),
                    signal: AbortSignal.timeout(30000)
                }
            );
        } catch (error) {
            console.error('Could not reach Python ML API:', error.message);

            return res.status(502).json({
                error: 'ML_SERVICE_UNAVAILABLE',
                detail: 'Check that the Python FastAPI service is running on port 8000.'
            });
        }

        const prediction = await mlResponse.json().catch(() => null);

        if (!mlResponse.ok) {
            return res.status(mlResponse.status).json(
                prediction ?? {
                    error: 'ML_PREDICTION_FAILED'
                }
            );
        }

        if (
            !prediction ||
            typeof prediction !== 'object' ||
            Array.isArray(prediction)
        ) {
            return res.status(502).json({
                error: 'INVALID_ML_RESPONSE'
            });
        }

        // Save the original request and ML response in PostgreSQL.
        const saved = await pool.query(
            `INSERT INTO predictions (
                phc_id,
                medicine_id,
                prediction_type,
                request_payload,
                response_payload,
                stockout_score,
                threshold_used,
                flag_at_threshold,
                source_data_type
            )
            VALUES (
                $1, $2, $3, $4::jsonb, $5::jsonb,
                $6, $7, $8, $9
            )
            RETURNING prediction_id, created_at`,
            [
                phc_id.trim(),
                medicine_id.trim(),
                predictionType,
                JSON.stringify(req.body),
                JSON.stringify(prediction),
                predictionType === 'stockout-risk'
                    ? (prediction.uncalibrated_stockout_score ?? prediction.stockout_score ?? prediction.score ?? null)
                    : null,
                predictionType === 'stockout-risk'
                    ? (prediction.validation_selected_threshold ?? prediction.threshold_used ?? prediction.threshold ?? null)
                    : null,
                predictionType === 'stockout-risk'
                    ? (prediction.flag_at_threshold ?? prediction.flag ?? null)
                    : null,
                'SYNTHETIC'
            ]
        );

        return res.status(200).json({
            ...prediction,
            saved_prediction: saved.rows[0]
        });
    } catch (error) {
        console.error(
            `${predictionType} prediction or database save failed:`,
            error.message
        );

        return res.status(500).json({
            error: 'PREDICTION_OR_DATABASE_SAVE_FAILED',
            detail: 'The prediction could not be saved. Check the backend terminal logs.'
        });
    }
}

// --------------------------------------------------
// 6. Demand forecast endpoint
// --------------------------------------------------
app.post('/api/forecast', (req, res) => {
    return runAndSavePrediction(req, res, 'forecast');
});

// --------------------------------------------------
// 7. Stockout-risk prediction endpoint
// --------------------------------------------------
app.post('/api/stockout-risk', (req, res) => {
    return runAndSavePrediction(req, res, 'stockout-risk');
});

// --------------------------------------------------
// 8. Start Express
// --------------------------------------------------
app.listen(PORT, () => {
    console.log(`Backend API running at http://localhost:${PORT}`);
    console.log(`Python ML API: ${ML_API_BASE}`);
});
