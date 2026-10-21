import { useEffect, useMemo, useState } from "react";

const API_BASE =
  import.meta.env.VITE_PHC_API_URL || "http://127.0.0.1:8000";
const BACKEND_BASE =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

const number = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const displayNumber = (value, maximumFractionDigits = 1) => {
  const parsed = number(value);
  return parsed === null
    ? "—"
    : parsed.toLocaleString(undefined, { maximumFractionDigits });
};

export default function PHCForecastExample() {
  const [phcId, setPhcId] = useState("");
  const [medicineId, setMedicineId] = useState("");
  const [horizonDays, setHorizonDays] = useState("");
  const [currentStock, setCurrentStock] = useState("");
  const [crisis, setCrisis] = useState(false);
  const [catalog, setCatalog] = useState({ phcs: [], medicines: [] });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [inventoryHistory, setInventoryHistory] = useState([]);
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showInventoryHistory, setShowInventoryHistory] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadCatalog() {
      try {
        const response = await fetch(`${API_BASE}/catalog`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Catalogue request failed (${response.status})`);
        }

        const data = await response.json();
        setCatalog({
          phcs: Array.isArray(data.phcs) ? data.phcs : [],
          medicines: Array.isArray(data.medicines) ? data.medicines : [],
        });

      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Unable to load the catalogue.");
        }
      }
    }

    loadCatalog();
    return () => controller.abort();
  }, []);

  const medicinesForPhc = useMemo(
    () => catalog.medicines.filter((medicine) => medicine.PHC_ID === phcId),
    [catalog.medicines, phcId]
  );

  useEffect(() => {
    if (!phcId || !medicineId) {
      setInventoryHistory([]);
      return undefined;
    }

    const controller = new AbortController();

    async function loadInventoryHistory() {
      setHistoryLoading(true);
      setHistoryError("");
      setInventoryHistory([]);

      try {
        const params = new URLSearchParams({
          phc_id: phcId,
          medicine_id: medicineId,
        });
        const response = await fetch(
          `${BACKEND_BASE}/api/inventory/history?${params.toString()}`,
          { signal: controller.signal }
        );
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(
            payload.error ||
              payload.detail ||
              `Inventory request failed (${response.status})`
          );
        }

        setInventoryHistory(
          Array.isArray(payload.history) ? payload.history : []
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          setHistoryError(
            err.message || "Unable to load inventory history."
          );
        }
      } finally {
        if (!controller.signal.aborted) setHistoryLoading(false);
      }
    }

    loadInventoryHistory();
    return () => controller.abort();
  }, [phcId, medicineId]);

  async function runForecast(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_BASE}/api/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phc_id: phcId,
          medicine_id: medicineId,
          horizon_days: Number(horizonDays),
          crisis,
          current_stock: Number(currentStock),
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          payload.detail || payload.error || `Forecast failed (${response.status})`
        );
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unable to contact the PHC forecasting API.");
    } finally {
      setLoading(false);
    }
  }

  const summary = result?.summary ?? {};
  const projection = Array.isArray(result?.inventory_projection)
    ? result.inventory_projection
    : [];
  const firstStockoutIndex = projection.findIndex(
    (row) => row.Projected_Stockout_Flag
  );
  const coverageDays =
    firstStockoutIndex >= 0 ? firstStockoutIndex : projection.length;
  const coverageLabel =
    firstStockoutIndex >= 0
      ? `${coverageDays} ${coverageDays === 1 ? "day" : "days"}`
      : projection.length > 0
        ? `${projection.length}+ days`
        : "—";
  const medicineName =
    medicinesForPhc.find((medicine) => medicine.Medicine_ID === medicineId)
      ?.Medicine_Name || medicineId;
  const unit =
    projection.find((row) => row.Unit_Of_Measure)?.Unit_Of_Measure || "units";

  return (
    <main className="forecast-page">
      <section className="forecast-intro">
        <div>
          <p className="forecast-eyebrow">DEMAND PLANNING</p>
          <h2>Medicine demand forecast</h2>
          <p>
            Estimate future demand, monitor available stock, and identify when
            replenishment may be needed.
          </p>
        </div>
        <span className="forecast-intro-icon" aria-hidden="true">↗</span>
      </section>

      <section className="forecast-panel">
        <div className="forecast-section-title">
          <div>
            <h3>Forecast parameters</h3>
            <p>Choose a health centre, medicine, and planning scenario.</p>
          </div>
          <span className="forecast-tag">Model analysis</span>
        </div>

        <form className="forecast-form" onSubmit={runForecast}>
          <label className="forecast-field">
            <span>Primary health centre</span>
            <select
              value={phcId}
              onChange={(event) => {
                const nextPhc = event.target.value;
                setPhcId(nextPhc);
                setMedicineId("");
                setResult(null);
              }}
              required
            >
              <option value="">Select a health centre</option>
              {catalog.phcs.map((id) => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </label>

          <label className="forecast-field">
            <span>Medicine</span>
            <select
              value={medicineId}
              disabled={!phcId}
              onChange={(event) => {
                setMedicineId(event.target.value);
                setResult(null);
              }}
              required
            >
              <option value="">Select a medicine</option>
              {medicinesForPhc.map((medicine) => (
                <option key={medicine.Medicine_ID} value={medicine.Medicine_ID}>
                  {medicine.Medicine_Name}
                </option>
              ))}
            </select>
          </label>

          <label className="forecast-field">
            <span>Forecast horizon</span>
            <div className="forecast-input-suffix">
              <input
                type="number"
                min="1"
                max="365"
                value={horizonDays}
                placeholder="Enter days"
                onChange={(event) => setHorizonDays(event.target.value)}
                required
              />
              <span>days</span>
            </div>
          </label>

          <label className="forecast-field">
            <span>Current usable stock</span>
            <div className="forecast-input-suffix">
              <input
                type="number"
                min="0"
                value={currentStock}
                placeholder="Enter stock quantity"
                onChange={(event) => setCurrentStock(event.target.value)}
                required
              />
              <span>{unit}</span>
            </div>
          </label>

          <label className="forecast-crisis">
            <input
              type="checkbox"
              checked={crisis}
              onChange={(event) => setCrisis(event.target.checked)}
            />
            <span>
              <strong>Simulate declared crisis</strong>
              <small>Evaluate the selected crisis scenario.</small>
            </span>
          </label>

          <div className="forecast-form-actions">
            <button
              className="forecast-primary-button"
              type="submit"
              disabled={loading || !medicineId}
            >
              {loading ? "Calculating forecast…" : "Generate forecast"}
              {!loading && <span aria-hidden="true"> →</span>}
            </button>
            <span className="forecast-form-hint">
              Uses the existing PHC forecasting API.
            </span>
          </div>
        </form>

        {error && <div className="forecast-alert" role="alert">{error}</div>}
      </section>

      {result && (
        <section className="forecast-results" aria-live="polite">
          <div className="forecast-section-title">
            <div>
              <h3>Planning summary</h3>
              <p>{phcId} · {medicineName}</p>
            </div>
            <span className="forecast-tag forecast-tag-success">Forecast ready</span>
          </div>

          <div className="forecast-metrics">
            <article className="forecast-metric-card">
              <span className="forecast-metric-icon">▥</span>
              <p>Total forecast demand</p>
              <strong>
                {displayNumber(
                  summary.Requested_Horizon_Forecast_Demand_Total ??
                  summary.Forecast_Demand_Total
                )}
              </strong>
              <small>{unit} · requested horizon where available</small>
            </article>
            <article className="forecast-metric-card">
              <span className="forecast-metric-icon">◷</span>
              <p>Estimated stock coverage</p>
              <strong>{coverageLabel}</strong>
              <small>
                Based on the first projected stock-out in the returned forecast
              </small>
            </article>
            <article className="forecast-metric-card">
              <span className="forecast-metric-icon">!</span>
              <p>First projected stock-out</p>
              <strong className={firstStockoutIndex >= 0 ? "forecast-risk-text" : ""}>
                {summary.Requested_Horizon_First_Stockout_Date ??
                  summary.First_Projected_Stockout_Date ??
                  (firstStockoutIndex >= 0
                    ? projection[firstStockoutIndex]?.Target_Date
                    : "Not within horizon")}
              </strong>
              <small>
                {displayNumber(
                  summary.Requested_Horizon_Projected_Stockout_Days ??
                  summary.Projected_Stockout_Flag_Days,
                  0
                ) === "—"
                  ? "Based on the current forecast"
                  : `${displayNumber(summary.Requested_Horizon_Projected_Stockout_Days ?? summary.Projected_Stockout_Flag_Days, 0)} projected stock-out days`}
              </small>
            </article>
            <article className="forecast-metric-card">
              <span className="forecast-metric-icon">＋</span>
              <p>Recommended order quantity</p>
              <strong>
                {displayNumber(summary.Recommended_Order_Quantity, 0)}
              </strong>
              <small>
                {summary.Reorder_Recommended_Now === true
                  ? "Reorder recommended by the planning summary"
                  : summary.Reorder_Recommended_Now === false
                    ? "Reorder not recommended by the planning summary"
                    : "Use the model summary for ordering guidance"}
              </small>
            </article>
          </div>

          <div className="forecast-summary-details">
            <div>
              <span>Starting usable stock</span>
              <strong>{displayNumber(summary.Starting_Usable_Stock ?? currentStock, 0)} {unit}</strong>
            </div>
            <div>
              <span>Reorder level</span>
              <strong>{displayNumber(summary.Reorder_Level, 0)} {unit}</strong>
            </div>
            <div>
              <span>Planning horizon</span>
              <strong>{displayNumber(summary.User_Requested_Horizon_Days ?? horizonDays, 0)} days requested</strong>
            </div>
            <div>
              <span>Order-up-to horizon</span>
              <strong>{displayNumber(summary.Order_Up_To_Cover_Days ?? summary.Planning_Horizon_Used_For_Reorder, 0)} days</strong>
            </div>
          </div>

          {summary.Warning && (
            <div className="forecast-notice" role="note">
              <strong>Demo data notice</strong>
              <p>{summary.Warning}</p>
            </div>
          )}

          <section className="forecast-table-section">
            <div className="forecast-section-title">
              <div>
                <h3>Daily demand projection</h3>
                <p>Projected stock position for each forecast date.</p>
              </div>
              <span className="forecast-tag">{projection.length} days returned</span>
            </div>

            {projection.length === 0 ? (
              <p className="forecast-empty">The API returned no daily projection rows.</p>
            ) : (
              <div className="forecast-table-wrap">
                <table className="forecast-table">
                  <thead>
                    <tr>
                      <th>Target date</th>
                      <th>Forecast demand</th>
                      <th>Usable stock before demand</th>
                      <th>Projected stock-out</th>
                      <th>Closing stock</th>
                    </tr>
                  </thead>
                  <tbody>
                    {projection.map((row) => (
                      <tr
                        key={row.Target_Date}
                        className={row.Projected_Stockout_Flag ? "forecast-row-risk" : ""}
                      >
                        <td>{row.Target_Date}</td>
                        <td>{displayNumber(row.Predicted_Demand_Units)} {row.Unit_Of_Measure}</td>
                        <td>{displayNumber(row.Usable_Stock_Before_Demand, 0)}</td>
                        <td>
                          <span className={`forecast-status ${row.Projected_Stockout_Flag ? "forecast-status-risk" : "forecast-status-ok"}`}>
                            {row.Projected_Stockout_Flag ? "Stock-out" : "In stock"}
                          </span>
                        </td>
                        <td>{displayNumber(row.Projected_Closing_Stock, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </section>
      )}

      <section className="forecast-panel forecast-history-panel">
        <div className="forecast-section-title forecast-history-heading">
          <div>
            <h3>Historical inventory</h3>
            <p>Open this section when you want to inspect past stock records.</p>
          </div>
          <div className="forecast-history-actions">
            {showInventoryHistory && <span className="forecast-tag">{inventoryHistory.length} records</span>}
            <button
              className="forecast-secondary-button"
              type="button"
              aria-expanded={showInventoryHistory}
              onClick={() => setShowInventoryHistory((visible) => !visible)}
            >
              {showInventoryHistory ? "Hide history" : "View history"}
              <span aria-hidden="true">{showInventoryHistory ? " ↑" : " ↓"}</span>
            </button>
          </div>
        </div>

        {showInventoryHistory && <>
        {historyLoading && <p className="forecast-empty">Loading inventory history…</p>}
        {historyError && (
          <div className="forecast-alert" role="alert">
            <strong>Unable to load inventory history.</strong> {historyError}
            <p>Check that the Express backend is running on port 3000.</p>
          </div>
        )}
        {!historyLoading && !historyError && inventoryHistory.length === 0 && (
          <p className="forecast-empty">No inventory history was found for this selection.</p>
        )}

        {!historyLoading && !historyError && inventoryHistory.length > 0 && (
          <>
            <p className="forecast-table-caption">
              Showing <strong>{inventoryHistory.length}</strong> records for{" "}
              <strong>{phcId}</strong> / <strong>{medicineId}</strong>.
            </p>
            <div className="forecast-table-wrap">
              <table className="forecast-table">
                <thead>
                  <tr>
                    {[
                      "Date",
                      "Opening stock",
                      "Available stock",
                      "Demand",
                      "Dispensed",
                      "Unmet demand",
                      "Closing stock",
                      "Reorder level",
                      "Stock-out",
                    ].map((heading) => <th key={heading}>{heading}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {inventoryHistory.map((row) => (
                    <tr key={`${row.inventory_date}-${row.phc_id}-${row.medicine_id}`}>
                      <td>{String(row.inventory_date).slice(0, 10)}</td>
                      <td>{displayNumber(row.opening_stock, 0)}</td>
                      <td>{displayNumber(row.available_stock, 0)}</td>
                      <td>{displayNumber(row.total_demand, 0)}</td>
                      <td>{displayNumber(row.dispensed_quantity, 0)}</td>
                      <td>{displayNumber(row.unmet_demand, 0)}</td>
                      <td>{displayNumber(row.closing_stock, 0)}</td>
                      <td>{displayNumber(row.reorder_level, 0)}</td>
                      <td>
                        <span className={`forecast-status ${row.stockout_flag ? "forecast-status-risk" : "forecast-status-ok"}`}>
                          {row.stockout_flag ? "Yes" : "No"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        </>}
      </section>

      <p className="forecast-disclaimer">
        <strong>Demo environment:</strong> The bundled history is synthetic and
        the models are not validated for real medicine procurement decisions.
        Forecast outputs are estimates, not clinical or procurement instructions.
      </p>
    </main>
  );
}
