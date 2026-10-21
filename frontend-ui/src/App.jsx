
import { useEffect, useState } from "react";
import PHCForecastExample from "./PHCForecastExample";
import "./App.css";
import "./inventory-ui.css";


const API_BASE =
  import.meta.env.VITE_PHC_API_URL || "http://127.0.0.1:8000";

const BACKEND_BASE =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

const navigation = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "forecast", label: "Forecast", icon: "↗" },
  { id: "risk", label: "Stock-out Risk", icon: "⚕" },
  { id: "inventory", label: "Inventory", icon: "▤" },
];

function MetricCard({ icon, label, value, note }) {
  return (
    <article className="metric-card">
      <div className="metric-card-top">
        <span className="metric-icon" aria-hidden="true">{icon}</span>
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

function Dashboard({ catalog, history, loading, error, onNavigate }) {
  const medicineCount = catalog?.medicines?.length ?? 0;
  const phcCount = catalog?.phcs?.length ?? 0;
  const historyCount = history.length;
  const latest = history[history.length - 1];

  const stockoutDays = history.filter(
    (row) => row.stockout_flag === true
  ).length;

  return (
    <>
      <div className="welcome-panel">
        <div>
          <div className="eyebrow">HEALTHCARE INVENTORY MANAGEMENT</div>
          <h2>Welcome to your inventory workspace</h2>
          <p>
            Monitor medicine demand, review historical stock, and explore
            demand forecasts and stock-out risk in one place.
          </p>
        </div>
        <div className="welcome-symbol" aria-hidden="true">✚</div>
      </div>

      {error && <div className="notice notice-error" role="alert">{error}</div>}

      <div className="section-heading">
        <div>
          <h2>Inventory overview</h2>
          <p>Live catalogue data and the latest loaded inventory history.</p>
        </div>
        <span className="live-label">
          <span className="status-dot" />
          API data
        </span>
      </div>

      <div className="metrics-grid">
        <MetricCard
          icon="▣"
          label="Medicines in catalogue"
          value={loading ? "…" : medicineCount}
          note="From the Python catalogue API"
        />
        <MetricCard
          icon="⌂"
          label="PHCs in catalogue"
          value={loading ? "…" : phcCount}
          note="Available health-centre entries"
        />
        <MetricCard
          icon="▤"
          label="Loaded history records"
          value={loading ? "…" : historyCount}
          note="For the first available PHC and medicine"
        />
        <MetricCard
          icon="◷"
          label="Latest available stock"
          value={latest ? latest.available_stock ?? "—" : "—"}
          note={
            latest
              ? `Inventory date: ${String(latest.inventory_date).slice(0, 10)}`
              : "Waiting for inventory data"
          }
        />
      </div>

      <div className="dashboard-grid">
        <section className="content-card">
          <div className="card-heading">
            <div>
              <h3>Inventory snapshot</h3>
              <p>Based on the currently loaded historical records.</p>
            </div>
            <span className="soft-badge">Historical</span>
          </div>

          {loading ? (
            <p className="empty-state">Loading inventory data…</p>
          ) : latest ? (
            <div className="snapshot-list">
              <div className="snapshot-row">
                <span>Latest available stock</span>
                <strong>{latest.available_stock ?? "—"}</strong>
              </div>
              <div className="snapshot-row">
                <span>Latest recorded demand</span>
                <strong>{latest.total_demand ?? "—"}</strong>
              </div>
              <div className="snapshot-row">
                <span>Latest closing stock</span>
                <strong>{latest.closing_stock ?? "—"}</strong>
              </div>
              <div className="snapshot-row">
                <span>Stock-out days in loaded history</span>
                <strong>{stockoutDays}</strong>
              </div>
            </div>
          ) : (
            <p className="empty-state">
              No inventory history is available yet.
            </p>
          )}

          <p className="small-disclaimer">
            These figures describe the loaded historical records, not the
            entire PHC network.
          </p>
        </section>

        <section className="content-card">
          <div className="card-heading">
            <div>
              <h3>Quick actions</h3>
              <p>Go directly to a project feature.</p>
            </div>
          </div>

          <div className="quick-actions">
            <button className="quick-action" onClick={() => onNavigate("forecast")}>
              <span className="quick-action-icon">↗</span>
              <span>
                <strong>Generate forecast</strong>
                <small>Estimate future medicine demand</small>
              </span>
              <span className="action-arrow">→</span>
            </button>

            <button className="quick-action" onClick={() => onNavigate("risk")}>
              <span className="quick-action-icon">⚕</span>
              <span>
                <strong>Check stock-out risk</strong>
                <small>Open the risk analysis section</small>
              </span>
              <span className="action-arrow">→</span>
            </button>

            <button className="quick-action" onClick={() => onNavigate("inventory")}>
              <span className="quick-action-icon">▤</span>
              <span>
                <strong>View inventory</strong>
                <small>Inspect historical stock records</small>
              </span>
              <span className="action-arrow">→</span>
            </button>
          </div>
        </section>
      </div>

      <div className="notice notice-info">
        <strong>Demo environment:</strong> The bundled data is synthetic and
        the models are not validated for real medicine procurement decisions.
      </div>
    </>
  );
}


function RiskAnalysis({ catalog }) {
  const [phcId, setPhcId] = useState("");
  const [medicineId, setMedicineId] = useState("");
  const [stock, setStock] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const phcs = Array.isArray(catalog?.phcs) ? catalog.phcs : [];
  const medicines = Array.isArray(catalog?.medicines)
    ? catalog.medicines.filter((medicine) => !phcId || medicine.PHC_ID === phcId)
    : [];

  function handlePhcChange(value) {
    setPhcId(value);
    setMedicineId("");
    setResult(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!phcId || !medicineId || stock.trim() === "") {
      setError("Select a health centre and medicine, then enter current usable stock.");
      return;
    }

    const stockValue = Number(stock);
    if (!Number.isFinite(stockValue) || stockValue < 0) {
      setError("Current usable stock must be a non-negative number.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_BASE}/api/stockout-risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phc_id: phcId,
          medicine_id: medicineId,
          current_usable_stock: stockValue
        })
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail || payload.error || `Risk request failed (${response.status})`
        );
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unable to calculate stock-out risk. Check that both APIs are running.");
    } finally {
      setLoading(false);
    }
  }

  const score =
    result?.uncalibrated_stockout_score ??
    result?.stockout_score ??
    result?.score ??
    null;
  const threshold =
    result?.validation_selected_threshold ??
    result?.threshold_used ??
    result?.threshold ??
    null;
  const flag =
    result?.flag_at_threshold ??
    result?.flag ??
    null;
  const scoreDisplay =
    typeof score === "number"
      ? score.toLocaleString(undefined, { maximumFractionDigits: 6 })
      : score ?? "Not provided";
  const thresholdDisplay =
    typeof threshold === "number"
      ? threshold.toLocaleString(undefined, { maximumFractionDigits: 4 })
      : threshold ?? "Not provided";
  const medicineName =
    catalog?.medicines?.find((medicine) => medicine.Medicine_ID === medicineId)?.Medicine_Name
    ?? medicineId;

  return (
    <div className="risk-page">
      <section className="risk-hero">
        <div>
          <div className="eyebrow">INVENTORY RISK MONITORING</div>
          <h2>Stock-out risk assessment</h2>
          <p>Assess the model's stock-out risk output for a selected medicine and current stock quantity.</p>
        </div>
        <div className="risk-hero-icon" aria-hidden="true">⚕</div>
      </section>

      <section className="content-card risk-form-card">
        <div className="card-heading">
          <div>
            <h3>Risk assessment parameters</h3>
            <p>Choose a health centre and medicine, then enter the current usable stock.</p>
          </div>
          <span className="soft-badge">Model analysis</span>
        </div>

        <form className="risk-form" onSubmit={handleSubmit}>
          <label className="risk-field">
            <span>Primary health centre</span>
            <select value={phcId} onChange={(event) => handlePhcChange(event.target.value)} required>
              <option value="">Select a health centre</option>
              {phcs.map((phc) => <option key={phc} value={phc}>{phc}</option>)}
            </select>
          </label>

          <label className="risk-field">
            <span>Medicine</span>
            <select
              value={medicineId}
              onChange={(event) => { setMedicineId(event.target.value); setResult(null); }}
              disabled={!phcId}
              required
            >
              <option value="">Select a medicine</option>
              {medicines.map((medicine) => (
                <option key={medicine.Medicine_ID} value={medicine.Medicine_ID}>
                  {medicine.Medicine_Name || medicine.Medicine_ID}
                </option>
              ))}
            </select>
          </label>

          <label className="risk-field">
            <span>Current usable stock</span>
            <div className="risk-input-with-unit">
              <input
                type="number"
                min="0"
                step="any"
                value={stock}
                onChange={(event) => { setStock(event.target.value); setResult(null); }}
                placeholder="Enter stock quantity"
                required
              />
              <span>units</span>
            </div>
          </label>

          <div className="risk-submit-row">
            <button className="risk-submit-button" type="submit" disabled={loading}>
              {loading ? "Checking risk…" : "Check stock-out risk →"}
            </button>
            <span className="risk-submit-hint">Uses the existing stock-out risk API.</span>
          </div>
        </form>

        {error && <div className="notice notice-error" role="alert">{error}</div>}
      </section>

      {result && (
        <section className="content-card risk-results-card" aria-live="polite">
          <div className="card-heading">
            <div>
              <h3>Risk assessment result</h3>
              <p>{phcId} · {medicineName}</p>
            </div>
            <span className="soft-badge">Prediction saved if API confirms success</span>
          </div>

          <div className="risk-metrics-grid">
            <article className="risk-metric">
              <span className="risk-metric-label">Model score</span>
              <strong>{scoreDisplay}</strong>
              <small>Uncalibrated model output</small>
            </article>
            <article className="risk-metric">
              <span className="risk-metric-label">Decision threshold</span>
              <strong>{thresholdDisplay}</strong>
              <small>Threshold returned by the model</small>
            </article>
            <article className={`risk-metric ${flag === true ? "risk-metric-alert" : flag === false ? "risk-metric-ok" : ""}`}>
              <span className="risk-metric-label">Flag at threshold</span>
              <strong>{flag === true ? "Flagged" : flag === false ? "Not flagged" : "Not provided"}</strong>
              <small>{flag === true ? "Model flag is true" : flag === false ? "Model flag is false" : "No flag was returned"}</small>
            </article>
          </div>

          {result.data_note && <div className="notice notice-info">{result.data_note}</div>}
          {result.warning && <div className="notice notice-info">{result.warning}</div>}
          <p className="small-disclaimer">
            Demo only: the model score is not a calibrated probability and has not been validated for real PHC decisions.
            Use it for demonstration, not clinical or procurement decisions.
          </p>
        </section>
      )}

      {!result && !error && (
        <div className="notice notice-info risk-disclaimer">
          The risk result will appear here after you submit the form. Inputs are intentionally blank until you enter them.
          This synthetic demo is not validated for real medicine procurement decisions.
        </div>
      )}
    </div>
  );
}


function InventoryPage({ catalog }) {
  const [phcId, setPhcId] = useState("");
  const [medicineId, setMedicineId] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [stockoutFilter, setStockoutFilter] = useState("all");

  const phcs = Array.isArray(catalog?.phcs) ? catalog.phcs : [];
  const medicines = Array.isArray(catalog?.medicines)
    ? catalog.medicines.filter((medicine) => !phcId || !medicine.PHC_ID || medicine.PHC_ID === phcId)
    : [];

  useEffect(() => {
    if (!phcId && phcs.length > 0) setPhcId(phcs[0]);
  }, [phcs, phcId]);

  useEffect(() => {
    if (!phcId) {
      setMedicineId("");
      setHistory([]);
      return;
    }
    const firstMedicine = medicines[0];
    if (firstMedicine && !medicines.some((medicine) => medicine.Medicine_ID === medicineId)) {
      setMedicineId(firstMedicine.Medicine_ID);
    }
  }, [phcId, medicines, medicineId]);

  useEffect(() => {
    if (!phcId || !medicineId) {
      setHistory([]);
      return;
    }

    const controller = new AbortController();
    async function loadHistory() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({ phc_id: phcId, medicine_id: medicineId });
        const response = await fetch(
          `${BACKEND_BASE}/api/inventory/history?${params.toString()}`,
          { signal: controller.signal }
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || payload.detail || `Inventory request failed (${response.status})`);
        }
        const rows = Array.isArray(payload.history)
          ? payload.history
          : Array.isArray(payload.inventory)
            ? payload.inventory
            : Array.isArray(payload)
              ? payload
              : [];
        setHistory(rows);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Unable to load inventory history. Check that the backend and database are running.");
          setHistory([]);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    loadHistory();
    return () => controller.abort();
  }, [phcId, medicineId]);

  const filteredHistory = history.filter((row) => {
    const date = String(row.inventory_date ?? row.date ?? "").slice(0, 10);
    const matchesSearch = !search || date.toLowerCase().includes(search.toLowerCase());
    const isStockout = row.stockout_flag === true || row.stockout_flag === "true" || row.stockout === true;
    const matchesStockout = stockoutFilter === "all" ||
      (stockoutFilter === "stockout" && isStockout) ||
      (stockoutFilter === "available" && !isStockout);
    return matchesSearch && matchesStockout;
  });

  const latest = history.length ? history[history.length - 1] : null;
  const stockoutDays = history.filter((row) =>
    row.stockout_flag === true || row.stockout_flag === "true" || row.stockout === true
  ).length;
  const medicineName = catalog?.medicines?.find(
    (medicine) => medicine.Medicine_ID === medicineId
  )?.Medicine_Name || medicineId;

  function value(row, ...keys) {
    for (const key of keys) {
      if (row?.[key] !== undefined && row?.[key] !== null) return row[key];
    }
    return "—";
  }

  return (
    <div className="inventory-page">
      <section className="inventory-hero">
        <div>
          <div className="eyebrow">STOCK VISIBILITY</div>
          <h2>Historical inventory</h2>
          <p>Inspect recorded stock, demand, dispensing, and stock-out history for a selected health centre and medicine.</p>
        </div>
        <div className="inventory-hero-icon" aria-hidden="true">▤</div>
      </section>

      <section className="content-card inventory-filter-card">
        <div className="card-heading">
          <div>
            <h3>Inventory filters</h3>
            <p>Select the PHC and medicine whose historical records you want to inspect.</p>
          </div>
          <span className="soft-badge">PostgreSQL data</span>
        </div>

        <div className="inventory-filter-grid">
          <label className="risk-field">
            <span>Primary health centre</span>
            <select value={phcId} onChange={(event) => { setPhcId(event.target.value); setMedicineId(""); }}>
              <option value="">Select a health centre</option>
              {phcs.map((phc) => <option key={phc} value={phc}>{phc}</option>)}
            </select>
          </label>

          <label className="risk-field">
            <span>Medicine</span>
            <select
              value={medicineId}
              onChange={(event) => setMedicineId(event.target.value)}
              disabled={!phcId || medicines.length === 0}
            >
              <option value="">Select a medicine</option>
              {medicines.map((medicine) => (
                <option key={medicine.Medicine_ID} value={medicine.Medicine_ID}>
                  {medicine.Medicine_Name || medicine.Medicine_ID}
                </option>
              ))}
            </select>
          </label>
        </div>

        {error && <div className="notice notice-error" role="alert">{error}</div>}
      </section>

      <section className="inventory-summary-grid">
        <MetricCard
          icon="▤"
          label="History records"
          value={loading ? "…" : history.length}
          note="Records returned by the inventory API"
        />
        <MetricCard
          icon="◷"
          label="Latest available stock"
          value={latest ? value(latest, "available_stock", "Available_Stock") : "—"}
          note={latest ? `Date: ${String(value(latest, "inventory_date", "Inventory_Date", "date")).slice(0, 10)}` : "No record loaded"}
        />
        <MetricCard
          icon="↘"
          label="Latest recorded demand"
          value={latest ? value(latest, "total_demand", "demand", "Total_Demand") : "—"}
          note="From the selected historical series"
        />
        <MetricCard
          icon="!"
          label="Stock-out days"
          value={loading ? "…" : stockoutDays}
          note="Within the loaded history"
        />
      </section>

      <section className="content-card inventory-table-card">
        <div className="card-heading inventory-table-heading">
          <div>
            <h3>Inventory records</h3>
            <p>{phcId || "No PHC selected"} · {medicineName || "No medicine selected"}</p>
          </div>
          <span className="soft-badge">{filteredHistory.length} shown</span>
        </div>

        <div className="inventory-table-controls">
          <label className="inventory-search">
            <span className="sr-only">Search by date</span>
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by date (YYYY-MM-DD)"
            />
          </label>
          <label className="inventory-stockout-filter">
            <span className="sr-only">Filter stock-out status</span>
            <select value={stockoutFilter} onChange={(event) => setStockoutFilter(event.target.value)}>
              <option value="all">All stock status</option>
              <option value="stockout">Stock-out only</option>
              <option value="available">No stock-out</option>
            </select>
          </label>
        </div>

        {loading ? (
          <p className="empty-state">Loading inventory history…</p>
        ) : !error && filteredHistory.length === 0 ? (
          <p className="empty-state">No inventory records match the selected PHC, medicine, and filters.</p>
        ) : (
          <div className="inventory-table-wrap">
            <table className="inventory-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opening stock</th>
                  <th>Available stock</th>
                  <th>Demand</th>
                  <th>Dispensed</th>
                  <th>Unmet demand</th>
                  <th>Closing stock</th>
                  <th>Reorder level</th>
                  <th>Stock-out</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.map((row, index) => {
                  const isStockout = row.stockout_flag === true || row.stockout_flag === "true" || row.stockout === true;
                  return (
                    <tr key={`${value(row, "inventory_date", "Inventory_Date", "date")}-${index}`}>
                      <td>{String(value(row, "inventory_date", "Inventory_Date", "date")).slice(0, 10)}</td>
                      <td>{value(row, "opening_stock", "Opening_Stock")}</td>
                      <td>{value(row, "available_stock", "Available_Stock")}</td>
                      <td>{value(row, "total_demand", "demand", "Total_Demand")}</td>
                      <td>{value(row, "dispensed", "quantity_dispensed", "Dispensed")}</td>
                      <td>{value(row, "unmet_demand", "Unmet_Demand")}</td>
                      <td>{value(row, "closing_stock", "Closing_Stock")}</td>
                      <td>{value(row, "reorder_level", "Reorder_Level")}</td>
                      <td>
                        <span className={`inventory-status ${isStockout ? "inventory-status-risk" : "inventory-status-ok"}`}>
                          {isStockout ? "Yes" : "No"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <p className="small-disclaimer">
          Historical records describe the selected PHC and medicine only. The bundled dataset is synthetic and is not validated for real procurement decisions.
        </p>
      </section>
    </div>
  );
}

function ComingSoon({ title, description }) {
  return (
    <section className="content-card coming-soon">
      <div className="coming-soon-icon">✚</div>
      <h2>{title}</h2>
      <p>{description}</p>
      <span className="soft-badge">Implementation pending</span>
    </section>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [catalog, setCatalog] = useState({ phcs: [], medicines: [] });
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboardData() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(`${API_BASE}/catalog`, {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Catalogue request failed (${response.status})`);
        }

        const data = await response.json();
        setCatalog(data);

        const firstPhc = data.phcs?.[0];
        const firstMedicine = data.medicines?.find(
          (medicine) => medicine.PHC_ID === firstPhc
        );

        if (!firstPhc || !firstMedicine) {
          setHistory([]);
          return;
        }

        const params = new URLSearchParams({
          phc_id: firstPhc,
          medicine_id: firstMedicine.Medicine_ID,
        });

        const historyResponse = await fetch(
          `${BACKEND_BASE}/api/inventory/history?${params.toString()}`,
          { signal: controller.signal }
        );

        const historyPayload = await historyResponse.json();

        if (!historyResponse.ok) {
          throw new Error(
            historyPayload.error ||
              historyPayload.detail ||
              `Inventory request failed (${historyResponse.status})`
          );
        }

        setHistory(
          Array.isArray(historyPayload.history) ? historyPayload.history : []
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Unable to load dashboard data.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadDashboardData();
    return () => controller.abort();
  }, []);

  const pageTitle =
    navigation.find((item) => item.id === activePage)?.label ?? "Dashboard";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">+</div>
          <div>
            <div className="brand-name">PHC Inventory</div>
            <div className="brand-caption">INTELLIGENCE</div>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${activePage === item.id ? "active" : ""}`}
              aria-current={activePage === item.id ? "page" : undefined}
              onClick={() => setActivePage(item.id)}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-dot" />
          <div>
            <div className="footer-title">Demo environment</div>
            <div className="footer-caption">Synthetic data</div>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <div className="breadcrumb">PHC / Inventory Intelligence</div>
            <h1>{pageTitle}</h1>
          </div>
          <div className="environment-badge">
            <span className="status-dot" />
            Demo system
          </div>
        </header>

        <section className="page-content">
          {activePage === "dashboard" && (
            <Dashboard
              catalog={catalog}
              history={history}
              loading={loading}
              error={error}
              onNavigate={setActivePage}
            />
          )}

          {activePage === "forecast" && (
            <div className="legacy-app">
              <PHCForecastExample />
            </div>
          )}

          {activePage === "risk" && <RiskAnalysis catalog={catalog} />}

          {activePage === "inventory" && <InventoryPage catalog={catalog} />}
        </section>
      </main>
    </div>
  );
}
