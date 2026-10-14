import { useEffect, useState } from "react";

// Set VITE_PHC_API_URL if your API is hosted somewhere else.
const API_BASE = import.meta.env.VITE_PHC_API_URL || "http://127.0.0.1:8000";

export default function PHCForecastExample() {
  const [phcId, setPhcId] = useState("PHC-DEMO-001");
  const [medicineId, setMedicineId] = useState("MED001");
  const [horizonDays, setHorizonDays] = useState(14);
  const [currentStock, setCurrentStock] = useState(100);
  const [crisis, setCrisis] = useState(false);
  const [catalog, setCatalog] = useState({ phcs: [], medicines: [] });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/catalog`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Catalog request failed (${response.status})`);
        return response.json();
      })
      .then(setCatalog)
      .catch((e) => setError(e.message));
  }, []);

  const medicinesForPhc = catalog.medicines.filter((m) => m.PHC_ID === phcId);

  async function runForecast(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/forecast`, {
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
      if (!response.ok) throw new Error(payload.detail || `Forecast failed (${response.status})`);
      setResult(payload);
    } catch (e) {
      setError(e.message || "Unable to contact the PHC ML API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 1000, margin: "2rem auto", padding: "1rem", fontFamily: "system-ui, sans-serif" }}>
      <h1>PHC Medicine Demand Forecast</h1>
      <p style={{ color: "#9a6700" }}>
        Demo only: bundled PHC history is synthetic and not validated for real procurement decisions.
      </p>
      <form onSubmit={runForecast} style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))" }}>
        <label>
          PHC
          <select value={phcId} onChange={(e) => {
            const nextPhc = e.target.value;
            setPhcId(nextPhc);
            const first = catalog.medicines.find((m) => m.PHC_ID === nextPhc);
            if (first) setMedicineId(first.Medicine_ID);
          }} style={{ display: "block", width: "100%", padding: ".6rem" }}>
            {catalog.phcs.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <label>
          Medicine
          <select value={medicineId} onChange={(e) => setMedicineId(e.target.value)} style={{ display: "block", width: "100%", padding: ".6rem" }}>
            {medicinesForPhc.map((m) => <option key={m.Medicine_ID} value={m.Medicine_ID}>{m.Medicine_Name}</option>)}
          </select>
        </label>
        <label>
          Forecast horizon (days)
          <input type="number" min="1" max="365" value={horizonDays} onChange={(e) => setHorizonDays(e.target.value)} style={{ display: "block", width: "100%", padding: ".6rem" }} required />
        </label>
        <label>
          Current usable stock
          <input type="number" min="0" value={currentStock} onChange={(e) => setCurrentStock(e.target.value)} style={{ display: "block", width: "100%", padding: ".6rem" }} required />
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: ".5rem" }}>
          <input type="checkbox" checked={crisis} onChange={(e) => setCrisis(e.target.checked)} />
          Simulate declared crisis
        </label>
        <button disabled={loading || catalog.medicines.length === 0} style={{ padding: ".75rem", alignSelf: "end" }}>
          {loading ? "Calculating…" : "Generate forecast"}
        </button>
      </form>
      {error && <p role="alert" style={{ color: "#b42318" }}>{error}</p>}
      {result && (
        <section style={{ marginTop: "2rem" }}>
          <h2>Planning summary</h2>
          <pre style={{ whiteSpace: "pre-wrap", background: "#f6f8fa", padding: "1rem", borderRadius: 8 }}>{JSON.stringify(result.summary, null, 2)}</pre>
          <h2>Demand forecast</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr>{["Target date", "Forecast demand", "Usable stock before demand", "Projected stock-out", "Closing stock"].map((h) => <th key={h} style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: ".5rem" }}>{h}</th>)}</tr></thead>
              <tbody>
                {result.inventory_projection.map((row) => <tr key={row.Target_Date}>
                  <td style={{ padding: ".5rem", borderBottom: "1px solid #eee" }}>{row.Target_Date}</td>
                  <td style={{ padding: ".5rem", borderBottom: "1px solid #eee" }}>{row.Predicted_Demand_Units} {row.Unit_Of_Measure}</td>
                  <td style={{ padding: ".5rem", borderBottom: "1px solid #eee" }}>{row.Usable_Stock_Before_Demand}</td>
                  <td style={{ padding: ".5rem", borderBottom: "1px solid #eee" }}>{row.Projected_Stockout_Flag ? "Yes" : "No"}</td>
                  <td style={{ padding: ".5rem", borderBottom: "1px solid #eee" }}>{row.Projected_Closing_Stock}</td>
                </tr>)}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
