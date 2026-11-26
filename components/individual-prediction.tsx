"use client"

import { FormEvent, useState } from "react"
import { Check, ChevronDown, Info, ShieldCheck, Sparkles } from "lucide-react"
import { api, type CustomerFeatures } from "@/lib/api"

const defaults: CustomerFeatures = {
  Tenure: 2, WarehouseToHome: 15, NumberOfDeviceRegistered: 2,
  PreferedOrderCat: "Mobile Phone", SatisfactionScore: 2, MaritalStatus: "Single",
  NumberOfAddress: 1, Complain: 1, DaySinceLastOrder: 35, CashbackAmount: 120,
}
type Result = Awaited<ReturnType<typeof api.predictIndividual>>

export function IndividualPrediction() {
  const [values, setValues] = useState<CustomerFeatures>(defaults)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const update = (key: keyof CustomerFeatures, value: string) => {
    const numeric = ["Tenure", "WarehouseToHome", "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"].includes(key)
    setValues(current => ({ ...current, [key]: numeric ? (value === "" ? null : Number(value)) : value }))
  }
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setLoading(true)
    try { setResult(await api.predictIndividual(values)) } catch (err) { setError(err instanceof Error ? err.message : "Prediksi gagal diproses.") } finally { setLoading(false) }
  }
  const probability = result ? Math.round(result.probability * 100) : null
  const factors = result ? [...result.explanation.drivers, ...result.explanation.protectors] : []

  return <>
    <div className="page-heading prediction-heading"><div><div className="eyebrow page-kicker">CASE REVIEW <span className="kicker-line" /></div><h1>Prediksi Risiko Pelanggan</h1><p>Analisis profil pelanggan, pahami penyebab risikonya, dan tentukan tindakan retensi.</p></div><div className="model-chip"><span className="status-dot" />XGBoost <span className="status-separator">·</span> Best model</div></div>
    <div className="individual-layout">
      <form className="profile-form" onSubmit={submit}>
        <div className="form-header"><div><div className="eyebrow">PROFIL PELANGGAN</div><h2>Customer signals</h2></div><span className="form-fields-count">10 INPUTS</span></div>
        <fieldset className="signal-group"><legend>Behavioral Signals</legend><p>Pola aktivitas pelanggan</p><div className="signal-grid">
          <NumberField label="Tenure" suffix="bulan" value={values.Tenure} onChange={v => update("Tenure", v)} />
          <NumberField label="Day Since Last Order" suffix="hari" value={values.DaySinceLastOrder} onChange={v => update("DaySinceLastOrder", v)} />
          <NumberField label="Number Of Device Registered" suffix="perangkat" value={values.NumberOfDeviceRegistered} onChange={v => update("NumberOfDeviceRegistered", v)} />
          <NumberField label="Number Of Address" suffix="alamat" value={values.NumberOfAddress} onChange={v => update("NumberOfAddress", v)} />
        </div></fieldset>
        <fieldset className="signal-group"><legend>Experience Signals</legend><p>Pengalaman dan nilai pesanan</p><div className="signal-grid">
          <NumberField label="Satisfaction Score" suffix="/ 5" value={values.SatisfactionScore} onChange={v => update("SatisfactionScore", v)} />
          <SelectField label="Complain" value={String(values.Complain)} values={[["1", "Ya"], ["0", "Tidak"]]} onChange={v => update("Complain", v)} />
          <NumberField label="Warehouse To Home" suffix="km" value={values.WarehouseToHome} onChange={v => update("WarehouseToHome", v)} />
          <NumberField label="Cashback Amount" suffix="IDR" value={values.CashbackAmount} onChange={v => update("CashbackAmount", v)} />
        </div></fieldset>
        <fieldset className="signal-group"><legend>Customer Preferences</legend><p>Preferensi dan konteks pelanggan</p><div className="signal-grid">
          <SelectField label="Preferred Order Category" value={values.PreferedOrderCat} values={["Mobile Phone", "Fashion", "Grocery", "Laptop & Accessory", "Others"]} onChange={v => update("PreferedOrderCat", v)} />
          <SelectField label="Marital Status" value={values.MaritalStatus} values={["Single", "Married", "Divorced"]} onChange={v => update("MaritalStatus", v)} />
        </div></fieldset>
        <div className="form-footer"><span><Info size={13} /> Diproses oleh pipeline model terlatih</span><button type="submit" className="primary-action" disabled={loading}><Sparkles size={14} />{loading ? "Menganalisis..." : "Analyze Customer Risk"}</button></div>
        {error && <p className="upload-error" role="alert">{error}</p>}
      </form>
      <section className="result-panel" aria-labelledby="prediction-result-title"><div className="result-topline"><div className="eyebrow">{result ? "HASIL PREDIKSI" : "MENUNGGU ANALISIS"}</div><span className="model-version"><span className="status-dot" />XGBOOST PIPELINE</span></div>
        {result && probability !== null ? <><div className="result-summary"><div className="probability-block"><span className="probability-label">Probabilitas churn</span><div className="probability-value">{probability}<span>%</span></div><div className="probability-meter"><span style={{ width: `${probability}%`, background: result.risk.color }} /></div><span className="probability-caption">Ambang risiko tinggi <b>≥ 65%</b></span></div><div className="result-status"><span className={`risk-badge ${result.risk.tier === "High Risk" ? "badge-high" : result.risk.tier === "Medium Risk" ? "badge-medium" : "badge-low"}`}><i />{result.risk.tier}</span><h2 id="prediction-result-title">{result.prediction ? "Berisiko Churn" : "Cenderung Bertahan"}</h2><p>{result.risk.indication}</p></div></div>
          <div className="recommendation"><div className="recommendation-icon"><ShieldCheck size={17} /></div><div className="recommendation-copy"><span className="eyebrow">TINDAKAN YANG DISARANKAN</span><h3>{result.risk.tier === "High Risk" ? "Service Recovery" : result.risk.tier === "Medium Risk" ? "Re-Engagement" : "Loyalty Reinforcement"}</h3><p>{result.risk.action}</p></div></div>
          <div className="shap-section"><div className="shap-heading"><div><div className="eyebrow">PENJELASAN MODEL <span className="shap-pill">SHAP</span></div><h3>Mengapa pelanggan ini berisiko?</h3></div></div><p className="shap-subtitle">Kontribusi fitur aktual pada prediksi pelanggan ini.</p><div className="shap-rows">{factors.map((factor: any) => <div className="shap-row" key={factor.feature}><div className="shap-factor"><strong>{factor.feature}</strong><span>{factor.importance > 0 ? "Meningkatkan risiko" : "Menurunkan risiko"}</span></div><div className="shap-plot"><div className="shap-track"><i className={factor.importance > 0 ? "shap-positive" : "shap-negative"} style={{ width: `${Math.min(Math.abs(factor.importance) * 18, 50)}%` }} /></div></div><span className={`shap-value ${factor.importance > 0 ? "increase" : "reduce"}`}>{factor.importance > 0 ? "+" : ""}{factor.importance.toFixed(3)}</span></div>)}</div></div>
          <div className="analysis-footnote analysis-complete"><Check size={14} /><span>Prediksi dan penjelasan SHAP dihitung dari artefak model repository.</span></div></> : <div className="empty-state">Isi profil pelanggan lalu jalankan analisis untuk melihat probabilitas, tier risiko, dan penjelasan SHAP.</div>}
      </section>
    </div>
  </>
}

function NumberField({ label, suffix, value, onChange }: { label: string; suffix: string; value: number | null; onChange: (value: string) => void }) {
  return <label className="signal-field"><span>{label}</span><div className="signal-control"><input type="number" min="0" value={value ?? ""} onChange={event => onChange(event.target.value)} />{suffix && <span className="field-suffix">{suffix}</span>}</div></label>
}
function SelectField({ label, value, values, onChange }: { label: string; value: string; values: (string | [string, string])[]; onChange: (value: string) => void }) {
  return <label className="signal-field"><span>{label}</span><div className="signal-control"><select value={value} onChange={event => onChange(event.target.value)}>{values.map(item => Array.isArray(item) ? <option key={item[0]} value={item[0]}>{item[1]}</option> : <option key={item}>{item}</option>)}</select><ChevronDown className="select-chevron" size={13} /></div></label>
}
