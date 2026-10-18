import {
  Activity,
  ArrowLeft,
  Clock3,
  Database,
  HeartPulse,
  RefreshCw,
  ShieldAlert,
  Stethoscope,
  X,
  LogOut,
} from "lucide-react";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  useNavigate,
  useParams,
} from "react-router-dom";

import {
  dischargePatient,
  getLatestRisk,
  getPatient,
  getRiskExplanation,
  getRiskExplanationHistory,
  getRiskHistory,
  type Patient,
  type RiskEvent,
  type RiskExplanation,
} from "../api/client";

import { useAuth } from "../auth/AuthContext";
import { ThemeToggle, useTheme } from "../theme/ThemeContext";

function percentage(
  value: number | null | undefined,
) {
  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  return `${(
    value * 100
  ).toFixed(1)}%`;
}

const IST_TIME_ZONE = "Asia/Kolkata";

// The backend stores producer timestamps as naive UTC datetimes.
function parseRiskTimestamp(value: string): Date {
  const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(value);
  return new Date(hasTimezone ? value : `${value}Z`);
}

function formatDate(value: string | null) {
  if (!value) return "—";

  const date = parseRiskTimestamp(value);
  if (Number.isNaN(date.getTime())) return "—";

  return (
    date.toLocaleString("en-IN", {
      timeZone: IST_TIME_ZONE,
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    }) + " IST"
  );
}


export default function PatientDetail() {
  const {
    id,
  } = useParams();

  const navigate =
    useNavigate();

  const { theme } = useTheme();

  const {
    user,
    logout,
  } = useAuth();

  const patientId =
    Number(id);

  const [patient, setPatient] =
    useState<Patient | null>(
      null,
    );

  const [latestRisk, setLatestRisk] =
    useState<RiskEvent | null>(
      null,
    );

  const [history, setHistory] =
    useState<RiskEvent[]>(
      [],
    );

  const [explanation, setExplanation] =
    useState<RiskExplanation | null>(
      null,
    );

  const [shapHistory, setShapHistory] =
    useState<RiskExplanation[]>([]);

  const [shapView, setShapView] = useState<"impact" | "waterfall" | "signed">("impact");

  const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [error, setError] =
    useState("");

  const [
    showDischargeModal,
    setShowDischargeModal,
  ] = useState(false);

  const [
    discharging,
    setDischarging,
  ] = useState(false);

  async function loadPatient(
    showRefresh = false,
  ) {
    if (!patientId) {
      setError(
        "Invalid patient ID.",
      );

      setLoading(false);
      return;
    }

    if (showRefresh) {
      setRefreshing(true);
    }

    try {
      setError("");

      const patientData =
        await getPatient(
          patientId,
        );

      setPatient(
        patientData,
      );

      const [
        risk,
        riskHistory,
        riskExplanation,
        historicalExplanations,
      ] = await Promise.all([
        getLatestRisk(
          patientId,
        ).catch(
          () => null,
        ),

        getRiskHistory(
          patientId,
          100,
        ).catch(
          () => [],
        ),

        getRiskExplanation(
          patientId,
        ).catch(
          () => null,
        ),

        getRiskExplanationHistory(
          patientId,
          100,
        ).catch(
          () => [],
        ),
      ]);

      setLatestRisk(
        risk,
      );

      setHistory(
        riskHistory,
      );

      setExplanation(
        riskExplanation,
      );

      setShapHistory(
        historicalExplanations,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load patient.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function handleDischarge() {
    if (!patient) {
      return;
    }

    setDischarging(true);
    setError("");

    try {
      const updatedPatient =
        await dischargePatient(
          patient.id,
        );

      setPatient(
        updatedPatient,
      );

      setShowDischargeModal(
        false,
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to discharge patient.";

      if (
        message.includes("Patient was discharged in the database") &&
        message.includes("realtime discharge event")
      ) {
        setError(
          `${message} The patient record is discharged; realtime delivery needs attention.`,
        );
        await loadPatient(true);
        setShowDischargeModal(false);
      } else {
        setError(message);
      }
    } finally {
      setDischarging(false);
    }
  }

  useEffect(() => {
    loadPatient();

    const interval =
      window.setInterval(
        () => {
          loadPatient();
        },
        10000,
      );

    return () =>
      window.clearInterval(
        interval,
      );
  }, [patientId]);

  const chartData = useMemo(
    () =>
      history
        .filter(
          (event) =>
            event.event_time !== null &&
            Number.isFinite(parseRiskTimestamp(event.event_time).getTime()),
        )
        .slice()
        .sort(
          (a, b) =>
            parseRiskTimestamp(a.event_time!).getTime() - parseRiskTimestamp(b.event_time!).getTime(),
        )
        .map((event) => ({
          timestamp: parseRiskTimestamp(event.event_time!).getTime(),
          eventTime: event.event_time,
          risk:
            event.ensemble_probability !== null
              ? Number((event.ensemble_probability * 100).toFixed(2))
              : null,
          threshold:
            event.threshold !== null
              ? Number((event.threshold * 100).toFixed(2))
              : null,
        })),
    [history],
  );

  // Only show SHAP when it belongs to the exact risk event on screen.
  const explanationMatchesLatestRisk =
    Boolean(
      latestRisk &&
      explanation &&
      latestRisk.stay_id === explanation.stay_id &&
      latestRisk.window_id === explanation.window_id
    );

  const shapFeatures =
    explanationMatchesLatestRisk && explanation?.status === "ok"
      ? explanation.top_features ?? []
      : [];

  const shapTrend = useMemo(() => {
    const validHistory = shapHistory
      .filter(
        (item) =>
          item.status === "ok" &&
          typeof item.event_time === "string" &&
          Array.isArray((item as RiskExplanation & {
            top_features?: import("../api/client").ShapFeature[];
          }).top_features),
      )
      .slice()
      .sort((a, b) => {
        const timeA = parseRiskTimestamp(a.event_time!).getTime();
        const timeB = parseRiskTimestamp(b.event_time!).getTime();
        return timeA - timeB || a.id - b.id;
      });

    const importance = new Map<string, { total: number; count: number }>();

    for (const item of validHistory) {
      const features = (item as RiskExplanation & {
        top_features?: import("../api/client").ShapFeature[];
      }).top_features ?? [];

      for (const feature of features) {
        if (
          !feature.feature ||
          !Number.isFinite(feature.shap_value)
        ) continue;

        const current = importance.get(feature.feature) ?? {
          total: 0,
          count: 0,
        };
        current.total += Math.abs(feature.shap_value);
        current.count += 1;
        importance.set(feature.feature, current);
      }
    }

    const selectedFeatures = [...importance.entries()]
      .sort(
        (a, b) =>
          b[1].total / b[1].count - a[1].total / a[1].count,
      )
      .slice(0, 5)
      .map(([feature]) => feature);

    const rows = validHistory.map((item) => {
      const features = (item as RiskExplanation & {
        top_features?: import("../api/client").ShapFeature[];
      }).top_features ?? [];

      const row: Record<string, string | number | null> = {
        time: item.event_time!,
        window: item.window_id,
      };

      for (const name of selectedFeatures) {
        const match = features.find((feature) => feature.feature === name);
        row[name] =
          match && Number.isFinite(match.shap_value)
            ? match.shap_value
            : null;
      }

      return row;
    });

    return { rows, selectedFeatures };
  }, [shapHistory]);

  const maxShap =
    Math.max(
      ...shapFeatures.map((feature) =>
        Math.abs(feature.shap_value),
      ),
      0.000001,
    );

  const waterfallSteps = (() => {
    if (
      !explanationMatchesLatestRisk ||
      !explanation ||
      explanation.status !== "ok" ||
      !Array.isArray(explanation.shap_values) ||
      explanation.shap_values.length === 0
    ) {
      return null;
    }

    const namedFeatures = [...(explanation.top_features ?? [])]
      .filter((feature) => Number.isFinite(feature.shap_value))
      .sort(
        (a, b) =>
          Math.abs(b.shap_value) - Math.abs(a.shap_value),
      );

    const namedTotal = namedFeatures.reduce(
      (sum, feature) => sum + feature.shap_value,
      0,
    );
    const fullTotal = explanation.shap_values.reduce(
      (sum, value) => sum + value,
      0,
    );
    if (
      !Number.isFinite(explanation.expected_value) ||
      !Number.isFinite(explanation.model_margin) ||
      !explanation.shap_values.every(Number.isFinite)
    ) {
      return null;
    }

    const reconstructedMargin = explanation.expected_value + fullTotal;
    const reconstructionError = explanation.model_margin - reconstructedMargin;
    const reconstructionMatches = Math.abs(reconstructionError) <= 0.001;
    const otherContribution = fullTotal - namedTotal;

    const contributions = namedFeatures.map((feature) => ({
      label: feature.feature,
      value: feature.shap_value,
    }));

    if (Math.abs(otherContribution) > 1e-8) {
      contributions.push({
        label: "Other features",
        value: otherContribution,
      });
    }

    let cumulative = explanation.expected_value;
    const steps = contributions.map((feature) => {
      const start = cumulative;
      cumulative += feature.value;
      return {
        ...feature,
        start,
        end: cumulative,
      };
    });

    return {
      baseline: explanation.expected_value,
      finalValue: explanation.model_margin,
      reconstructedMargin,
      reconstructionError,
      reconstructionMatches,
      steps,
      min: Math.min(
        explanation.expected_value,
        explanation.model_margin,
        ...steps.map((step) => Math.min(step.start, step.end)),
      ),
      max: Math.max(
        explanation.expected_value,
        explanation.model_margin,
        ...steps.map((step) => Math.max(step.start, step.end)),
      ),
    };
  })();

  function handleLogout() {
    logout();

    navigate("/login", {
      replace: true,
    });
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />

        <span>
          Loading patient record...
        </span>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="loading-screen">
        <div className="not-found-box">
          <h2>
            Patient not found
          </h2>

          <p>
            {error ||
              "The requested patient could not be loaded."}
          </p>

          <button
            className="primary-button"
            onClick={() =>
              navigate(
                "/patients",
              )
            }
          >
            Back to patients
          </button>
        </div>
      </div>
    );
  }

  const elevated =
    latestRisk?.prediction ===
    1;

  const isDischarged =
    patient.status ===
    "DISCHARGED";

  const canDischarge =
    user?.role ===
      "NURSE" &&
    !isDischarged;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">
            <Activity size={22} />
          </div>

          <div>
            <strong>
              VITALIS
            </strong>

            <span>
              Clinical Intelligence
            </span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">
            WORKSPACE
          </div>

          <button
            className="nav-item nav-button"
            onClick={() =>
              navigate(
                "/dashboard",
              )
            }
          >
            <Activity size={18} />
            Overview
          </button>

          <button
            className="nav-item nav-button active"
            onClick={() =>
              navigate(
                "/patients",
              )
            }
          >
            <UsersIcon />
            Patients
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-pulse" />

            <div>
              <strong>
                System operational
              </strong>

              <span>
                Realtime inference active
              </span>
            </div>
          </div>

          <button
            className="sign-out-button"
            onClick={handleLogout}
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className="main-content patient-detail-main">
        <div className="detail-topbar">
          <button
            className="back-link"
            onClick={() => navigate("/patients")}
          >
            <ArrowLeft size={17} />
            Back to patients
          </button>

          <div className="detail-topbar-actions">
            <ThemeToggle />
            <button
              className="refresh-button"
              onClick={() => loadPatient(true)}
              disabled={refreshing}
            >
              <RefreshCw
                size={16}
                className={refreshing ? "spin" : ""}
              />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        <section className="patient-hero">
          <div className="patient-hero-main">
            <div className="patient-hero-avatar">
              {patient.name
                .split(" ")
                .map(
                  (part) =>
                    part[0],
                )
                .slice(
                  0,
                  2,
                )
                .join("")}
            </div>

            <div>
              <span className="eyebrow">
                PATIENT RECORD
              </span>

              <h1>
                {patient.name}
              </h1>

              <p>
                {
                  patient.patient_identifier
                }

                <span>
                  •
                </span>

                {
                  patient.gender ??
                  "Gender —"
                }

                <span>
                  •
                </span>

                Age{" "}
                {patient.age ??
                  "—"}
              </p>
            </div>
          </div>

          <div
            className={
              elevated
                ? "hero-status elevated"
                : "hero-status normal"
            }
          >
            <span>
              {latestRisk?.status ??
                "NO DATA"}
            </span>

            <strong>
              {latestRisk
                ? elevated
                  ? "ELEVATED RISK"
                  : "BELOW THRESHOLD"
                : "NO RISK DATA"}
            </strong>
          </div>
        </section>

        <section className="patient-meta-grid">
          <div className="meta-card">
            <span>
              ICU UNIT
            </span>

            <strong>
              {
                patient.icu_unit ??
                "—"
              }
            </strong>
          </div>

          <div className="meta-card">
            <span>
              BED
            </span>

            <strong>
              {
                patient.bed_number ??
                "—"
              }
            </strong>
          </div>

          <div className="meta-card">
            <span>
              ADMISSION
            </span>

            <strong>
              {
                patient.admission_type ??
                "—"
              }
            </strong>
          </div>

          <div className="meta-card">
            <span>
              STATUS
            </span>

            <strong>
              {patient.status}
            </strong>
          </div>
        </section>

        <section className="risk-hero-card">
          <div className="risk-hero-heading">
            <div>
              <span className="eyebrow">
                CURRENT MORTALITY RISK
              </span>

              <h2>
                Ensemble prediction
              </h2>
            </div>

            <div className="risk-hero-value">
              {percentage(
                latestRisk?.ensemble_probability,
              )}
            </div>
          </div>

          <div className="threshold-track">
            <span
              className="threshold-risk"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(
                    0,
                    (
                      latestRisk?.ensemble_probability ??
                      0
                    ) * 100,
                  ),
                )}%`,
              }}
            />

            {latestRisk?.threshold != null && (
              <span
                className="threshold-marker"
                style={{
                  left: `${Math.min(
                    100,
                    Math.max(0, latestRisk.threshold * 100),
                  )}%`,
                }}
              />
            )}
          </div>

          <div className="threshold-labels">
            <span>
              0%
            </span>

            <span>
              Threshold{" "}
              {percentage(
                latestRisk?.threshold,
              )}
            </span>

            <span>
              100%
            </span>
          </div>

          <div className="risk-meta-line">
            <span>
              Prediction:{" "}
              <strong>
                {latestRisk?.prediction ===
                1
                  ? "Positive"
                  : latestRisk?.prediction ===
                      0
                    ? "Negative"
                    : "—"}
              </strong>
            </span>

            <span>
              Window:{" "}
              <strong>
                {latestRisk?.window_id ??
                  "—"}
              </strong>
            </span>

            <span>
              Sequence:{" "}
              <strong>
                {
                  latestRisk?.sequence_length ??
                  "—"
                }
              </strong>
            </span>

            <span>
              Updated:{" "}
              <strong>
                {formatDate(
                  latestRisk?.event_time ??
                    null,
                )}
              </strong>
            </span>
          </div>
        </section>

        <section className="model-grid">
          <ModelCard
            icon={
              <Database size={18} />
            }
            title="XGBoost"
            value={percentage(
              latestRisk?.xgb_probability,
            )}
            subtitle={`Raw ${percentage(
              latestRisk?.xgb_raw_probability,
            )}`}
          />

          <ModelCard
            icon={
              <Activity size={18} />
            }
            title="LSTM"
            value={percentage(
              latestRisk?.lstm_probability,
            )}
            subtitle={`Raw ${percentage(
              latestRisk?.lstm_raw_probability,
            )}`}
          />

          <ModelCard
            icon={
              <HeartPulse size={18} />
            }
            title="Ensemble"
            value={percentage(
              latestRisk?.ensemble_probability,
            )}
            subtitle="Frozen ensemble model"
            featured
          />
        </section>

        <section className={`detail-two-column ${user?.role !== "DOCTOR" ? "detail-single-column" : ""}`}>
          <div className="detail-left-stack">
<div className="detail-panel risk-history-panel">
            <div className="panel-header">
              <div>
                <span className="eyebrow">
                  TEMPORAL VIEW
                </span>

                <h2>
                  Risk history
                </h2>
              </div>

              <Clock3 size={18} />
            </div>

            {chartData.length ===
            0 ? (
              <div className="panel-empty">
                No risk history available.
              </div>
            ) : (
              <div className="chart-wrapper">
                <ResponsiveContainer
                  width="100%"
                  height={260}
                >
                  <AreaChart
                    data={
                      chartData
                    }
                  >
                    <defs>
                      <linearGradient
                        id="riskGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#24c7b1"
                          stopOpacity={
                            0.35
                          }
                        />

                        <stop
                          offset="100%"
                          stopColor="#24c7b1"
                          stopOpacity={
                            0
                          }
                        />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      stroke={theme === "light" ? "#e2e8f0" : "#334155"}
                      vertical={false}
                    />

                    <XAxis
                      dataKey="timestamp"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(value) =>
                        new Date(Number(value)).toLocaleString("en-IN", {
                          timeZone: IST_TIME_ZONE,
                          day: "2-digit",
                          month: "short",
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      }
                      minTickGap={65}
                      stroke={theme === "light" ? "#64748b" : "#a3afbf"}
                      tick={{ fill: theme === "light" ? "#475569" : "#cbd5e1", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />

                    <YAxis
                      domain={[
                        0,
                        100,
                      ]}
                      stroke="#647083"
                      tickLine={
                        false
                      }
                      axisLine={
                        false
                      }
                      tickFormatter={(
                        value,
                      ) =>
                        `${value}%`
                      }
                    />

                    <Tooltip
                      labelFormatter={(value) =>
                        new Date(Number(value)).toLocaleString("en-IN", {
                          timeZone: IST_TIME_ZONE,
                          day: "2-digit",
                          month: "short",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                          hour12: true,
                        }) + " IST"
                      }
                      contentStyle={{
                        background: theme === "light" ? "#ffffff" : "#111722",
                        border: `1px solid ${theme === "light" ? "#cbd5e1" : "#475569"}`,
                        borderRadius: 10,
                        color: theme === "light" ? "#0f172a" : "#edf2f8",
                      }}
                      formatter={(value, name) => [
                        value == null ? "—" : `${Number(value).toFixed(1)}%`,
                        name === "threshold" ? "Threshold" : "Risk",
                      ]}
                    />

                    <Area
                      type="monotone"
                      dataKey="risk"
                      stroke="#24c7b1"
                      strokeWidth={
                        2
                      }
                      fill="url(#riskGradient)"
                      connectNulls
                    />

                    <Area
                      type="monotone"
                      dataKey="threshold"
                      stroke="#f1b75b"
                      strokeDasharray="5 5"
                      strokeWidth={
                        1.5
                      }
                      fill="none"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
<div className="shap-history-section">
              <div className="panel-header">
                <div>
                  <span className="eyebrow">HISTORICAL EXPLAINABILITY</span>
                  <h3>SHAP contributions over time</h3>
                </div>
                <span>{shapTrend.rows.length} windows</span>
              </div>



              {shapTrend.rows.length < 2 || shapTrend.selectedFeatures.length === 0 ? (
                <div className="panel-empty">
                  At least two historical SHAP explanations are needed to show trends.
                </div>
              ) : (
                <div style={{ width: "100%", height: 320 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={shapTrend.rows}
                      margin={{ top: 12, right: 18, left: 4, bottom: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color, #64748b)" opacity={0.25} />
                      <XAxis
                        dataKey="time"
                        tickFormatter={(value: string) => {
                          const date = parseRiskTimestamp(value);
                          return Number.isNaN(date.getTime())
                            ? ""
                            : date.toLocaleString("en-IN", {
                                timeZone: IST_TIME_ZONE,
                                day: "2-digit",
                                month: "short",
                                hour: "2-digit",
                                minute: "2-digit",
                                hour12: true,
                              });
                        }}
                        minTickGap={28}
                      />
                      <YAxis />
                      <Tooltip
                        labelFormatter={(value) => {
                          const date = parseRiskTimestamp(String(value));
                          return Number.isNaN(date.getTime())
                            ? String(value)
                            : `${formatDate(String(value))}`;
                        }}
                        formatter={(value, name) => [
                          typeof value === "number" ? value.toFixed(4) : "Unavailable",
                          String(name),
                        ]}
                      />
                      <Legend />
                      {shapTrend.selectedFeatures.map((feature, index) => (
                        <Line
                          key={feature}
                          type="monotone"
                          dataKey={feature}
                          name={feature}
                          connectNulls={false}
                          dot={false}
                          activeDot={{ r: 4 }}
                          strokeWidth={2}
                          stroke={
                            ["#24c7b1", "#e6a23c", "#8b7cf6", "#e879a9", "#60a5fa"][index]
                          }
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
</div>

          {user?.role === "DOCTOR" && (
            <div className="detail-panel">
              <div className="panel-header">
                <div>
                  <span className="eyebrow">
                    EXPLAINABILITY
                </span>

                <h2>
                  {shapView === "impact" ? "Top SHAP features" : shapView === "waterfall" ? "SHAP contribution impact" : "Signed SHAP contributions"}
                </h2>
              </div>

              <ShieldAlert size={18} />
            </div>

            <div className="shap-view-toggle" role="group" aria-label="SHAP plot type">
              <button
                type="button"
                className={shapView === "impact" ? "active" : ""}
                onClick={() => setShapView("impact")}
              >
                Feature impact
              </button>
              <button
                type="button"
                className={shapView === "waterfall" ? "active" : ""}
                onClick={() => setShapView("waterfall")}
              >
                Contribution view
              </button>
              <button
                type="button"
                className={shapView === "signed" ? "active" : ""}
                onClick={() => setShapView("signed")}
              >
                Signed contributions
              </button>
            </div>

            {shapFeatures.length ===
            0 ? (
              <div className="panel-empty">
                No SHAP explanation
                available.
              </div>
            ) : (
              shapView === "signed" ? (
                <div className="shap-contribution-chart">
                  <div className="shap-contribution-guide">
                    <div className="shap-contribution-legend">
                      <span><i className="positive" /> Positive SHAP value</span>
                      <span><i className="negative" /> Negative SHAP value</span>
                    </div>
                  </div>
                  {(() => {
                    const maxAbs = Math.max(
                      ...shapFeatures.map((feature) => Math.abs(feature.shap_value)),
                      0.000001,
                    );
                    return (
                      <>
                        <div className="shap-contribution-axis">
                          <span>{(-maxAbs).toFixed(3)}</span>
                          <span>0</span>
                          <span>{maxAbs.toFixed(3)}</span>
                        </div>
                        {shapFeatures.map((feature, index) => {
                          const positive = feature.shap_value >= 0;
                          const width = (Math.abs(feature.shap_value) / maxAbs) * 50;
                          const left = positive ? 50 : 50 - width;
                          return (
                            <div className="shap-contribution-row" key={`${feature.feature}-${index}`}>
                              <div className="shap-contribution-label">
                                {feature.feature}
                                <small>
                                  Value: {feature.value === null
                                    ? "missing"
                                    : typeof feature.value === "number"
                                      ? Number.isInteger(feature.value)
                                        ? feature.value
                                        : feature.value.toFixed(2)
                                      : feature.value}
                                </small>
                              </div>
                              <div className="shap-contribution-track">
                                <span className="shap-zero-line" />
                                <span
                                  className={`shap-contribution-bar ${positive ? "positive" : "negative"}`}
                                  style={{ left: `${left}%`, width: `${width}%` }}
                                />
                              </div>
                              <strong className={positive ? "shap-positive" : "shap-negative"}>
                                {positive ? "+" : ""}{feature.shap_value.toFixed(3)}
                              </strong>
                            </div>
                          );
                        })}
                      </>
                    );
                  })()}
                </div>
              ) : shapView === "waterfall" ? (
                !waterfallSteps || waterfallSteps.steps.length === 0 ? (
                  <div className="panel-empty">
                    Full SHAP waterfall data is not available yet.
                  </div>
                ) : (
                  <div className="shap-waterfall">
                    <div className="shap-waterfall-summary">
                      <div>
                        <span>Baseline</span>
                        <strong>{waterfallSteps.baseline.toFixed(3)}</strong>
                      </div>
                      <div>
                        <span>Final model margin</span>
                        <strong>{waterfallSteps.finalValue.toFixed(3)}</strong>
                      </div>
                    </div>


                    <p className="shap-waterfall-note">
                      SHAP reconstruction:{" "}
                      <strong>
                        {waterfallSteps.reconstructionMatches
                          ? "Consistent"
                          : "Mismatch detected"}
                      </strong>
                      {" · "}Reconstructed margin{" "}
                      {waterfallSteps.reconstructedMargin.toFixed(3)}
                      {" · "}Difference{" "}
                      {waterfallSteps.reconstructionError >= 0 ? "+" : ""}
                      {waterfallSteps.reconstructionError.toFixed(4)}
                      {" · "}Tolerance ±0.001
                    </p>

                    {(() => {
                      const range = Math.max(
                        waterfallSteps.max - waterfallSteps.min,
                        0.000001,
                      );
                      const position = (value: number) =>
                        ((value - waterfallSteps.min) / range) * 100;

                      return (
                        <div className="shap-waterfall-rows">
                          <div className="shap-waterfall-axis">
                            <span>{waterfallSteps.min.toFixed(2)}</span>
                            <span>{((waterfallSteps.min + waterfallSteps.max) / 2).toFixed(2)}</span>
                            <span>{waterfallSteps.max.toFixed(2)}</span>
                          </div>

                          {waterfallSteps.steps.map((step, index) => {
                            const left = Math.min(
                              position(step.start),
                              position(step.end),
                            );
                            const width = Math.max(
                              Math.abs(position(step.end) - position(step.start)),
                              0.5,
                            );
                            const positive = step.value >= 0;

                            return (
                              <div
                                className="shap-waterfall-row"
                                key={`${step.label}-${index}`}
                              >
                                <div className="shap-waterfall-label" title={step.label}>
                                  {step.label}
                                </div>
                                <div className="shap-waterfall-track">
                                  <span
                                    className={`shap-waterfall-bar ${positive ? "positive" : "negative"}`}
                                    style={{
                                      left: `${left}%`,
                                      width: `${width}%`,
                                    }}
                                  />
                                </div>
                                <strong className={positive ? "shap-positive" : "shap-negative"}>
                                  {positive ? "+" : ""}{step.value.toFixed(3)}
                                </strong>
                                <small>
                                  {step.start.toFixed(3)} → {step.end.toFixed(3)}
                                </small>
                              </div>
                            );
                          })}

                          <div className="shap-waterfall-final">
                            <span>Model output</span>
                            <strong>{waterfallSteps.finalValue.toFixed(3)}</strong>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )
              ) : (
                <div className="shap-list">
                  {shapFeatures.map((feature, index) => {
                    const positive = feature.shap_value >= 0;
                    const width = (Math.abs(feature.shap_value) / maxShap) * 100;

                    return (
                      <div className="shap-item" key={`${feature.feature}-${index}`}>
                        <div className="shap-header">
                          <span>{feature.feature}</span>
                          <strong className={positive ? "shap-positive" : "shap-negative"}>
                            {positive ? "+" : ""}{feature.shap_value.toFixed(3)}
                          </strong>
                        </div>

                        <div className="shap-track">
                          <span
                            className={positive ? "shap-positive-bar" : "shap-negative-bar"}
                            style={{ width: `${width}%` }}
                          />
                        </div>

                        <small>
                          Value: {feature.value === null
                            ? "missing"
                            : typeof feature.value === "number"
                              ? Number.isInteger(feature.value)
                                ? feature.value
                                : feature.value.toFixed(2)
                              : feature.value}
                        </small>
                      </div>
                    );
                  })}
                </div>
              )
            )}


            </div>
          )}
        </section>



        <section className="detail-panel history-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">
                INFERENCE EVENTS
              </span>

              <h2>
                Recent risk events
              </h2>
            </div>

            <Activity size={18} />
          </div>

          <div className="history-table">
            <div className="history-header">
              <span>
                Time
              </span>

              <span>
                Window
              </span>

              <span>
                Status
              </span>

              <span>
                Sequence
              </span>

              <span>
                XGB
              </span>

              <span>
                LSTM
              </span>

              <span>
                Ensemble
              </span>
            </div>

            {history.map(
              (event) => (
                <div
                  className="history-row"
                  key={
                    event.id
                  }
                >
                  <span>
                    {formatDate(
                      event.event_time,
                    )}
                  </span>

                  <span>
                    #
                    {
                      event.window_id
                    }
                  </span>

                  <span>
                    <b
                      className={
                        event.status ===
                        "FINAL"
                          ? "history-final"
                          : "history-live"
                      }
                    >
                      {
                        event.status
                      }
                    </b>
                  </span>

                  <span>
                    {
                      event.sequence_length ??
                      "—"
                    }
                  </span>

                  <span>
                    {percentage(
                      event.xgb_probability,
                    )}
                  </span>

                  <span>
                    {percentage(
                      event.lstm_probability,
                    )}
                  </span>

                  <span className="history-ensemble">
                    {percentage(
                      event.ensemble_probability,
                    )}
                  </span>
                </div>
              ),
            )}
          </div>
        </section>

        {canDischarge && (
          <section className="patient-action-bar">
            <div>
              <span className="eyebrow">
                PATIENT ACTION
              </span>

              <h3>
                End active ICU stay
              </h3>

              <p>
                Discharge this patient when
                their current ICU stay has
                ended.
              </p>
            </div>

            <button
              type="button"
              className="danger-button"
              onClick={() =>
                setShowDischargeModal(
                  true,
                )
              }
            >
              <LogOut size={17} />
              Discharge Patient
            </button>
          </section>
        )}

        {user?.role === "DOCTOR" && (
          <section className="technical-footer">
          <div>
            <span>
              Model version
            </span>

            <strong>
              {
                latestRisk?.model_version ??
                "—"
              }
            </strong>
          </div>

          <div>
            <span>
              SHAP status
            </span>

            <strong>
              {
                latestRisk?.shap_status ??
                "—"
              }
            </strong>
          </div>

          <div>
            <span>
              Window start
            </span>

            <strong>
              {formatDate(
                latestRisk?.window_start ??
                  null,
              )}
            </strong>
          </div>

          <div>
            <span>
              Window end
            </span>

            <strong>
              {formatDate(
                latestRisk?.window_end ??
                  null,
              )}
            </strong>
          </div>
          </section>
        )}
        {showDischargeModal && (
          <div
            className="modal-backdrop"
            role="presentation"
            onClick={() =>
              !discharging &&
              setShowDischargeModal(false)
            }
          >
            <div
              className="confirm-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="discharge-title"
              onClick={(event) =>
                event.stopPropagation()
              }
            >
              <div className="confirm-modal-header">
                <div className="confirm-modal-icon">
                  <LogOut size={20} />
                </div>

                <button
                  type="button"
                  className="modal-close-button"
                  onClick={() =>
                    !discharging &&
                    setShowDischargeModal(false)
                  }
                  disabled={discharging}
                  aria-label="Close"
                >
                  <X size={18} />
                </button>
              </div>

              <div className="confirm-modal-content">
                <span className="eyebrow">
                  DISCHARGE PATIENT
                </span>

                <h2 id="discharge-title">
                  Discharge {patient.name}?
                </h2>

                <p>
                  This will mark the patient as
                  discharged, close the active ICU
                  stay, remove the active doctor
                  assignment, and send the discharge
                  event to the realtime monitoring
                  pipeline.
                </p>

                <div className="discharge-summary">
                  <div>
                    <span>
                      Patient
                    </span>

                    <strong>
                      {patient.name}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Patient ID
                    </span>

                    <strong>
                      {patient.patient_identifier}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Current status
                    </span>

                    <strong>
                      {patient.status}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="confirm-modal-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() =>
                    setShowDischargeModal(false)
                  }
                  disabled={discharging}
                >
                  Cancel
                </button>

                <button
                  type="button"
                  className="danger-button"
                  onClick={handleDischarge}
                  disabled={discharging}
                >
                  {discharging ? (
                    <>
                      <RefreshCw
                        size={16}
                        className="spin"
                      />
                      Discharging...
                    </>
                  ) : (
                    <>
                      <LogOut size={16} />
                      Confirm Discharge
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

function ModelCard({
  icon,
  title,
  value,
  subtitle,
  featured = false,
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  subtitle: string;
  featured?: boolean;
}) {
  return (
    <div
      className={
        featured
          ? "model-card ensemble-model"
          : "model-card"
      }
    >
      <div className="model-card-icon">
        {icon}
      </div>

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        {subtitle}
      </small>
    </div>
  );
}

function UsersIcon() {
  return (
    <Stethoscope size={18} />
  );
}
