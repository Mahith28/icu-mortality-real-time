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
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
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
  getRiskHistory,
  type Patient,
  type RiskEvent,
} from "../api/client";

import { useAuth } from "../auth/AuthContext";

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

function formatDate(
  value: string | null,
) {
  if (!value) {
    return "—";
  }

  return new Date(
    value,
  ).toLocaleString();
}

function formatTime(
  value: string | null,
) {
  if (!value) {
    return "";
  }

  return new Date(
    value,
  ).toLocaleTimeString(
    [],
    {
      hour: "2-digit",
      minute: "2-digit",
    },
  );
}

export default function PatientDetail() {
  const {
    id,
  } = useParams();

  const navigate =
    useNavigate();

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
    useState<RiskEvent | null>(
      null,
    );

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
      setError(
        err instanceof Error
          ? err.message
          : "Unable to discharge patient.",
      );
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

  const chartData =
    useMemo(
      () =>
        [...history]
          .reverse()
          .map(
            (event) => ({
              time:
                formatTime(
                  event.event_time,
                ),

              risk:
                event.ensemble_probability !==
                null
                  ? Number(
                      (
                        event.ensemble_probability *
                        100
                      ).toFixed(
                        2,
                      ),
                    )
                  : null,

              threshold:
                event.threshold !==
                null
                  ? Number(
                      (
                        event.threshold *
                        100
                      ).toFixed(
                        2,
                      ),
                    )
                  : null,
            }),
          ),
      [history],
    );

  const shapFeatures =
    explanation?.top_features ??
    latestRisk?.top_features ??
    [];

  const maxShap =
    Math.max(
      ...shapFeatures.map(
        (feature) =>
          Math.abs(
            feature.shap_value,
          ),
      ),
      0.000001,
    );

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
            onClick={() =>
              navigate(
                "/patients",
              )
            }
          >
            <ArrowLeft size={17} />
            Back to patients
          </button>

          <button
            className="refresh-button"
            onClick={() =>
              loadPatient(
                true,
              )
            }
            disabled={
              refreshing
            }
          >
            <RefreshCw
              size={16}
              className={
                refreshing
                  ? "spin"
                  : ""
              }
            />

            Refresh
          </button>
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

            <span
              className="threshold-marker"
              style={{
                left: `${Math.min(
                  100,
                  Math.max(
                    0,
                    (
                      latestRisk?.threshold ??
                      0.108
                    ) * 100,
                  ),
                )}%`,
              }}
            />
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

        <section className="detail-two-column">
          <div className="detail-panel">
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
                  height={310}
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
                      stroke="#202734"
                      vertical={
                        false
                      }
                    />

                    <XAxis
                      dataKey="time"
                      stroke="#647083"
                      tickLine={
                        false
                      }
                      axisLine={
                        false
                      }
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
                      contentStyle={{
                        background:
                          "#111722",
                        border:
                          "1px solid rgba(148,163,184,.18)",
                        borderRadius:
                          10,
                        color:
                          "#edf2f8",
                      }}
                      formatter={(
                        value,
                      ) => [
                        `${Number(
                          value,
                        ).toFixed(
                          1,
                        )}%`,
                        "Risk",
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

          <div className="detail-panel">
            <div className="panel-header">
              <div>
                <span className="eyebrow">
                  EXPLAINABILITY
                </span>

                <h2>
                  Top SHAP features
                </h2>
              </div>

              <ShieldAlert size={18} />
            </div>

            {shapFeatures.length ===
            0 ? (
              <div className="panel-empty">
                No SHAP explanation
                available.
              </div>
            ) : (
              <div className="shap-list">
                {shapFeatures.map(
                  (
                    feature,
                    index,
                  ) => {
                    const positive =
                      feature.shap_value >=
                      0;

                    const width =
                      (
                        Math.abs(
                          feature.shap_value,
                        ) /
                        maxShap
                      ) *
                      100;

                    return (
                      <div
                        className="shap-item"
                        key={`${feature.feature}-${index}`}
                      >
                        <div className="shap-header">
                          <span>
                            {
                              feature.feature
                            }
                          </span>

                          <strong
                            className={
                              positive
                                ? "shap-positive"
                                : "shap-negative"
                            }
                          >
                            {positive
                              ? "+"
                              : ""}
                            {feature.shap_value.toFixed(
                              3,
                            )}
                          </strong>
                        </div>

                        <div className="shap-track">
                          <span
                            className={
                              positive
                                ? "shap-positive-bar"
                                : "shap-negative-bar"
                            }
                            style={{
                              width: `${width}%`,
                            }}
                          />
                        </div>

                        <small>
                          Value:{" "}
                          {feature.value ===
                          null
                            ? "missing"
                            : feature.value}
                        </small>
                      </div>
                    );
                  },
                )}
              </div>
            )}
          </div>
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
