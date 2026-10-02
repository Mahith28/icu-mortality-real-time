import {
  Activity,
  Bell,
  ChevronRight,
  CircleUserRound,
  LogOut,
  RefreshCw,
  ShieldCheck,
  Users,
} from "lucide-react";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  getLatestRisk,
  getPatients,
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

export default function Dashboard() {
  const navigate = useNavigate();

  const {
    user,
    logout,
  } = useAuth();

  const [patients, setPatients] =
    useState<Patient[]>([]);

  const [risks, setRisks] =
    useState<
      Record<
        number,
        RiskEvent | null
      >
    >({});

  const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [error, setError] =
    useState("");

  async function loadDashboard(
    showRefresh = false,
  ) {
    if (showRefresh) {
      setRefreshing(true);
    }

    try {
      setError("");

      const patientList =
        await getPatients();

      setPatients(patientList);

      const results =
        await Promise.all(
          patientList.map(
            async (patient) => {
              try {
                const risk =
                  await getLatestRisk(
                    patient.id,
                  );

                return [
                  patient.id,
                  risk,
                ] as const;
              } catch {
                return [
                  patient.id,
                  null,
                ] as const;
              }
            },
          ),
        );

      setRisks(
        Object.fromEntries(results),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load dashboard.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadDashboard();

    const interval =
      window.setInterval(
        () => {
          loadDashboard();
        },
        10000,
      );

    return () =>
      window.clearInterval(
        interval,
      );
  }, []);

  const elevatedCount =
    useMemo(
      () =>
        patients.filter(
          (patient) =>
            risks[patient.id]
              ?.prediction === 1,
        ).length,
      [patients, risks],
    );

  function handleLogout() {
    logout();

    navigate("/login", {
      replace: true,
    });
  }

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
            className="nav-item nav-button active"
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
            className="nav-item nav-button"
            onClick={() =>
              navigate(
                "/patients",
              )
            }
          >
            <Users size={18} />
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
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              ICU COMMAND CENTER
            </span>

            <h1>
              Clinical overview
            </h1>
          </div>

          <div className="topbar-actions">
            <button
              className="icon-button"
              onClick={() =>
                loadDashboard(
                  true,
                )
              }
              title="Refresh"
              disabled={
                refreshing
              }
            >
              <RefreshCw
                size={18}
                className={
                  refreshing
                    ? "spin"
                    : ""
                }
              />
            </button>

            <button
              className="icon-button"
              title="Notifications"
            >
              <Bell size={18} />
            </button>

            <div className="profile-chip">
              <CircleUserRound
                size={19}
              />

              <div>
                <strong>
                  {user?.name}
                </strong>

                <span>
                  {user?.role}
                </span>
              </div>
            </div>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        <section className="stat-grid">
          <div className="stat-card">
            <div className="stat-icon">
              <Users size={19} />
            </div>

            <span>
              Patients under care
            </span>

            <strong>
              {patients.length}
            </strong>

            <small>
              Current ICU census
            </small>
          </div>

          <div className="stat-card">
            <div className="stat-icon warning-icon">
              <ShieldCheck
                size={19}
              />
            </div>

            <span>
              Elevated risk
            </span>

            <strong>
              {elevatedCount}
            </strong>

            <small>
              Above ensemble threshold
            </small>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Activity size={19} />
            </div>

            <span>
              Monitoring
            </span>

            <strong>
              LIVE
            </strong>

            <small>
              Realtime risk pipeline
            </small>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <RefreshCw size={19} />
            </div>

            <span>
              Update cycle
            </span>

            <strong>
              5 min
            </strong>

            <small>
              Clinical prediction windows
            </small>
          </div>
        </section>

        <section className="dashboard-section">
          <div className="section-heading">
            <div>
              <h2>
                Patients
              </h2>

              <p>
                Latest available ICU
                mortality risk for each
                patient.
              </p>
            </div>

            <div className="live-indicator">
              <span />
              Auto-refreshing
            </div>
          </div>

          {loading ? (
            <div className="dashboard-loading">
              <RefreshCw
                size={22}
                className="spin"
              />

              Loading clinical
              data...
            </div>
          ) : patients.length === 0 ? (
            <div className="dashboard-loading">
              No patients available.
            </div>
          ) : (
            <div className="dashboard-patient-grid">
              {patients.map(
                (patient) => {
                  const risk =
                    risks[
                      patient.id
                    ];

                  const elevated =
                    risk?.prediction ===
                    1;

                  const riskValue =
                    risk?.ensemble_probability ??
                    0;

                  return (
                    <article
                      className="patient-card"
                      key={patient.id}
                    >
                      <div className="patient-card-header">
                        <div className="patient-card-name">
                          <div className="patient-avatar">
                            {patient.name
                              .split(
                                " ",
                              )
                              .map(
                                (
                                  part,
                                ) =>
                                  part[0],
                              )
                              .slice(
                                0,
                                2,
                              )
                              .join(
                                "",
                              )}
                          </div>

                          <div>
                            <strong>
                              {
                                patient.name
                              }
                            </strong>

                            <span>
                              {
                                patient.patient_identifier
                              }
                            </span>
                          </div>
                        </div>

                        <span
                          className={
                            elevated
                              ? "risk-badge elevated"
                              : "risk-badge normal"
                          }
                        >
                          {risk
                            ? elevated
                              ? "ELEVATED"
                              : "NORMAL"
                            : "NO DATA"}
                        </span>
                      </div>

                      <div className="patient-card-info">
                        <div>
                          <span>
                            Age
                          </span>

                          <strong>
                            {
                              patient.age ??
                              "—"
                            }
                          </strong>
                        </div>

                        <div>
                          <span>
                            ICU
                          </span>

                          <strong>
                            {
                              patient.icu_unit ??
                              "—"
                            }
                          </strong>
                        </div>

                        <div>
                          <span>
                            Bed
                          </span>

                          <strong>
                            {
                              patient.bed_number ??
                              "—"
                            }
                          </strong>
                        </div>
                      </div>

                      <div className="patient-risk-row">
                        <div>
                          <span>
                            Ensemble risk
                          </span>

                          <strong>
                            {percentage(
                              risk?.ensemble_probability,
                            )}
                          </strong>
                        </div>

                        <div>
                          <span>
                            Status
                          </span>

                          <strong>
                            {
                              risk?.status ??
                              "—"
                            }
                          </strong>
                        </div>
                      </div>

                      <div className="risk-progress">
                        <span
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(
                                0,
                                riskValue *
                                  100,
                              ),
                            )}%`,
                          }}
                        />
                      </div>

                      <div className="patient-card-footer">
                        <span>
                          {elevated
                            ? "Elevated risk"
                            : "Within threshold"}
                        </span>

                        <button
                          onClick={() =>
                            navigate(
                              `/patients/${patient.id}`,
                            )
                          }
                        >
                          View patient
                          <ChevronRight
                            size={15}
                          />
                        </button>
                      </div>
                    </article>
                  );
                },
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
