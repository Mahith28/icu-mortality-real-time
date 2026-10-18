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
  useRef,
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  getDoctorAvailability,
  getHandoverAlerts,
  getLatestRisk,
  getPatients,
  setDoctorAvailability,
  type DoctorAvailability,
  type HandoverAlert,
  type Patient,
  type RiskEvent,
} from "../api/client";

import { useAuth } from "../auth/AuthContext";
import { ThemeToggle } from "../theme/ThemeContext";

function percentage(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export default function Dashboard() {
  const navigate = useNavigate();

  const {
    user,
    logout,
  } = useAuth();

  const [notificationsOpen, setNotificationsOpen] =
    useState(false);

  const [patients, setPatients] =
    useState<Patient[]>([]);

  const knownDoctorPatientIds = useRef<Set<number> | null>(null);
  const [doctorAssignmentNotifications, setDoctorAssignmentNotifications] =
    useState<Patient[]>([]);
  const [unreadDoctorAssignmentIds, setUnreadDoctorAssignmentIds] =
    useState<Set<number>>(new Set());

  const [risks, setRisks] =
    useState<
      Record<
        number,
        RiskEvent | null
      >
    >({});

  const [handoverAlerts, setHandoverAlerts] =
    useState<HandoverAlert[]>([]);

  const [dismissedHandoverIds, setDismissedHandoverIds] =
    useState<Set<number>>(new Set());

  useEffect(() => {
    if (!user?.id || user.role !== "NURSE") return;

    try {
      const stored = localStorage.getItem(
        `vitalis_dismissed_handover_alerts_${user.id}`
      );
      const ids: unknown = stored ? JSON.parse(stored) : [];
      setDismissedHandoverIds(
        new Set(
          Array.isArray(ids)
            ? ids.filter((id): id is number => Number.isInteger(id))
            : []
        )
      );
    } catch {
      setDismissedHandoverIds(new Set());
    }
  }, [user?.id, user?.role]);

  const [readHandoverIds, setReadHandoverIds] =
    useState<Set<number>>(new Set());

  useEffect(() => {
    if (!user?.id || user.role !== "NURSE") return;

    try {
      const stored = localStorage.getItem(
        `vitalis_read_handover_alerts_${user.id}`
      );
      const ids: unknown = stored ? JSON.parse(stored) : [];
      setReadHandoverIds(
        new Set(
          Array.isArray(ids)
            ? ids.filter((id): id is number => Number.isInteger(id))
            : []
        )
      );
    } catch {
      setReadHandoverIds(new Set());
    }
  }, [user?.id, user?.role]);

  const [handoverToast, setHandoverToast] =
    useState<HandoverAlert | null>(null);

  const knownHandoverIds = useRef<Set<number> | null>(null);

  const [doctorAvailability, updateDoctorAvailability] =
    useState<DoctorAvailability | null>(null);

  const [availabilitySaving, setAvailabilitySaving] =
    useState(false);

  const [availabilityError, setAvailabilityError] =
    useState("");

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


      if (user?.role === "DOCTOR") {
        const previousIds = knownDoctorPatientIds.current;
        const currentIds = new Set(patientList.map((patient) => patient.id));

        if (previousIds !== null) {
          const newlyAssigned = patientList.filter(
            (patient) => !previousIds.has(patient.id)
          );

          if (newlyAssigned.length > 0) {
            setDoctorAssignmentNotifications((current) => {
              const byId = new Map(current.map((patient) => [patient.id, patient]));
              newlyAssigned.forEach((patient) => byId.set(patient.id, patient));
              return [...byId.values()].slice(-50);
            });

            setUnreadDoctorAssignmentIds((current) => {
              const next = new Set(current);
              newlyAssigned.forEach((patient) => next.add(patient.id));
              return next;
            });
          }
        }

        knownDoctorPatientIds.current = currentIds;
      }

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

      if (user?.role === "NURSE") {
        try {
          const alerts = await getHandoverAlerts();
          const previousIds = knownHandoverIds.current;
          knownHandoverIds.current = new Set(alerts.map((alert) => alert.id));

          if (previousIds !== null) {
            const newAlert = alerts.find(
              (alert) =>
                !previousIds.has(alert.id) &&
                !dismissedHandoverIds.has(alert.id)
            );
            if (newAlert) setHandoverToast(newAlert);
          }

          const visibleAlerts = alerts.filter(
            (alert) => !dismissedHandoverIds.has(alert.id)
          );
          setHandoverAlerts(visibleAlerts);
          setHandoverToast((current) =>
            current && !alerts.some((alert) => alert.id === current.id)
              ? null
              : current
          );
        } catch (alertError) {
          console.error("Unable to load handover alerts:", alertError);
        }
      } else {
        setHandoverAlerts([]);
      }

      if (user?.role === "DOCTOR") {
        try {
          updateDoctorAvailability(await getDoctorAvailability());
          setAvailabilityError("");
        } catch (availabilityLoadError) {
          console.error("Unable to load doctor availability:", availabilityLoadError);
          setAvailabilityError(
            availabilityLoadError instanceof Error
              ? availabilityLoadError.message
              : "Unable to load availability.",
          );
        }
      } else {
        updateDoctorAvailability(null);
        setAvailabilityError("");
      }
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

  async function handleAvailabilityToggle() {
    if (!doctorAvailability || availabilitySaving) return;

    setAvailabilitySaving(true);
    setAvailabilityError("");

    try {
      await setDoctorAvailability(
        !doctorAvailability.is_online,
      );
      const updated = await getDoctorAvailability();
      updateDoctorAvailability(updated);
    } catch (err) {
      setAvailabilityError(
        err instanceof Error
          ? err.message
          : "Unable to update availability.",
      );
    } finally {
      setAvailabilitySaving(false);
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

  const activePatients = useMemo(
    () =>
      patients.filter(
        (patient) => patient.status === "ACTIVE",
      ),
    [patients],
  );



  const elevatedCount = useMemo(
    () =>
      activePatients.filter(
        (patient) => risks[patient.id]?.prediction === 1,
      ).length,
    [activePatients, risks],
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
                Service status
              </strong>

              <span>
                Inference health unverified
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
        {user?.role === "DOCTOR" && availabilityError && (
          <div className="availability-error" role="alert">
            Doctor availability: {availabilityError}
          </div>
        )}

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
            <ThemeToggle />

            {user?.role === "DOCTOR" && doctorAvailability && (
              <button
                type="button"
                className={`availability-toggle ${doctorAvailability.is_online ? "online" : "offline"}`}
                onClick={handleAvailabilityToggle}
                disabled={availabilitySaving}
                title="Set your clinical availability"
              >
                <span className="availability-dot" />
                {availabilitySaving
                  ? "Saving..."
                  : doctorAvailability.is_online
                    ? "Go Offline"
                    : "Go Online"}
              </button>
            )}

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

            <div className="notification-container">
              <button
                type="button"
                className="icon-button"
                title="Notifications"
                aria-label="Notifications"
                aria-expanded={notificationsOpen}
                onClick={() => {
                  const opening = !notificationsOpen;

                  if (opening && user?.role === "NURSE" && user.id) {
                    const nextRead = new Set([
                      ...readHandoverIds,
                      ...handoverAlerts.map((alert) => alert.id),
                    ]);

                    setReadHandoverIds(nextRead);
                    localStorage.setItem(
                      `vitalis_read_handover_alerts_${user.id}`,
                      JSON.stringify([...nextRead])
                    );
                  }

                  if (opening && user?.role === "DOCTOR") {
                    setUnreadDoctorAssignmentIds(new Set());
                  }

                  setNotificationsOpen(opening);
                }}
              >
                <Bell size={18} />
                {user?.role === "NURSE" && handoverAlerts.filter(
                  (alert) => !readHandoverIds.has(alert.id)
                ).length > 0 && (
                  <span className="notification-badge">
                    {(() => {
                      const unreadCount = handoverAlerts.filter(
                        (alert) => !readHandoverIds.has(alert.id)
                      ).length;
                      return unreadCount > 99 ? "99+" : unreadCount;
                    })()}
                  </span>
                )}

                {user?.role === "DOCTOR" && unreadDoctorAssignmentIds.size > 0 && (
                  <span className="notification-badge">
                    {unreadDoctorAssignmentIds.size > 99
                      ? "99+"
                      : unreadDoctorAssignmentIds.size}
                  </span>
                )}
              </button>

              {user?.role === "NURSE" && handoverToast && (
                <button
                  type="button"
                  className="handover-toast"
                  onClick={() => {
                    setHandoverToast(null);
                    setNotificationsOpen(false);
                    navigate("/patients");
                  }}
                  aria-label={`Doctor offline alert for ${handoverToast.patient_identifier}. Open patient assignments.`}
                >
                  <span className="handover-toast-icon"><Bell size={18} /></span>
                  <span className="handover-toast-content">
                    <strong>VITALIS · Handover alert</strong>
                    <span>Doctor offline · {handoverToast.patient_identifier}</span>
                    <span>{handoverToast.message}</span>
                    <span className="handover-toast-action">Open patient assignments →</span>
                  </span>
                  <span
                    className="handover-toast-close"
                    onClick={(event) => {
                      event.stopPropagation();
                      setHandoverToast(null);
                    }}
                    role="button"
                    aria-label="Dismiss handover alert"
                  >×</span>
                </button>
              )}

              {notificationsOpen && (
                <div
                  className="notification-panel"
                  role="dialog"
                  aria-label="Dashboard notifications"
                >
                  <div className="notification-panel-header">
                    <strong>Notifications</strong>
                    <button
                      type="button"
                      className="notification-close"
                      onClick={() => setNotificationsOpen(false)}
                      aria-label="Close notifications"
                    >
                      ×
                    </button>
                  </div>

                  {user?.role === "NURSE" && (
                    <>
                      <div className="notification-section-label">
                        Doctor handover alerts · {handoverAlerts.length}
                      </div>

                      {handoverAlerts.length === 0 ? (
                        <div className="notification-item">
                          <strong>No open handover alerts</strong>
                          <span>No active-patient doctor-offline alerts are currently available.</span>
                        </div>
                      ) : (
                        handoverAlerts.map((alert) => (
                          <div className="notification-item" key={`handover-${alert.id}`}>
                            <strong>Doctor offline · {alert.patient_identifier}</strong>
                            <span>{alert.message}</span>
                            <button
                              type="button"
                              className="notification-patient-link"
                              onClick={() => {
                                setNotificationsOpen(false);
                                navigate("/patients");
                              }}
                            >
                              Open patient assignments
                              <ChevronRight size={14} />
                            </button>
                          </div>
                        ))
                      )}
                    </>
                  )}

                  {user?.role === "NURSE" && handoverAlerts.length > 0 && (
                    <div className="notification-panel-footer">
                      <button
                        type="button"
                        className="notification-clear-all"
                        onClick={() => {
                          if (!user.id) return;

                          const nextDismissed = new Set([
                            ...dismissedHandoverIds,
                            ...handoverAlerts.map((alert) => alert.id),
                          ]);

                          setDismissedHandoverIds(nextDismissed);
                          localStorage.setItem(
                            `vitalis_dismissed_handover_alerts_${user.id}`,
                            JSON.stringify([...nextDismissed])
                          );
                          setHandoverAlerts([]);
                          setHandoverToast(null);
                        }}
                      >
                        Clear all
                      </button>
                    </div>
                  )}

                  {user?.role === "DOCTOR" && (
                    <>
                      <div className="notification-section-label">
                        New patient assignments · {doctorAssignmentNotifications.length}
                      </div>

                      {doctorAssignmentNotifications.length === 0 ? (
                        <div className="notification-item">
                          <strong>No new patient assignments</strong>
                          <span>New assignments will appear here while this dashboard is open.</span>
                        </div>
                      ) : (
                        [...doctorAssignmentNotifications].reverse().map((patient) => (
                          <div className="notification-item" key={`assignment-${patient.id}`}>
                            <strong>New patient assigned</strong>
                            <span>{patient.name} · {patient.patient_identifier}</span>
                            <button
                              type="button"
                              className="notification-patient-link"
                              onClick={() => {
                                setNotificationsOpen(false);
                                navigate(`/patients/${patient.id}`);
                              }}
                            >
                              View patient
                              <ChevronRight size={14} />
                            </button>
                          </div>
                        ))
                      )}

                      <div className="notification-panel-footer">
                        <button
                          type="button"
                          className="notification-clear-all"
                          onClick={() => {
                            setDoctorAssignmentNotifications([]);
                            setUnreadDoctorAssignmentIds(new Set());
                          }}
                          disabled={doctorAssignmentNotifications.length === 0}
                        >
                          Clear all
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

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
              {activePatients.length}
            </strong>

            <small>
              Active ICU patients
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
              Page refresh
            </span>

            <strong>
              10 sec
            </strong>

            <small>
              Dashboard refresh interval
            </small>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <RefreshCw size={19} />
            </div>

            <span>
              Risk data
            </span>

            <strong>
              Latest
            </strong>

            <small>
              Most recent available prediction
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

                  const riskStatus =
                    risk?.prediction === 1
                      ? "elevated"
                      : risk?.prediction === 0
                        ? "normal"
                        : "unavailable";

                  const riskValue =
                    risk?.ensemble_probability;

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

                        <span className={`risk-badge ${riskStatus}`}>
                          {riskStatus === "elevated"
                            ? "ELEVATED"
                            : riskStatus === "normal"
                              ? "NORMAL"
                              : "UNAVAILABLE"}
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
                            width: `${
                              riskValue == null
                                ? 0
                                : Math.min(
                                    100,
                                    Math.max(0, riskValue * 100),
                                  )
                            }%`,
                          }}
                        />
                      </div>

                      <div className="patient-card-footer">
                        <span>
                          {riskStatus === "elevated"
                            ? "Elevated risk"
                            : riskStatus === "normal"
                              ? "Normal prediction"
                              : "Risk data unavailable"}
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
