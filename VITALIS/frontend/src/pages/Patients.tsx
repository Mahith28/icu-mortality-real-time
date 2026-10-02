import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  ClipboardPlus,
  RefreshCw,
  Search,
  Stethoscope,
  UserPlus,
  Users,
  X,
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
  assignDoctor,
  createPatient,
  dischargePatient,
  getAssignments,
  getDoctors,
  getLatestRisk,
  getPatients,
  type Assignment,
  type DoctorOption,
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

function getInitials(name: string) {
  return name
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

type StatusFilter =
  | "ALL"
  | "ACTIVE"
  | "DISCHARGED";

interface PatientForm {
  patient_identifier: string;
  name: string;
  age: string;
  gender: string;
  race: string;
  insurance: string;
  admission_type: string;
  admission_location: string;
  icu_unit: string;
  bed_number: string;
  admission_diagnosis: string;
  height_cm: string;
  weight_kg: string;
}

const emptyForm: PatientForm = {
  patient_identifier: "",
  name: "",
  age: "",
  gender: "",
  race: "",
  insurance: "",
  admission_type: "EMERGENCY",
  admission_location: "",
  icu_unit: "",
  bed_number: "",
  admission_diagnosis: "",
  height_cm: "",
  weight_kg: "",
};

export default function Patients() {
  const navigate = useNavigate();

  const { user, logout } = useAuth();

  const isNurse =
    user?.role === "NURSE";

  const [patients, setPatients] =
    useState<Patient[]>([]);

  const [risks, setRisks] =
    useState<
      Record<
        number,
        RiskEvent | null
      >
    >({});

  const [assignments, setAssignments] =
    useState<
      Record<
        number,
        Assignment | null
      >
    >({});

  const [doctors, setDoctors] =
    useState<DoctorOption[]>([]);

  const [search, setSearch] =
    useState("");

  const [statusFilter, setStatusFilter] =
    useState<StatusFilter>("ALL");

  const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [error, setError] =
    useState("");

  const [modal, setModal] =
    useState<
      "create" |
      "assign" |
      null
    >(null);

  const [selectedPatient, setSelectedPatient] =
    useState<Patient | null>(null);

  const [form, setForm] =
    useState<PatientForm>(
      emptyForm,
    );

  const [selectedDoctor, setSelectedDoctor] =
    useState("");

  const [assignmentNotes, setAssignmentNotes] =
    useState("");

  const [saving, setSaving] =
    useState(false);

  const [actionError, setActionError] =
    useState("");

  async function loadPatients(
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

      const riskResults =
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
        Object.fromEntries(
          riskResults,
        ),
      );

      const assignmentResults =
        await Promise.all(
          patientList.map(
            async (patient) => {
              try {
                const result =
                  await getAssignments(
                    patient.id,
                  );

                const active =
                  result.find(
                    (item) =>
                      item.unassigned_at ===
                      null,
                  ) ?? null;

                return [
                  patient.id,
                  active,
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

      setAssignments(
        Object.fromEntries(
          assignmentResults,
        ),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load patients.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function loadDoctors() {
    if (!isNurse) {
      return;
    }

    try {
      const result =
        await getDoctors();

      setDoctors(result);
    } catch (err) {
      console.error(
        "Unable to load doctors:",
        err,
      );
    }
  }

  useEffect(() => {
    loadPatients();
    loadDoctors();

    const interval =
      window.setInterval(
        () => {
          loadPatients();
        },
        10000,
      );

    return () =>
      window.clearInterval(
        interval,
      );
  }, []);

  const filteredPatients =
    useMemo(() => {
      const query =
        search
          .trim()
          .toLowerCase();

      return patients.filter(
        (patient) => {
          const matchesStatus =
            statusFilter === "ALL" ||
            patient.status ===
              statusFilter;

          if (!matchesStatus) {
            return false;
          }

          if (!query) {
            return true;
          }

          return (
            patient.name
              .toLowerCase()
              .includes(query) ||
            patient.patient_identifier
              .toLowerCase()
              .includes(query) ||
            (
              patient.icu_unit ?? ""
            )
              .toLowerCase()
              .includes(query) ||
            (
              patient.bed_number ?? ""
            )
              .toLowerCase()
              .includes(query) ||
            (
              patient.admission_diagnosis ??
              ""
            )
              .toLowerCase()
              .includes(query)
          );
        },
      );
    }, [
      patients,
      search,
      statusFilter,
    ]);

  const activeCount =
    patients.filter(
      (patient) =>
        patient.status === "ACTIVE",
    ).length;

  const dischargedCount =
    patients.filter(
      (patient) =>
        patient.status ===
        "DISCHARGED",
    ).length;

  function handleLogout() {
    logout();

    navigate("/login", {
      replace: true,
    });
  }

  function openCreateModal() {
    setActionError("");
    setForm(emptyForm);
    setModal("create");
  }

  function openAssignModal(
    patient: Patient,
  ) {
    setSelectedPatient(patient);
    setActionError("");

    const current =
      assignments[patient.id];

    setSelectedDoctor(
      current
        ? String(current.doctor_id)
        : "",
    );

    setAssignmentNotes(
      current?.notes ?? "",
    );

    setModal("assign");
  }

  function closeModal() {
    if (saving) {
      return;
    }

    setModal(null);
    setSelectedPatient(null);
    setActionError("");
  }

  async function handleCreatePatient(
    event: React.FormEvent,
  ) {
    event.preventDefault();

    if (!form.patient_identifier.trim()) {
      setActionError(
        "Patient identifier is required.",
      );
      return;
    }

    if (!form.name.trim()) {
      setActionError(
        "Patient name is required.",
      );
      return;
    }

    if (!form.age) {
      setActionError(
        "Patient age is required.",
      );
      return;
    }

    if (!form.gender) {
      setActionError(
        "Gender is required.",
      );
      return;
    }

    try {
      setSaving(true);
      setActionError("");

      const created =
        await createPatient({
          patient_identifier:
            form.patient_identifier.trim(),
          name: form.name.trim(),
          age: Number(form.age),
          gender: form.gender.trim(),
          race:
            form.race.trim() || null,
          insurance:
            form.insurance.trim() || null,
          admission_type:
            form.admission_type.trim() ||
            null,
          admission_location:
            form.admission_location.trim() ||
            null,
          icu_unit:
            form.icu_unit.trim() || null,
          bed_number:
            form.bed_number.trim() || null,
          admission_diagnosis:
            form.admission_diagnosis.trim() ||
            null,
          height_cm:
            form.height_cm
              ? Number(form.height_cm)
              : null,
          weight_kg:
            form.weight_kg
              ? Number(form.weight_kg)
              : null,
        });

      setModal(null);
      setForm(emptyForm);

      await loadPatients();

      setSelectedPatient(
        created,
      );

      setSelectedDoctor("");
      setAssignmentNotes("");

      setModal("assign");
    } catch (err) {
      setActionError(
        err instanceof Error
          ? err.message
          : "Unable to create patient.",
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleAssignDoctor(
    event: React.FormEvent,
  ) {
    event.preventDefault();

    if (!selectedPatient) {
      return;
    }

    if (!selectedDoctor) {
      setActionError(
        "Please select a doctor.",
      );
      return;
    }

    try {
      setSaving(true);
      setActionError("");

      await assignDoctor(
        selectedPatient.id,
        Number(selectedDoctor),
        assignmentNotes,
      );

      setModal(null);
      setSelectedPatient(null);

      await loadPatients();
    } catch (err) {
      setActionError(
        err instanceof Error
          ? err.message
          : "Unable to assign doctor.",
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleDischarge(
    patient: Patient,
  ) {
    const confirmed =
      window.confirm(
        `Discharge ${patient.name} (${patient.patient_identifier})?`,
      );

    if (!confirmed) {
      return;
    }

    try {
      setRefreshing(true);
      setError("");

      await dischargePatient(
        patient.id,
      );

      await loadPatients(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to discharge patient.",
      );
    } finally {
      setRefreshing(false);
    }
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
            Sign out
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              PATIENT MANAGEMENT
            </span>

            <h1>
              Patients
            </h1>
          </div>

          <div
            style={{
              display: "flex",
              gap: 10,
              alignItems: "center",
            }}
          >
            {isNurse && (
              <button
                className="primary-action-button"
                onClick={
                  openCreateModal
                }
              >
                <UserPlus size={16} />
                Add patient
              </button>
            )}

            <button
              className="refresh-button"
              onClick={() =>
                loadPatients(true)
              }
              disabled={refreshing}
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
        </header>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <section className="patients-toolbar">
          <div>
            <h2>
              ICU patients
            </h2>

            <p>
              {filteredPatients.length} of{" "}
              {patients.length} patient
              {patients.length === 1
                ? ""
                : "s"}{" "}
              visible to you.
            </p>
          </div>

          <div
            style={{
              display: "flex",
              gap: 10,
              alignItems: "center",
              flexWrap: "wrap",
              justifyContent: "flex-end",
            }}
          >
            <div className="search-box">
              <Search size={17} />

              <input
                value={search}
                onChange={(
                  event,
                ) =>
                  setSearch(
                    event.target
                      .value,
                  )
                }
                placeholder="Search patients..."
              />
            </div>
          </div>
        </section>

        <section
          style={{
            display: "flex",
            gap: 8,
            marginBottom: 18,
            flexWrap: "wrap",
          }}
        >
          {(
            [
              [
                "ALL",
                "All patients",
                patients.length,
              ],
              [
                "ACTIVE",
                "Active",
                activeCount,
              ],
              [
                "DISCHARGED",
                "Discharged",
                dischargedCount,
              ],
            ] as const
          ).map(
            ([
              value,
              label,
              count,
            ]) => (
              <button
                key={value}
                type="button"
                onClick={() =>
                  setStatusFilter(
                    value,
                  )
                }
                style={{
                  display:
                    "inline-flex",
                  alignItems:
                    "center",
                  gap: 8,
                  border:
                    "1px solid var(--border, #1f2937)",
                  borderRadius: 8,
                  padding:
                    "8px 13px",
                  background:
                    statusFilter ===
                    value
                      ? "rgba(0, 229, 204, 0.10)"
                      : "transparent",
                  color:
                    statusFilter ===
                    value
                      ? "#00e5cc"
                      : "#8b98aa",
                  cursor:
                    "pointer",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {label}

                <span
                  style={{
                    padding:
                      "2px 6px",
                    borderRadius: 10,
                    background:
                      "rgba(255,255,255,0.06)",
                  }}
                >
                  {count}
                </span>
              </button>
            ),
          )}
        </section>

        {loading ? (
          <div className="dashboard-loading">
            <RefreshCw
              size={22}
              className="spin"
            />

            Loading patients...
          </div>
        ) : filteredPatients.length ===
          0 ? (
          <div className="dashboard-loading">
            <Users size={22} />

            {search
              ? "No patients match your search."
              : statusFilter !== "ALL"
                ? `No ${statusFilter.toLowerCase()} patients.`
                : "No patients available."}
          </div>
        ) : (
          <section className="patients-table-card">
            <div className="patients-table-header">
              <span>
                Patient
              </span>

              <span>
                Location
              </span>

              <span>
                Age / Gender
              </span>

              <span>
                Risk
              </span>

              <span>
                Status
              </span>

              {isNurse && (
                <span>
                  Actions
                </span>
              )}

              <span />
            </div>

            {filteredPatients.map(
              (patient) => {
                const risk =
                  risks[
                    patient.id
                  ];

                const assignment =
                  assignments[
                    patient.id
                  ];

                const elevated =
                  risk?.prediction ===
                  1;

                return (
                  <div
                    className="patient-row"
                    key={
                      patient.id
                    }
                  >
                    <button
                      type="button"
                      style={{
                        display:
                          "contents",
                      }}
                      onClick={() =>
                        navigate(
                          `/patients/${patient.id}`,
                        )
                      }
                    >
                      <div className="patient-row-name">
                        <div className="patient-avatar">
                          {getInitials(
                            patient.name,
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

                          {assignment && (
                            <small
                              style={{
                                display:
                                  "block",
                                marginTop:
                                  4,
                                color:
                                  "#64748b",
                              }}
                            >
                              <Stethoscope
                                size={11}
                                style={{
                                  verticalAlign:
                                    "middle",
                                  marginRight:
                                    4,
                                }}
                              />

                              {doctors.find(
                                (
                                  doctor,
                                ) =>
                                  doctor.id ===
                                  assignment.doctor_id,
                              )?.name ??
                                "Assigned doctor"}
                            </small>
                          )}
                        </div>
                      </div>

                      <div className="table-location">
                        <strong>
                          {
                            patient.icu_unit ??
                            "—"
                          }
                        </strong>

                        <span>
                          Bed{" "}
                          {
                            patient.bed_number ??
                            "—"
                          }
                        </span>
                      </div>

                      <div className="table-basic">
                        <strong>
                          {patient.age ??
                            "—"}
                        </strong>

                        <span>
                          {patient.gender ??
                            "—"}
                        </span>
                      </div>

                      <div className="table-risk">
                        <strong>
                          {percentage(
                            risk?.ensemble_probability,
                          )}
                        </strong>

                        <div className="mini-risk-bar">
                          <span
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(
                                  0,
                                  (
                                    risk?.ensemble_probability ??
                                    0
                                  ) * 100,
                                ),
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <span
                          className={
                            patient.status ===
                            "DISCHARGED"
                              ? "risk-badge normal"
                              : elevated
                                ? "risk-badge elevated"
                                : "risk-badge normal"
                          }
                        >
                          {patient.status ===
                          "DISCHARGED"
                            ? "DISCHARGED"
                            : risk
                              ? elevated
                                ? "ELEVATED"
                                : "NORMAL"
                              : "NO DATA"}
                        </span>
                      </div>
                    </button>

                    {isNurse && (
                      <div
                        style={{
                          display:
                            "flex",
                          gap: 6,
                          alignItems:
                            "center",
                        }}
                      >
                        {patient.status ===
                          "ACTIVE" && (
                          <>
                            <button
                              type="button"
                              title="Assign doctor"
                              onClick={() =>
                                openAssignModal(
                                  patient,
                                )
                              }
                              style={{
                                display:
                                  "inline-flex",
                                alignItems:
                                  "center",
                                gap: 6,
                                border:
                                  "1px solid rgba(0,229,204,0.18)",
                                borderRadius:
                                  7,
                                padding:
                                  "7px 9px",
                                background:
                                  "rgba(0,229,204,0.05)",
                                color:
                                  "#00e5cc",
                                cursor:
                                  "pointer",
                                fontSize:
                                  11,
                              }}
                            >
                              <Stethoscope
                                size={13}
                              />
                              Assign
                            </button>

                            <button
                              type="button"
                              title="Discharge patient"
                              onClick={() =>
                                handleDischarge(
                                  patient,
                                )
                              }
                              style={{
                                display:
                                  "inline-flex",
                                alignItems:
                                  "center",
                                gap: 6,
                                border:
                                  "1px solid rgba(239,68,68,0.20)",
                                borderRadius:
                                  7,
                                padding:
                                  "7px 9px",
                                background:
                                  "rgba(239,68,68,0.05)",
                                color:
                                  "#f87171",
                                cursor:
                                  "pointer",
                                fontSize:
                                  11,
                              }}
                            >
                              <CheckCircle2
                                size={13}
                              />
                              Discharge
                            </button>
                          </>
                        )}
                      </div>
                    )}

                    <button
                      type="button"
                      aria-label={`Open ${patient.name}`}
                      onClick={() =>
                        navigate(
                          `/patients/${patient.id}`,
                        )
                      }
                      style={{
                        border:
                          "none",
                        background:
                          "transparent",
                        color:
                          "#64748b",
                        cursor:
                          "pointer",
                        display:
                          "flex",
                        alignItems:
                          "center",
                        justifyContent:
                          "center",
                      }}
                    >
                      <ChevronRight
                        size={18}
                      />
                    </button>
                  </div>
                );
              },
            )}
          </section>
        )}
      </main>

      {modal === "create" && (
        <div
          className="vitalis-modal-backdrop"
          onMouseDown={(
            event,
          ) => {
            if (
              event.target ===
              event.currentTarget
            ) {
              closeModal();
            }
          }}
        >
          <div className="vitalis-modal large">
            <div className="vitalis-modal-header">
              <div>
                <span className="eyebrow">
                  PATIENT REGISTRATION
                </span>

                <h2>
                  Add ICU patient
                </h2>

                <p>
                  Create the patient
                  record and initial
                  ICU stay.
                </p>
              </div>

              <button
                type="button"
                className="modal-close"
                onClick={
                  closeModal
                }
              >
                <X size={18} />
              </button>
            </div>

            {actionError && (
              <div className="error-banner">
                <AlertTriangle
                  size={15}
                />
                {actionError}
              </div>
            )}

            <form
              className="patient-form"
              onSubmit={
                handleCreatePatient
              }
            >
              <div className="form-section-title">
                Identity
              </div>

              <div className="form-grid">
                <label>
                  Patient ID *
                  <input
                    value={
                      form.patient_identifier
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        patient_identifier:
                          event.target
                            .value,
                      })
                    }
                    placeholder="VTL-ICU-003"
                    required
                  />
                </label>

                <label>
                  Full name *
                  <input
                    value={
                      form.name
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        name: event
                          .target
                          .value,
                      })
                    }
                    placeholder="Patient name"
                    required
                  />
                </label>

                <label>
                  Age *
                  <input
                    type="number"
                    min="0"
                    max="120"
                    value={
                      form.age
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        age: event
                          .target
                          .value,
                      })
                    }
                    required
                  />
                </label>

                <label>
                  Gender *
                  <select
                    value={
                      form.gender
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        gender:
                          event.target
                            .value,
                      })
                    }
                    required
                  >
                    <option value="">
                      Select gender
                    </option>
                    <option value="Male">
                      Male
                    </option>
                    <option value="Female">
                      Female
                    </option>
                    <option value="Other">
                      Other
                    </option>
                  </select>
                </label>

                <label>
                  Race
                  <input
                    value={
                      form.race
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        race: event
                          .target
                          .value,
                      })
                    }
                    placeholder="e.g. WHITE"
                  />
                </label>

                <label>
                  Insurance
                  <input
                    value={
                      form.insurance
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        insurance:
                          event.target
                            .value,
                      })
                    }
                    placeholder="Insurance provider"
                  />
                </label>
              </div>

              <div className="form-section-title">
                Admission
              </div>

              <div className="form-grid">
                <label>
                  Admission type
                  <select
                    value={
                      form.admission_type
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        admission_type:
                          event.target
                            .value,
                      })
                    }
                  >
                    <option value="EMERGENCY">
                      Emergency
                    </option>
                    <option value="ELECTIVE">
                      Elective
                    </option>
                    <option value="URGENT">
                      Urgent
                    </option>
                    <option value="NEWBORN">
                      Newborn
                    </option>
                  </select>
                </label>

                <label>
                  Admission location
                  <input
                    value={
                      form.admission_location
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        admission_location:
                          event.target
                            .value,
                      })
                    }
                    placeholder="Emergency room"
                  />
                </label>

                <label>
                  ICU unit
                  <input
                    value={
                      form.icu_unit
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        icu_unit:
                          event.target
                            .value,
                      })
                    }
                    placeholder="Medical ICU"
                  />
                </label>

                <label>
                  Bed number
                  <input
                    value={
                      form.bed_number
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        bed_number:
                          event.target
                            .value,
                      })
                    }
                    placeholder="MICU-03"
                  />
                </label>

                <label className="form-span-2">
                  Admission diagnosis
                  <textarea
                    value={
                      form.admission_diagnosis
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        admission_diagnosis:
                          event.target
                            .value,
                      })
                    }
                    placeholder="Primary admission diagnosis"
                    rows={3}
                  />
                </label>
              </div>

              <div className="form-section-title">
                Measurements
              </div>

              <div className="form-grid">
                <label>
                  Height (cm)
                  <input
                    type="number"
                    min="0"
                    value={
                      form.height_cm
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        height_cm:
                          event.target
                            .value,
                      })
                    }
                  />
                </label>

                <label>
                  Weight (kg)
                  <input
                    type="number"
                    min="0"
                    value={
                      form.weight_kg
                    }
                    onChange={(
                      event,
                    ) =>
                      setForm({
                        ...form,
                        weight_kg:
                          event.target
                            .value,
                      })
                    }
                  />
                </label>
              </div>

              <div className="modal-actions">
                <button
                  type="button"
                  className="secondary-action-button"
                  onClick={
                    closeModal
                  }
                  disabled={saving}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="primary-action-button"
                  disabled={saving}
                >
                  {saving ? (
                    <>
                      <RefreshCw
                        size={15}
                        className="spin"
                      />
                      Creating...
                    </>
                  ) : (
                    <>
                      <ClipboardPlus
                        size={15}
                      />
                      Create patient
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {modal === "assign" &&
        selectedPatient && (
          <div
            className="vitalis-modal-backdrop"
            onMouseDown={(
              event,
            ) => {
              if (
                event.target ===
                event.currentTarget
              ) {
                closeModal();
              }
            }}
          >
            <div className="vitalis-modal">
              <div className="vitalis-modal-header">
                <div>
                  <span className="eyebrow">
                    CARE TEAM
                  </span>

                  <h2>
                    Assign doctor
                  </h2>

                  <p>
                    {selectedPatient.name}{" "}
                    ·{" "}
                    {
                      selectedPatient.patient_identifier
                    }
                  </p>
                </div>

                <button
                  type="button"
                  className="modal-close"
                  onClick={
                    closeModal
                  }
                >
                  <X size={18} />
                </button>
              </div>

              {actionError && (
                <div className="error-banner">
                  <AlertTriangle
                    size={15}
                  />
                  {actionError}
                </div>
              )}

              <form
                className="patient-form"
                onSubmit={
                  handleAssignDoctor
                }
              >
                <label>
                  Attending doctor *
                  <select
                    value={
                      selectedDoctor
                    }
                    onChange={(
                      event,
                    ) =>
                      setSelectedDoctor(
                        event.target
                          .value,
                      )
                    }
                    required
                  >
                    <option value="">
                      Select doctor
                    </option>

                    {doctors.map(
                      (
                        doctor,
                      ) => (
                        <option
                          key={
                            doctor.id
                          }
                          value={
                            doctor.id
                          }
                        >
                          {doctor.name}{" "}
                          —{" "}
                          {
                            doctor.email
                          }
                        </option>
                      ),
                    )}
                  </select>
                </label>

                <label>
                  Assignment notes
                  <textarea
                    value={
                      assignmentNotes
                    }
                    onChange={(
                      event,
                    ) =>
                      setAssignmentNotes(
                        event.target
                          .value,
                      )
                    }
                    placeholder="Optional care-team notes"
                    rows={4}
                  />
                </label>

                {doctors.length ===
                  0 && (
                  <div className="dashboard-loading compact">
                    <Stethoscope
                      size={18}
                    />
                    No active doctors
                    are currently
                    available.
                  </div>
                )}

                <div className="modal-actions">
                  <button
                    type="button"
                    className="secondary-action-button"
                    onClick={
                      closeModal
                    }
                    disabled={
                      saving
                    }
                  >
                    Cancel
                  </button>

                  <button
                    type="submit"
                    className="primary-action-button"
                    disabled={
                      saving ||
                      doctors.length ===
                        0
                    }
                  >
                    {saving ? (
                      <>
                        <RefreshCw
                          size={15}
                          className="spin"
                        />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Stethoscope
                          size={15}
                        />
                        Assign doctor
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
    </div>
  );
}