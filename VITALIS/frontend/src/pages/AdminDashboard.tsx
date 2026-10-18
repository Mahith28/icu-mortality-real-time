import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  CircleUserRound,
  Building2,
  KeyRound,
  LogOut,
  RefreshCw,
  ShieldCheck,
  UserRound,
  Users,
  UserPlus,
  ArrowRightLeft,
  UserX,
  Check,
  X,
} from "lucide-react";
import { useAuth } from "../auth/AuthContext";
import { ThemeToggle } from "../theme/ThemeContext";
import {
  assignAdminStaffBranches,
  createAdminBranch,
  deactivateAdminBranch,
  reactivateAdminBranch,
  createAdminStaff,
  deactivateAdminStaff,
  getAdminBranches,
  getAdminStaff,
  getAdminStaffBranches,
  resetAdminStaffPassword,
  type AdminBranch,
  type AdminStaff,
} from "../api/client";

type Section = "NURSE" | "DOCTOR" | "BRANCHES";
type StaffForm = { name: string; email: string; password: string };

export default function AdminDashboard() {
  const { user, logout } = useAuth();
  const [section, setSection] = useState<Section>("NURSE");
  const [staff, setStaff] = useState<AdminStaff[]>([]);
  const [branches, setBranches] = useState<AdminBranch[]>([]);
  const [memberships, setMemberships] = useState<Record<number, number[]>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [branchFilter, setBranchFilter] = useState("ALL");
  const [staffSearch, setStaffSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [branchCreateOpen, setBranchCreateOpen] = useState(false);
  const [staffForm, setStaffForm] = useState<StaffForm>({
    name: "",
    email: "",
    password: "",
  });
  const [createBranchIds, setCreateBranchIds] = useState<number[]>([]);
  const [branchForm, setBranchForm] = useState({
    code: "",
    name: "",
    location: "",
  });
  const [editingStaff, setEditingStaff] = useState<number | null>(null);
  const [editBranchIds, setEditBranchIds] = useState<number[]>([]);
  const [passwordStaff, setPasswordStaff] = useState<number | null>(null);
  const [newPassword, setNewPassword] = useState("");

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [staffData, branchData] = await Promise.all([
        getAdminStaff(),
        getAdminBranches(true),
      ]);
      const entries = await Promise.all(
        staffData.map(async (person) => {
          const assigned = await getAdminStaffBranches(person.id);
          return [person.id, assigned.map((item) => item.branch_id)] as const;
        }),
      );
      setStaff(staffData);
      setBranches(branchData);
      setMemberships(Object.fromEntries(entries));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load administration data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const activeBranches = useMemo(
    () => branches.filter((branch) => branch.is_active),
    [branches],
  );

  const currentRole = section === "NURSE" || section === "DOCTOR" ? section : null;
  const roleStaff = currentRole ? staff.filter((person) => person.role === currentRole) : [];
  const filteredStaff = roleStaff.filter((person) => {
    const ids = memberships[person.id] ?? [];
    const matchesBranch =
      branchFilter === "UNASSIGNED"
        ? ids.length === 0
        : branchFilter === "ALL"
          ? true
          : ids.includes(Number(branchFilter));

    const query = staffSearch.trim().toLowerCase();
    const matchesSearch =
      !query ||
      person.name.toLowerCase().includes(query) ||
      person.email.toLowerCase().includes(query);

    return matchesBranch && matchesSearch;
  });

  function toggleId(id: number, checked: boolean, values: number[], setter: (next: number[]) => void) {
    setter(checked
      ? [...values.filter((value) => value !== id), id]
      : values.filter((value) => value !== id));
  }

  async function runOperation(operation: () => Promise<unknown>, message: string) {
    setSaving(true);
    setError("");
    setNotice("");
    try {
      await operation();
      setNotice(message);
      setEditingStaff(null);
      setPasswordStaff(null);
      setNewPassword("");
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "The operation failed.");
    } finally {
      setSaving(false);
    }
  }

  async function submitStaff(event: FormEvent) {
    event.preventDefault();
    const email = staffForm.email.trim().toLowerCase();

    if (activeBranches.length === 0) {
      setError("Create an active hospital branch before creating a staff account.");
      return;
    }

    if (createBranchIds.length === 0) {
      setError("Select at least one branch for this staff member.");
      return;
    }

    const expectedDomain = section === "NURSE" ? "nurse.vitalis.com" : "doctor.vitalis.com";
    if (email.split("@")[1] !== expectedDomain) {
      setError(`Use an email ending in @${expectedDomain}.`);
      return;
    }
    await runOperation(
      async () => {
        const created = await createAdminStaff({
          ...staffForm,
          name: staffForm.name.trim(),
          email,
          branch_ids: createBranchIds,
        });
        setStaffForm({ name: "", email: "", password: "" });
        setCreateBranchIds([]);
        setCreateOpen(false);
        return created;
      },
      `${section === "NURSE" ? "Nurse" : "Doctor"} account created.`,
    );
  }

  async function submitBranch(event: FormEvent) {
    event.preventDefault();
    await runOperation(
      () => createAdminBranch({
        code: branchForm.code.trim(),
        name: branchForm.name.trim(),
        location: branchForm.location.trim() || null,
      }),
      "Branch created.",
    );
    setBranchForm({ code: "", name: "", location: "" });
  }

  async function removeStaff(person: AdminStaff) {
    const confirmed = window.confirm(
      `Remove ${person.name} from the hospital?\n\nTheir account will be deactivated and they will no longer be able to sign in. Historical records will be retained. Doctors with active patient assignments must have those patients reassigned first.`,
    );
    if (!confirmed) return;
    await runOperation(
      () => deactivateAdminStaff(person.id),
      `${person.name}'s account has been deactivated.`,
    );
  }

  async function changeBranchStatus(branch: AdminBranch) {
    const confirmed = window.confirm(
      branch.is_active
        ? `Deactivate ${branch.name}?\n\nActive patients must be transferred or their care status resolved first. Historical records and staff memberships will be retained.`
        : `Reactivate ${branch.name}?\n\nThe branch will become available for new assignments and patient creation again.`,
    );

    if (!confirmed) return;

    await runOperation(
      () =>
        branch.is_active
          ? deactivateAdminBranch(branch.id)
          : reactivateAdminBranch(branch.id),
      `${branch.name} has been ${branch.is_active ? "deactivated" : "reactivated"}.`,
    );
  }

  const title = section === "NURSE" ? "Nurses" : section === "DOCTOR" ? "Doctors" : "Branches";

  return (
    <div className="vitalis-page admin-shell">
      <header className="vitalis-topbar">
        <div className="vitalis-brand">
          <div className="vitalis-brand-mark"><ShieldCheck size={21} /></div>
          <div><strong>VITALIS</strong><span>Administration</span></div>
        </div>
        <div className="vitalis-topbar-actions admin-header-actions">
          <ThemeToggle />
          <button type="button" className="icon-button" onClick={() => void loadData()} disabled={loading} aria-label="Refresh" title="Refresh">
            <RefreshCw size={16} />
          </button>
          <div className="profile-chip">
            <CircleUserRound size={19} />
            <div>
              <strong>{user?.name}</strong>
              <span>ADMIN</span>
            </div>
          </div>
        </div>
      </header>

      <div className="admin-workspace">
        <aside className="admin-sidebar" aria-label="Administration navigation">
          <span className="admin-sidebar-label">MANAGEMENT</span>
          <button className={section === "NURSE" ? "active" : ""} onClick={() => { setSection("NURSE"); setBranchFilter("ALL"); }}>
            <Users size={18} /> Nurses
          </button>
          <button className={section === "DOCTOR" ? "active" : ""} onClick={() => { setSection("DOCTOR"); setBranchFilter("ALL"); }}>
            <UserRound size={18} /> Doctors
          </button>
          <button className={section === "BRANCHES" ? "active" : ""} onClick={() => setSection("BRANCHES")}>
            <Building2 size={18} /> Branches
          </button>
          <div className="sidebar-bottom">
            <div className="system-status">
              <span className="status-pulse" />
              <div>
                <strong>Service status</strong>
                <span>Administration active</span>
              </div>
            </div>

            <button type="button" className="sign-out-button" onClick={logout}>
              <LogOut size={16} />
              Sign out
            </button>
          </div>
        </aside>

        <main className="vitalis-content admin-main">
          <div className="vitalis-page-heading">
            <span className="eyebrow">SYSTEM MANAGEMENT</span>
            <h1>{title}</h1>
            <p>Manage staff access, branch memberships, and hospital branches.</p>
          </div>

          {error && <div className="error-banner" role="alert">{error}</div>}
          {notice && <div className="admin-notice" role="status">{notice}</div>}

          {loading ? (
            <section className="admin-panel">Loading administration data…</section>
          ) : section === "BRANCHES" ? (
            <div className="admin-branches-page">
              <div className="admin-summary-grid">
                <div className="admin-summary-card">
                  <Building2 size={19} />
                  <span>Total branches</span>
                  <strong>{branches.length}</strong>
                </div>
                <div className="admin-summary-card">
                  <Check size={19} />
                  <span>Active branches</span>
                  <strong>{branches.filter((branch) => branch.is_active).length}</strong>
                </div>
                <div className="admin-summary-card">
                  <Users size={19} />
                  <span>Staff assigned to branches</span>
                  <strong>{staff.filter((person) => person.is_active && (memberships[person.id] ?? []).length > 0).length}</strong>
                </div>
              </div>

              <section className="admin-panel admin-branches-directory">
                <div className="admin-directory-heading">
                  <div>
                    <h2>Hospital branches</h2>
                    <p>View branch details and the number of staff assigned to each location.</p>
                  </div>
                  <button
                    type="button"
                    className="auth-submit admin-inline-button"
                    onClick={() => { setBranchCreateOpen(!branchCreateOpen); setError(""); }}
                  >
                    <Building2 size={16} />
                    {branchCreateOpen ? "Close form" : "Add branch"}
                  </button>
                </div>

                {branchCreateOpen && (
                  <form className="admin-create-form admin-branch-create-form" onSubmit={(event) => void submitBranch(event)}>
                    <label>Branch code
                      <input required minLength={2} maxLength={30} autoComplete="off" placeholder="e.g. MAIN-01" value={branchForm.code} onChange={(event) => setBranchForm({ ...branchForm, code: event.target.value })} />
                    </label>
                    <label>Branch name
                      <input required minLength={2} maxLength={120} placeholder="Enter branch name" value={branchForm.name} onChange={(event) => setBranchForm({ ...branchForm, name: event.target.value })} />
                    </label>
                    <label className="admin-field-wide">Location
                      <input placeholder="City or address (optional)" value={branchForm.location} onChange={(event) => setBranchForm({ ...branchForm, location: event.target.value })} />
                    </label>
                    <div className="admin-field-wide admin-editor-actions">
                      <button className="auth-submit" disabled={saving}>{saving ? "Creating…" : "Create branch"}</button>
                      <button type="button" className="admin-action-button" onClick={() => setBranchCreateOpen(false)}>Cancel</button>
                    </div>
                  </form>
                )}

                {branches.length === 0 ? (
                  <div className="admin-branches-empty">
                    <Building2 size={28} />
                    <strong>No branches yet</strong>
                    <p>Create a hospital branch to start assigning staff.</p>
                  </div>
                ) : (
                  <div className="admin-branch-grid">
                    {branches.map((branch) => {
                      const assignedStaff = staff.filter(
                        (person) => person.is_active && (memberships[person.id] ?? []).includes(branch.id),
                      );
                      return (
                        <article className="admin-branch-card" key={branch.id}>
                          <div className="admin-branch-card-top">
                            <div className="admin-branch-icon"><Building2 size={20} /></div>
                            <span className={`admin-status ${branch.is_active ? "active" : ""}`}>
                              {branch.is_active ? "Active" : "Inactive"}
                            </span>
                          </div>
                          <div className="admin-branch-card-title">
                            <h3>{branch.name}</h3>
                            <span>{branch.code}</span>
                          </div>
                          <p className="admin-branch-location">{branch.location || "Location not provided"}</p>
                          <div className="admin-branch-card-footer">
                            <div>
                              <Users size={16} />
                              <strong>{assignedStaff.length}</strong>
                              <span>active staff assigned</span>
                            </div>
                            <button
                              type="button"
                              className="admin-action-button"
                              disabled={saving}
                              onClick={() => void changeBranchStatus(branch)}
                            >
                              {branch.is_active ? "Deactivate branch" : "Reactivate branch"}
                            </button>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>
            </div>
          ) : (
            <>
              <div className="admin-summary-grid">
                <div className="admin-summary-card"><Users size={19} /><span>Total {title.toLowerCase()}</span><strong>{roleStaff.length}</strong></div>
                <div className="admin-summary-card"><Check size={19} /><span>Active accounts</span><strong>{roleStaff.filter((person) => person.is_active).length}</strong></div>
                <div className="admin-summary-card"><Building2 size={19} /><span>Unassigned</span><strong>{roleStaff.filter((person) => (memberships[person.id] ?? []).length === 0).length}</strong></div>
              </div>

              <section className="admin-panel admin-directory">
                <div className="admin-directory-heading">
                  <div>
                    <h2>{title} directory</h2>
                    <p>Filter by branch or manage an individual staff account.</p>
                  </div>
                  <button type="button" className="auth-submit admin-inline-button" onClick={() => { setCreateOpen(!createOpen); setError(""); }}>
                    <UserPlus size={16} /> {createOpen ? "Close form" : `Create ${section === "NURSE" ? "nurse" : "doctor"}`}
                  </button>
                </div>

                {createOpen && (
                  <form className="admin-create-form patient-form" onSubmit={(event) => void submitStaff(event)}>
                    <label>Full name
                      <input required minLength={2} maxLength={120} value={staffForm.name} onChange={(event) => setStaffForm({ ...staffForm, name: event.target.value })} />
                    </label>
                    <label>Work email
                      <input required type="email" value={staffForm.email} onChange={(event) => setStaffForm({ ...staffForm, email: event.target.value })} />
                    </label>
                    <label>Initial password
                      <input required type="password" minLength={8} maxLength={128} value={staffForm.password} onChange={(event) => setStaffForm({ ...staffForm, password: event.target.value })} />
                    </label>
                    <div className="admin-field-wide admin-branch-selection">
                      <div className="admin-branch-selection-heading">
                        <span className="admin-field-title">Assign hospital branch</span>
                        <span className="admin-required-label">Required</span>
                      </div>
                      <p className="admin-branch-selection-help">
                        Select one or more branches this staff member is authorized to access.
                      </p>
                      {activeBranches.length > 0 ? (
                        <div className="admin-branch-choice-grid">
                          {activeBranches.map((branch) => {
                            const selected = createBranchIds.includes(branch.id);
                            return (
                              <label
                                className={`admin-branch-choice${selected ? " selected" : ""}`}
                                key={branch.id}
                              >
                                <input
                                  type="checkbox"
                                  checked={selected}
                                  onChange={(event) => toggleId(branch.id, event.target.checked, createBranchIds, setCreateBranchIds)}
                                />
                                <span className="admin-branch-choice-icon"><Building2 size={19} /></span>
                                <span className="admin-branch-choice-copy">
                                  <strong>{branch.name}</strong>
                                  <small>{branch.code}</small>
                                </span>
                                <span className="admin-branch-choice-check" aria-hidden="true">
                                  {selected && <Check size={15} />}
                                </span>
                              </label>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="admin-branch-selection-empty">
                          Create an active branch on the Branches page before creating staff accounts.
                        </p>
                      )}
                      {activeBranches.length > 0 && createBranchIds.length === 0 && (
                        <p className="admin-branch-selection-error">Select at least one branch to continue.</p>
                      )}
                    </div>
                    <button className="auth-submit admin-field-wide" disabled={saving}>{saving ? "Creating…" : "Create account"}</button>
                  </form>
                )}

                <div className="admin-filter-row">
                  <label className="admin-search-field" htmlFor="staff-search">
                    <span>Search staff</span>
                    <input
                      id="staff-search"
                      type="search"
                      placeholder="Search name or email"
                      value={staffSearch}
                      onChange={(event) => setStaffSearch(event.target.value)}
                    />
                  </label>
                  <label className="admin-branch-filter" htmlFor="staff-branch-filter">
                    <span>Branch</span>
                    <select id="staff-branch-filter" value={branchFilter} onChange={(event) => setBranchFilter(event.target.value)}>
                      <option value="ALL">All branches</option>
                      {activeBranches.map((branch) => <option key={branch.id} value={branch.id}>{branch.name} ({branch.code})</option>)}
                      <option value="UNASSIGNED">Unassigned</option>
                    </select>
                  </label>
                  <span className="admin-result-count">{filteredStaff.length} of {roleStaff.length} staff</span>
                </div>

                <div className="admin-staff-card-grid">
                  {filteredStaff.map((person) => {
                    const assignedIds = memberships[person.id] ?? [];
                    const assignedBranches = branches.filter((branch) => assignedIds.includes(branch.id));

                    return (
                      <article className="admin-staff-card" key={person.id}>
                        <div className="admin-staff-card-header">
                          <div className="admin-staff-identity">
                            <div className="admin-staff-avatar" aria-hidden="true">
                              <UserRound size={21} />
                            </div>
                            <div className="admin-staff-name">
                              <h3>{person.name}</h3>
                              <p>{person.email}</p>
                            </div>
                          </div>
                          <span className={`admin-status ${person.is_active ? "active" : ""}`}>
                            {person.is_active ? "Active" : "Inactive"}
                          </span>
                        </div>

                        <div className="admin-staff-branches">
                          <div className="admin-staff-section-label">
                            <Building2 size={15} />
                            <span>Branch memberships</span>
                          </div>
                          {assignedBranches.length > 0 ? (
                            <div className="admin-staff-branch-list">
                              {assignedBranches.map((branch) => (
                                <div className="admin-staff-branch-item" key={branch.id}>
                                  <span className="admin-staff-branch-icon"><Building2 size={17} /></span>
                                  <span className="admin-staff-branch-copy">
                                    <strong>{branch.name}</strong>
                                    <small>{branch.code}</small>
                                  </span>
                                  <Check size={15} className="admin-staff-branch-check" />
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="admin-staff-unassigned">
                              <span>Unassigned</span>
                              <small>No branch access assigned</small>
                            </div>
                          )}
                        </div>

                        {person.is_active && (
                          <div className="admin-staff-actions">
                            <button
                              type="button"
                              className="admin-action-button"
                              onClick={() => {
                                setEditingStaff(editingStaff === person.id ? null : person.id);
                                setEditBranchIds(assignedIds);
                                setPasswordStaff(null);
                              }}
                            >
                              <ArrowRightLeft size={14} />
                              Move branch
                            </button>
                            <button
                              type="button"
                              className="admin-action-button"
                              onClick={() => {
                                setPasswordStaff(passwordStaff === person.id ? null : person.id);
                                setNewPassword("");
                                setEditingStaff(null);
                              }}
                            >
                              <KeyRound size={14} />
                              Reset password
                            </button>
                            <button
                              type="button"
                              className="admin-action-button danger"
                              disabled={saving}
                              onClick={() => void removeStaff(person)}
                            >
                              <UserX size={14} />
                              Remove staff
                            </button>
                          </div>
                        )}

                        {editingStaff === person.id && person.is_active && (
                          <form className="admin-inline-editor" onSubmit={(event) => {
                            event.preventDefault();
                            void runOperation(
                              () => assignAdminStaffBranches(person.id, editBranchIds),
                              `Branch memberships updated for ${person.name}.`,
                            );
                          }}>
                            <div className="admin-assignment-heading">
                              <div>
                                <strong>Set authorized branches</strong>
                                <p>Select the hospital branches this staff member can access.</p>
                              </div>
                              <span>{editBranchIds.length} selected</span>
                            </div>
                            <div className="admin-assignment-grid">
                              {activeBranches.map((branch) => {
                                const selected = editBranchIds.includes(branch.id);
                                return (
                                  <label
                                    className={`admin-assignment-choice${selected ? " selected" : ""}`}
                                    key={branch.id}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={selected}
                                      onChange={(event) => toggleId(branch.id, event.target.checked, editBranchIds, setEditBranchIds)}
                                    />
                                    <span className="admin-assignment-icon"><Building2 size={19} /></span>
                                    <span className="admin-assignment-copy">
                                      <strong>{branch.name}</strong>
                                      <small>{branch.code}</small>
                                    </span>
                                    <span className="admin-assignment-indicator" aria-hidden="true">
                                      {selected && <Check size={14} />}
                                    </span>
                                  </label>
                                );
                              })}
                              {activeBranches.length === 0 && <p>No active branches are available.</p>}
                            </div>
                            <p>Save with no branches selected to make this staff member unassigned.</p>
                            <div className="admin-editor-actions">
                              <button type="submit" className="auth-submit" disabled={saving}>Save branches</button>
                              <button type="button" className="admin-action-button" onClick={() => setEditingStaff(null)}>
                                <X size={14} /> Cancel
                              </button>
                            </div>
                          </form>
                        )}

                        {passwordStaff === person.id && person.is_active && (
                          <form className="admin-inline-editor" onSubmit={(event) => {
                            event.preventDefault();
                            if (newPassword.length < 8) {
                              setError("The new password must be at least 8 characters.");
                              return;
                            }
                            void runOperation(
                              () => resetAdminStaffPassword(person.id, newPassword),
                              `Password reset for ${person.name}.`,
                            );
                          }}>
                            <strong>Set a new password</strong>
                            <label>New password
                              <input
                                type="password"
                                required
                                minLength={8}
                                maxLength={128}
                                autoComplete="new-password"
                                value={newPassword}
                                onChange={(event) => setNewPassword(event.target.value)}
                              />
                            </label>
                            <div className="admin-editor-actions">
                              <button type="submit" className="auth-submit" disabled={saving}>Save password</button>
                              <button type="button" className="admin-action-button" onClick={() => setPasswordStaff(null)}>
                                <X size={14} /> Cancel
                              </button>
                            </div>
                          </form>
                        )}
                      </article>
                    );
                  })}
                  {filteredStaff.length === 0 && (
                    <div className="admin-staff-empty">
                      <Users size={26} />
                      <strong>No matching staff</strong>
                      <p>Try another search term or branch filter.</p>
                    </div>
                  )}
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
