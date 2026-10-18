import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import { useAuth } from "./auth/AuthContext";

import { lazy, Suspense } from "react";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Login = lazy(() => import("./pages/Login"));
const Patients = lazy(() => import("./pages/Patients"));
const PatientDetail = lazy(() => import("./pages/PatientDetail"));
const AdminDashboard = lazy(() => import("./pages/AdminDashboard"));

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-brand">
        VITALIS
      </div>

      <div className="loading-spinner" />

      <span>
        Loading clinical intelligence...
      </span>
    </div>
  );
}

function roleHome(role: string | undefined) {
  return role === "ADMIN" ? "/admin" : "/dashboard";
}

function RoleRoute({
  children,
  allowedRoles,
}: {
  children: React.ReactNode;
  allowedRoles: Array<"NURSE" | "DOCTOR" | "ADMIN">;
}) {
  const { user, loading } = useAuth();

  if (loading) return <LoadingScreen />;
  if (!user) return <Navigate to="/login" replace />;

  if (!allowedRoles.includes(user.role)) {
    return <Navigate to={roleHome(user.role)} replace />;
  }

  return children;
}

function PublicRoute({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, loading } = useAuth();

  if (loading) {
    return <LoadingScreen />;
  }

  if (user) {
    return (
      <Navigate
        to="/dashboard"
        replace
      />
    );
  }

  return children;
}

function HomeRedirect() {
  const { user, loading } = useAuth();

  if (loading) return <LoadingScreen />;

  return (
    <Navigate
      to={user ? roleHome(user.role) : "/login"}
      replace
    />
  );
}

export default function App() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <Routes>
      <Route
        path="/login"
        element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        }
      />

      <Route
        path="/dashboard"
        element={
          <RoleRoute allowedRoles={["NURSE", "DOCTOR"]}>
            <Dashboard />
          </RoleRoute>
        }
      />


      <Route
        path="/admin"
        element={
          <RoleRoute allowedRoles={["ADMIN"]}>
            <Suspense fallback={<LoadingScreen />}>
              <AdminDashboard />
            </Suspense>
          </RoleRoute>
        }
      />

      <Route
        path="/patients"
        element={
          <RoleRoute allowedRoles={["NURSE", "DOCTOR"]}>
            <Patients />
          </RoleRoute>
        }
      />

      <Route
        path="/patients/:id"
        element={
          <RoleRoute allowedRoles={["NURSE", "DOCTOR"]}>
            <PatientDetail />
          </RoleRoute>
        }
      />

      <Route
        path="/"
        element={
          <Navigate
            to="/login"
            replace
          />
        }
      />

      <Route
        path="*"
        element={<HomeRedirect />}
      />
      </Routes>
    </Suspense>
  );
}
