// qlang/GATE_VISUALS.ts

export type GateShape = "rect" | "circle" | "connector" | "measure";

export interface GateVisual {
  name: string;
  label: string;
  qubits: number;
  shape: GateShape;
  color: string;
  textColor?: string;
  size?: { width: number; height: number };
  offset?: number; // Para alinhar medidores ou ajustes finos
}

export const GATE_VISUALS: Record<string, GateVisual> = {
  h: {
    name: "h",
    label: "H",
    qubits: 1,
    shape: "rect",
    color: "#E2E8F0", // slate-200
  },
  x: {
    name: "x",
    label: "X",
    qubits: 1,
    shape: "rect",
    color: "#FECACA", // red-200
  },
  y: {
    name: "y",
    label: "Y",
    qubits: 1,
    shape: "rect",
    color: "#DBEAFE", // blue-100
  },
  z: {
    name: "z",
    label: "Z",
    qubits: 1,
    shape: "rect",
    color: "#DCFCE7", // green-100
  },
  s: {
    name: "s",
    label: "S",
    qubits: 1,
    shape: "rect",
    color: "#DDD6FE", // violet-200
  },
  t: {
    name: "t",
    label: "T",
    qubits: 1,
    shape: "rect",
    color: "#C4B5FD", // violet-300
  },
  rx: {
    name: "rx",
    label: "Rx",
    qubits: 1,
    shape: "rect",
    color: "#DBEAFE", // blue-100
  },
  ry: {
    name: "ry",
    label: "Ry",
    qubits: 1,
    shape: "rect",
    color: "#DBEAFE",
  },
  rz: {
    name: "rz",
    label: "Rz",
    qubits: 1,
    shape: "rect",
    color: "#DBEAFE",
  },
  u3: {
    name: "u3",
    label: "U3",
    qubits: 1,
    shape: "rect",
    color: "#FDE68A", // yellow-200
    size: { width: 34, height: 30 },
  },
  cnot: {
    name: "cnot",
    label: "●",
    qubits: 2,
    shape: "connector",
    color: "#4B5563", // gray-700
  },
  swap: {
    name: "swap",
    label: "×",
    qubits: 2,
    shape: "connector",
    color: "#4B5563",
  },
  m: {
    name: "m",
    label: "M",
    qubits: 1,
    shape: "measure",
    color: "#FBBF24", // yellow-400
    offset: 10,
  },
};
