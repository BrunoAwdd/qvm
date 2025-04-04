interface GateVisual {
  name: string;
  label: string;
  qubits: number;
  shape: "rect" | "circle" | "connector" | "measure";
  color: string;
  textColor?: string;
  size?: { width: number; height: number };
  offset?: number; // usado para alinhar ou deslocar visualmente
}

export const GATE_VISUALS: Record<string, GateVisual> = {
  h: {
    name: "h",
    label: "H",
    qubits: 1,
    shape: "rect",
    color: "#E2E8F0", // tailwind slate-200
  },
  x: {
    name: "x",
    label: "X",
    qubits: 1,
    shape: "rect",
    color: "#FECACA", // tailwind red-200
  },
  cnot: {
    name: "cnot",
    label: "●",
    qubits: 2,
    shape: "connector",
    color: "#4B5563", // dark circle
  },
  m: {
    name: "m",
    label: "M",
    qubits: 1,
    shape: "measure",
    color: "#FBBF24", // yellow-400
  },
};
