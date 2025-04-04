import { GATE_VISUALS } from "./gateVisuals"; // ajuste o caminho conforme seu projeto
import type { GateVisual } from "./gateVisuals";

const DEFAULT_VISUAL: GateVisual = {
  name: "unknown",
  label: "?",
  qubits: 1,
  shape: "rect",
  color: "#E5E7EB", // gray-200
  textColor: "#111827", // gray-900
};

export function getGateVisual(name: string): GateVisual {
  const lower = name.toLowerCase();
  return GATE_VISUALS[lower] ?? { ...DEFAULT_VISUAL, name: lower };
}
