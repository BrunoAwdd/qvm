import init, { parse_circuit } from "../wasm/qlang_wasm";

export interface QLangGate {
  name: string;
  targets: number[];
  parameters: number[];
}

export interface CircuitModel {
  qubits: number;
  gates: QLangGate[];
  diagnostics: string[];
}

let initialization: Promise<unknown> | undefined;

export async function parseQLang(code: string): Promise<CircuitModel> {
  initialization ??= init();
  await initialization;
  return parse_circuit(code) as CircuitModel;
}
