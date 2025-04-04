export type GateName = "h" | "x" | "y" | "z" | "cnot" | "m";

export interface QLangGate {
  name: GateName;
  targets: number[];
}

export interface CircuitModel {
  qubits: number;
  gates: QLangGate[];
}

export function parseQLang(code: string): CircuitModel {
  const lines = code
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("//"));

  let qubits = 0;
  const gates: QLangGate[] = [];

  for (const line of lines) {
    if (line.startsWith("create(")) {
      qubits = parseInt(line.match(/create\((\d+)\)/)?.[1] || "0");
      continue;
    }

    const match = line.match(/^(\w+)\((.*?)\)$/);
    if (!match) continue;

    const [, name, argStr] = match;
    const args = argStr.split(",").map((a) => parseInt(a.trim()));
    gates.push({ name: name as GateName, targets: args });
  }

  return { qubits, gates };
}
