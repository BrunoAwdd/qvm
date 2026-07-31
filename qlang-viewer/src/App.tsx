import { useEffect, useState } from "react";
import QLangCircuit from "./components/QLangCircuit";
import { type CircuitModel, parseQLang } from "./qlang/parseQLang";

const defaultCode = `create(3)
h(0)
cnot(0, 1)
m()`;

const emptyCircuit: CircuitModel = { qubits: 0, gates: [], diagnostics: [] };

export default function App() {
  const [code, setCode] = useState(defaultCode);
  const [circuit, setCircuit] = useState<CircuitModel>(emptyCircuit);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let active = true;
    parseQLang(code)
      .then((model) => {
        if (active) {
          setCircuit(model);
          setError(undefined);
        }
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => {
      active = false;
    };
  }, [code]);

  return (
    <main className="p-4 space-y-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold">QLang Visualizer</h1>
      <p className="text-sm text-slate-600">
        Parsed by the same Rust syntax crate used by the QLang runtime.
      </p>

      <textarea
        aria-label="QLang source"
        className="w-full h-40 p-2 border rounded font-mono"
        value={code}
        onChange={(event) => setCode(event.target.value)}
      />

      {error ? (
        <pre role="alert" className="p-3 rounded bg-red-50 text-red-800 whitespace-pre-wrap">
          {error}
        </pre>
      ) : (
        <QLangCircuit circuit={circuit} />
      )}

      {circuit.diagnostics.length > 0 && (
        <ul className="p-3 rounded bg-amber-50 text-amber-900 text-sm">
          {circuit.diagnostics.map((diagnostic, index) => (
            <li key={`${diagnostic}-${index}`}>{diagnostic}</li>
          ))}
        </ul>
      )}
    </main>
  );
}
