import React, { useState } from "react";
import QLangCircuit from "./components/QLangCircuit";
import { parseQLang } from "./qlang/parseQLang";

const defaultCode = `create(3)
h(0)
cnot(0,1)
m()`;

export default function App() {
  const [code, setCode] = useState(defaultCode);
  const circuit = parseQLang(code);

  return (
    <div className="p-4 space-y-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold">🔍 QLang Visualizer</h1>

      <textarea
        className="w-full h-40 p-2 border rounded font-mono"
        value={code}
        onChange={(e) => setCode(e.target.value)}
      />

      <QLangCircuit circuit={circuit} />
    </div>
  );
}
