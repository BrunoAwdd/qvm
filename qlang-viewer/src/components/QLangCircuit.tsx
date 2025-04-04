import React from "react";
import { CircuitModel } from "../qlang/parseQLang";
import { GATE_VISUALS } from "./gateVisuals";
import { getGateVisual } from "./getGateVisual";

interface Props {
  circuit: CircuitModel;
}

const QLangCircuit: React.FC<Props> = ({ circuit }) => {
  const spacingX = 60;
  const spacingY = 50;
  const gateSize = 30;

  const { qubits, gates: parsedGates } = circuit;
  const gates = parsedGates.filter((g) => getGateVisual(g.name));

  const width = (gates.length + 2) * spacingX;
  const height = (qubits + 1) * spacingY;

  gates.forEach((gate) => {
    console.log("Gate:", gate.name, gate.targets);
  });

  return (
    <svg
      width={width}
      height={height}
      className="bg-white rounded border shadow"
    >
      {/* fios */}
      {Array.from({ length: qubits }).map((_, i) => {
        const y = (i + 1) * spacingY;
        return <line key={i} x1={0} y1={y} x2={width} y2={y} stroke="#aaa" />;
      })}

      {/* portas */}
      {gates.map((gate, idx) => {
        const x = spacingX * (idx + 1);
        const visual = GATE_VISUALS[gate.name] || {
          shape: "rect",
          label: gate.name.toUpperCase(),
          color: "#ddd",
          qubits: gate.targets.length,
        };

        if (visual.shape === "connector" && gate.targets.length === 2) {
          const [c, t] = gate.targets;
          const yc = (c + 1) * spacingY;
          const yt = (t + 1) * spacingY;

          return (
            <g key={idx}>
              <line x1={x} y1={yc} x2={x} y2={yt} stroke="#000" />
              <circle cx={x} cy={yc} r={5} fill="#000" />
              <circle cx={x} cy={yt} r={10} stroke="#000" fill="none" />
            </g>
          );
        }

        // Renderização padrão
        return gate.targets.map((q, j) => {
          const y = (q + 1) * spacingY;
          if (visual.shape === "measure") {
            const y = (qubits * spacingY) / 2;
            return (
              <g key={`${idx}-${j}`}>
                <rect
                  x={x - gateSize / 3}
                  y={y}
                  width={gateSize}
                  height={gateSize}
                  fill={visual.color}
                  stroke="#333"
                  rx={4}
                />
                <text
                  x={x + 5}
                  y={y + 20}
                  textAnchor="middle"
                  fontSize={14}
                  fill={visual.textColor ?? "#000"}
                  fontFamily="monospace"
                >
                  {visual.label}
                </text>
              </g>
            );
          }
          console.log("gateSize", visual.label, y);
          return (
            <g key={`${idx}-${j}`}>
              {visual.shape === "rect" && (
                <rect
                  x={x - gateSize / 2}
                  y={y - gateSize / 2}
                  width={gateSize}
                  height={gateSize}
                  fill={visual.color}
                  stroke="#333"
                  rx={4}
                />
              )}
              {visual.shape === "circle" && (
                <circle
                  cx={x}
                  cy={y}
                  r={gateSize / 2}
                  fill={visual.color}
                  stroke="#333"
                />
              )}
              <text
                x={x}
                y={y + 5}
                textAnchor="middle"
                fontSize={14}
                fill="#333"
                fontFamily="monospace"
              >
                {visual.label}
              </text>
            </g>
          );
        });
      })}
    </svg>
  );
};

export default QLangCircuit;
