import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Line, Sparkles } from "@react-three/drei";
import { useRef } from "react";
import type { Group } from "three";

function PhoenixForm() {
  const form = useRef<Group>(null);
  useFrame((state, delta) => {
    if (!form.current) return;
    form.current.rotation.y += (state.pointer.x * 0.3 - form.current.rotation.y) * delta * 1.4;
    form.current.rotation.x += (-state.pointer.y * 0.12 - form.current.rotation.x) * delta * 1.2;
    form.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.35) * 0.045;
  });

  const feathers = Array.from({ length: 7 }, (_, index) => index);
  return <group ref={form} scale={1.15}>
    <mesh position={[0, 0, 0]}><icosahedronGeometry args={[0.72, 2]} /><meshStandardMaterial color="#242622" metalness={0.92} roughness={0.32} /></mesh>
    <mesh position={[0, 0.75, 0.02]} rotation={[0.15, 0, 0]}><coneGeometry args={[0.33, 0.95, 5]} /><meshStandardMaterial color="#6f1118" metalness={0.85} roughness={0.28} /></mesh>
    <mesh position={[0, 1.2, 0.08]} scale={[1, 0.8, 1]}><icosahedronGeometry args={[0.27, 1]} /><meshStandardMaterial color="#b81c25" metalness={0.72} roughness={0.25} /></mesh>
    {[-1, 1].map((side) => <group key={side}>
      {feathers.map((index) => {
        const spread = 0.52 + index * 0.31;
        const height = 0.36 + index * 0.18;
        return <mesh key={index} position={[side * spread, height, -0.06]} rotation={[0, 0, side * (-0.72 + index * 0.09)]}>
          <coneGeometry args={[0.13 + index * 0.018, 1.05 + index * 0.13, 4]} />
          <meshStandardMaterial color={index % 2 ? "#491016" : "#222522"} metalness={0.88} roughness={0.35} />
        </mesh>;
      })}
    </group>)}
    <Line points={[[-2.2, -0.7, 0], [0, -1.1, 0], [2.2, -0.7, 0]]} color="#9e1118" transparent opacity={0.35} lineWidth={0.7} />
  </group>;
}

export default function PhoenixHero() {
  return <div className="phoenix-canvas" aria-hidden="true">
    <Canvas dpr={[1, 1.5]} camera={{ position: [0, 0.15, 6], fov: 42 }} gl={{ alpha: true, antialias: true }}>
      <ambientLight intensity={0.35} /><pointLight color="#b81c25" intensity={18} position={[-3, 2, 3]} distance={8} /><pointLight color="#5c7d58" intensity={4} position={[3, -1, 2]} distance={7} />
      <Float speed={0.7} rotationIntensity={0.12} floatIntensity={0.45}><PhoenixForm /></Float>
      <Sparkles count={34} scale={5} size={1.2} speed={0.18} color="#a8aaa3" opacity={0.45} />
    </Canvas>
  </div>;
}
