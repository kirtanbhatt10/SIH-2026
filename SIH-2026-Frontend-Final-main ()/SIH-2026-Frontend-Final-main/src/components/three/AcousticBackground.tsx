import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

const cameraPath = new THREE.CatmullRomCurve3([
  new THREE.Vector3(2.7, .8, 6), new THREE.Vector3(1.7, .25, 4),
  new THREE.Vector3(.2, 1.1, 6.6), new THREE.Vector3(-2.1, -.15, 3.8),
  new THREE.Vector3(.1, .45, 5.8),
], false, "catmullrom", .45);

function useReady(introPlaying: boolean, showSite: boolean) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    setReady(false);
    if (introPlaying || !showSite || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setTimeout(() => setReady(true), 1200);
    return () => window.clearTimeout(timer);
  }, [introPlaying, showSite]);
  return ready;
}

function CameraPath() {
  const progress = useRef(0);
  const lookAt = useRef(new THREE.Vector3(.6, 0, 0));
  const { camera } = useThree();
  useFrame((state, delta) => {
    const range = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
    progress.current = THREE.MathUtils.damp(progress.current, THREE.MathUtils.clamp(window.scrollY / range, 0, 1), 2.8, delta);
    const next = cameraPath.getPointAt(progress.current);
    camera.position.lerp(new THREE.Vector3(next.x + Math.sin(state.clock.elapsedTime * .35) * .035 + state.pointer.x * .08, next.y + Math.cos(state.clock.elapsedTime * .55) * .025 - state.pointer.y * .05, next.z), .045);
    lookAt.current.lerp(new THREE.Vector3(THREE.MathUtils.lerp(1.1, -.6, progress.current), THREE.MathUtils.lerp(.1, -.3, progress.current), 0), .045);
    camera.lookAt(lookAt.current);
  });
  return null;
}

function TransducerCore() {
  const core = useRef<THREE.Group>(null);
  const diaphragm = useRef<THREE.Mesh>(null);
  const dark = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#2a2428", metalness: .92, roughness: .18, clearcoat: .78, clearcoatRoughness: .14 }), []);
  const rim = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#791018", metalness: .8, roughness: .2, clearcoat: .58, emissive: "#1d0003", emissiveIntensity: .4 }), []);
  const signal = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#e12a35", metalness: .7, roughness: .18, emissive: "#700008", emissiveIntensity: .9 }), []);
  useFrame((state) => {
    if (!core.current) return;
    const time = state.clock.elapsedTime;
    core.current.rotation.x = .32 - state.pointer.y * .06;
    core.current.rotation.y = -.42 + time * .075 + state.pointer.x * .08;
    core.current.position.y = .1 + Math.sin(time * 1.1) * .06;
    if (diaphragm.current) diaphragm.current.position.z = .16 + Math.sin(time * 8) * .025;
  });
  return <group ref={core} position={[1.65, .1, 0]} scale={1.3}>
    <mesh rotation={[Math.PI / 2, 0, 0]} material={dark}><cylinderGeometry args={[1.72, 1.82, .25, 64]} /></mesh>
    <mesh position={[0, 0, .14]} material={rim}><torusGeometry args={[1.74, .055, 16, 64]} /></mesh>
    {[.46, .78, 1.08, 1.38].map((radius) => <mesh key={radius} position={[0, 0, .16]} material={rim}><torusGeometry args={[radius, .018, 12, 64]} /></mesh>)}
    <mesh ref={diaphragm} position={[0, 0, .16]} rotation={[Math.PI / 2, 0, 0]} material={dark}><cylinderGeometry args={[.48, .48, .06, 48]} /></mesh>
    <mesh position={[0, 0, .21]} rotation={[Math.PI / 2, 0, 0]} material={signal}><cylinderGeometry args={[.13, .13, .07, 32]} /></mesh>
  </group>;
}

function WavePulses() {
  const rings = useRef<THREE.Mesh[]>([]);
  useFrame((state) => rings.current.forEach((ring, index) => {
    if (!ring) return;
    const phase = (state.clock.elapsedTime * .22 + index / 5) % 1;
    ring.scale.setScalar(.9 + phase * 4.1);
    ring.position.z = .12 + phase * 2.9;
    (ring.material as THREE.MeshBasicMaterial).opacity = Math.max(.1, (1 - phase) * .72);
  }));
  return <group position={[1.65, .1, 0]}>{Array.from({ length: 5 }).map((_, index) => <mesh key={index} ref={(node) => { if (node) rings.current[index] = node; }}><torusGeometry args={[.82, .021, 10, 64]} /><meshBasicMaterial color="#f32b36" transparent blending={THREE.AdditiveBlending} depthWrite={false} /></mesh>)}</group>;
}

function HarmonicStream() {
  const points = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const values = new Float32Array(210 * 3);
    for (let index = 0; index < 210; index += 1) { const offset = index * 3; values[offset] = THREE.MathUtils.lerp(3, -3.5, index / 210) + (Math.random() - .5) * 1.25; values[offset + 1] = (Math.random() - .5) * 2.5; values[offset + 2] = (Math.random() - .5) * 2.2; }
    return values;
  }, []);
  useFrame((state) => {
    if (!points.current) return;
    const attribute = points.current.geometry.attributes.position as THREE.BufferAttribute;
    const values = attribute.array as Float32Array;
    for (let index = 0; index < 210; index += 1) { const offset = index * 3; values[offset] -= .012; if (values[offset] < -3.9) values[offset] = 3.25 + Math.random() * .65; values[offset + 1] += Math.sin(state.clock.elapsedTime * 1.4 + index) * .002; }
    attribute.needsUpdate = true;
  });
  return <points ref={points}><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry><pointsMaterial color="#ee2732" size={.042} transparent opacity={.72} blending={THREE.AdditiveBlending} depthWrite={false} /></points>;
}

function Scene() {
  return <><fog attach="fog" args={["#050505", 6, 22]} /><ambientLight intensity={.7} color="#f7e9e3" /><directionalLight position={[5, 7, 5]} intensity={3.8} color="#fff3e8" /><pointLight position={[1.5, .5, 2]} intensity={3.6} color="#df121d" distance={9} /><pointLight position={[-2.8, -1, 2]} intensity={.8} color="#fff0dc" distance={7} /><CameraPath /><TransducerCore /><WavePulses /><HarmonicStream /></>;
}

export default function AcousticBackground({ introPlaying, showSite }: { introPlaying: boolean; showSite: boolean }) {
  const ready = useReady(introPlaying, showSite);
  if (!ready) return null;
  return <div className="acoustic-background" aria-hidden="true"><Canvas dpr={[1, 1.5]} camera={{ position: [2.7, .8, 6], fov: 44 }} gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}><Suspense fallback={null}><Scene /></Suspense></Canvas></div>;
}
