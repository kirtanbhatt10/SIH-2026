import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

export type WebStrand = {
  from: THREE.Vector3;
  to: THREE.Vector3;
  createdAt: number;
  lifetime: number;
};

/** Lightweight silk strand. Expired strands are filtered by the owner. */
export function SpiderWebStrand({ strand }: { strand: WebStrand }) {
  const line = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints([strand.from, strand.to]);
    const material = new THREE.LineBasicMaterial({ color: "#d5d3ca", transparent: true, opacity: 0 });
    return new THREE.Line(geometry, material);
  }, [strand]);
  const material = useRef<THREE.LineBasicMaterial>(line.material as THREE.LineBasicMaterial);

  useFrame(({ clock }) => {
    if (!material.current) return;
    const age = (clock.getElapsedTime() - strand.createdAt) / strand.lifetime;
    material.current.opacity = Math.max(0, 0.3 * (1 - age) ** 2);
  });

  return <primitive object={line} />;
}
