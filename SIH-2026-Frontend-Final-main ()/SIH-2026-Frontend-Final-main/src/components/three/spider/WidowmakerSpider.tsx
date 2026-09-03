import { useFrame, useLoader, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";

export const MODEL_URL = "/models/widowmaker/model/model.dae";

const route = new THREE.CatmullRomCurve3([
  new THREE.Vector3(5.8, 3.4, -9.2), new THREE.Vector3(3.8, 2.1, -6.7),
  new THREE.Vector3(1.2, .5, -2.3), new THREE.Vector3(-1.7, -.9, -3.8),
  new THREE.Vector3(-3.7, -2.15, -5.3), new THREE.Vector3(.45, -2.5, -4.4),
  new THREE.Vector3(3.45, -.15, -5.5), new THREE.Vector3(1.4, -3.4, -3.4),
  new THREE.Vector3(-.8, -2.4, -10.2),
], false, "catmullrom", .36);

/** Supplied model moving through the fixed 3D background world, not the DOM. */
export function WidowmakerSpider() {
  const group = useRef<THREE.Group>(null);
  const progress = useRef(.015);
  const gaitMeshes = useRef<THREE.Mesh[]>([]);
  const collada = useLoader(ColladaLoader, MODEL_URL);
  const scene = useMemo(() => collada?.scene.clone(true) ?? new THREE.Group(), [collada]);
  const { gl } = useThree();

  useEffect(() => {
    const meshes: THREE.Mesh[] = [];
    let meshIndex = 0;
    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.userData.gaitBasePosition = child.position.clone();
      // The source DAE is split into separate pieces but ships with uniform grey
      // materials. Keep the model premium and recognisably widow-like: obsidian
      // on every part, with one deep-crimson abdomen piece rather than rainbow.
      const isAbdomenAccent = meshIndex === 3;
      const source = Array.isArray(child.material) ? child.material[0] : child.material;
      if (source instanceof THREE.MeshPhongMaterial) {
        const material = source.clone();
        material.color.set(isAbdomenAccent ? "#650c12" : "#1a181b");
        material.emissive.set(isAbdomenAccent ? "#2d0205" : "#030303");
        material.emissiveIntensity = isAbdomenAccent ? .2 : .065;
        material.specular.set(isAbdomenAccent ? "#b82a31" : "#555158");
        material.shininess = isAbdomenAccent ? 76 : 64;
        child.material = material;
      }
      meshes.push(child);
      meshIndex += 1;
    });
    gaitMeshes.current = meshes;
    return () => { gaitMeshes.current = []; };
  }, [scene]);

  useFrame((state, delta) => {
    if (!group.current) return;
    const scrollable = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
    const target = THREE.MathUtils.clamp(window.scrollY / scrollable, .015, .985);
    progress.current = THREE.MathUtils.damp(progress.current, target, 3.4, delta);
    const point = route.getPointAt(progress.current);
    const gaitPhase = state.clock.elapsedTime * 7.2;
    group.current.position.copy(point);
    group.current.position.y += Math.sin(gaitPhase * 2) * .045;
    // Keep the asset's body orientation stable: turning is communicated by the
    // world-space route, not by a 180° model flip at spline direction changes.
    group.current.rotation.set(0, 0, Math.sin(state.clock.elapsedTime * 7) * .012);
    group.current.scale.setScalar(.42);

    // The source asset is unrigged but split into separate mesh parts. Alternate
    // tiny lifts across those parts to sell a crawling gait without changing the
    // authored body shape or the route through the scene.
    gaitMeshes.current.forEach((mesh, index) => {
      const base = mesh.userData.gaitBasePosition as THREE.Vector3 | undefined;
      if (!base) return;
      const phase = gaitPhase + (index % 2 ? Math.PI : 0) + index * .28;
      mesh.position.copy(base);
      mesh.position.y += Math.sin(phase) * .055;
      mesh.position.x += Math.cos(phase) * .018;
    });

    // Camera-style depth of field on WebGL only. DOM UI remains crisp.
    const focusDepth = -3.1;
    const farDefocus = THREE.MathUtils.smoothstep(Math.abs(point.z - focusDepth), 1.05, 6.3);
    const nearDefocus = THREE.MathUtils.smoothstep(point.z, 1.1, 3) * .38;
    // The opening reveal is intentionally crisp; cinematic depth blur eases in
    // only after the first part of the scroll journey.
    const blurGate = THREE.MathUtils.smoothstep(progress.current, .06, .2);
    const defocus = THREE.MathUtils.clamp((farDefocus + nearDefocus) * blurGate, 0, 1);
    gl.domElement.style.filter = `blur(${(defocus * 3.5).toFixed(2)}px)`;
  });

  return <group ref={group}><primitive object={scene} /></group>;
}
