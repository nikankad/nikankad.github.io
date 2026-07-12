import { DotGlobe } from "dot-globe";

export default function HeroGlobe() {
  return (
    <DotGlobe
      backgroundOpacity={0}
      rotationSpeed={0.0003}
      tilt={[20, -18]}
      radius={10}
      cameraDistance={19}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
