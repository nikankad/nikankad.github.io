import { DotGlobe } from "dot-globe";

export default function HeroGlobe() {
  return (
    <DotGlobe
      backgroundOpacity={0}
      backgroundColor={0x0d0d0f}
      rotationSpeed={0.0009}
      tilt={[34, -18]}
      radius={12}
      cameraDistance={24}
      chars="*"
      atmosphere
      atmosphereOpacity={0.2}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
